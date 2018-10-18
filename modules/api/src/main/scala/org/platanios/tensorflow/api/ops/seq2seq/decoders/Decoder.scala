/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.ops.seq2seq.decoders

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{NestedStructure, Zero}
import org.platanios.tensorflow.api.ops.{Basic, Math, OpSpecification, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.rnn.RNN
import org.platanios.tensorflow.api.ops.rnn.cell.RNNCell
import org.platanios.tensorflow.api.ops.variables.VariableScope

import scala.language.postfixOps

/** Recurrent Neural Network (RNN) decoder abstract interface.
  *
  *  Concepts used by this interface:
  *
  *    - `input`: (structure of) tensors and tensor arrays that is passed as input to the RNN cell composing the
  *      decoder, at each time step.
  *    - `state`: Sequence of tensors that is passed to the RNN cell instance as the state.
  *    - `finished`: Boolean tensor indicating whether each sequence in the batch has finished decoding.
  *
  * @param  cell RNN cell to use for decoding.
  * @param  name Name prefix used for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Decoder[Out, State, DecOut, DecState, DecFinalOut, DecFinalState](
    val cell: RNNCell[Out, State],
    val name: String = "RNNDecoder"
)(implicit
    evZeroOut: Zero.Aux[Out, _, _, _],
    evZeroDecOut: Zero.Aux[DecOut, _, _, _],
    evStructureDecState: NestedStructure.Aux[DecState, _, _, _],
    evStructureDecFinalOut: NestedStructure.Aux[DecFinalOut, _, _, _]
) {
  /** Scalar tensor representing the batch size of the input values. */
  val batchSize: Output[Int]

  /** Describes whether the decoder keeps track of finished states.
    *
    * Most decoders will emit a true/false `finished` value independently at each time step. In this case, the
    * `dynamicDecode()` function keeps track of which batch entries have already finished, and performs a logical OR to
    * insert new batches to the finished set.
    *
    * Some decoders, however, shuffle batches/beams between time steps and `dynamicDecode()` will mix up the finished
    * state across these entries because it does not track the reshuffling across time steps. In this case, it is up to
    * the decoder to declare that it will keep track of its own finished state by setting this property to `true`.
    */
  val tracksOwnFinished: Boolean = false

  def zeroOutput: DecOut

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  def initialize(): (Output[Boolean], Out, DecState)

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) the decoder output for this step, (ii) the next decoder state, (iii) the next input,
    *         and (iv) a scalar tensor specifying whether decoding has finished.
    */
  def next(
      time: Output[Int],
      input: Out,
      state: DecState
  ): (DecOut, DecState, Out, Output[Boolean])

  /** Finalizes the output of the decoding process.
    *
    * @param  output          Final output after decoding.
    * @param  state           Final state after decoding.
    * @param  sequenceLengths Tensor containing the sequence lengths that the decoder cell outputs.
    * @return Finalized output and state to return from the decoding process.
    */
  def finalize(
      output: DecOut,
      state: DecState,
      sequenceLengths: Output[Int]
  ): (DecFinalOut, DecFinalState, Output[Int])

  /** Performs dynamic decoding using this decoder.
    *
    * This method calls `initialize()` once and `next()` repeatedly.
    */
  def decode(
      outputTimeMajor: Boolean = false,
      imputeFinished: Boolean = false,
      maximumIterations: Output[Int] = null,
      parallelIterations: Int = 32,
      swapMemory: Boolean = false,
      name: String = s"$name/DynamicRNNDecode"
  ): (DecFinalOut, DecFinalState, Output[Int]) = {
    if (maximumIterations != null && maximumIterations.rank != 0) {
      throw InvalidShapeException(
        s"'maximumIterations' (shape = ${maximumIterations.shape}) must be a scalar.")
    }
    // Create a new variable scope in which the caching device is either determined by the parent scope, or is set to
    // place the cached variables using the same device placement as for the rest of the RNN.
    val currentVariableScope = VariableScope.scope(name)(VariableScope.current)
    val cachingDevice = {
      if (currentVariableScope.cachingDevice == null)
        (opSpecification: OpSpecification) => opSpecification.device
      else
        currentVariableScope.cachingDevice
    }
    VariableScope.updatedScope(currentVariableScope, cachingDevice = cachingDevice) {
      var (initialFinished, initialInput, initialState) = initialize()
      val zeroOutput = this.zeroOutput
      val zeroOutputs = evZeroDecOut.structure.outputs(zeroOutput)
      val initialOutputTensorArrays = zeroOutputs.map(output => {
        TensorArray.create(
          size = 0,
          dynamicSize = true,
          elementShape = output.shape
        )(TF.fromDataType(output.dataType))
      })
      if (maximumIterations != null)
        initialFinished = Math.logicalOr(initialFinished, Math.greaterEqual(0, maximumIterations))
      val initialSequenceLengths = Basic.zerosLike(initialFinished).castTo[Int]
      val initialTime = Basic.zeros[Int](Shape.scalar())

      type LoopVariables = (
          Output[Int],
              Seq[TensorArray[Any]],
              DecState,
              Out,
              Output[Boolean],
              Output[Int])

      def condition(loopVariables: LoopVariables): Output[Boolean] = {
        Math.logicalNot(Math.all(loopVariables._5))
      }

      def body(loopVariables: LoopVariables): LoopVariables = {
        val (time, outputTensorArrays, state, input, finished, sequenceLengths) = loopVariables
        val (decoderOutput, decoderState, nextInput, decoderFinished) = next(time, input, state)
        val decoderOutputs = evZeroDecOut.structure.outputs(decoderOutput)
        val decoderStates = evStructureDecState.outputs(decoderState)
        var nextFinished = {
          if (tracksOwnFinished)
            decoderFinished
          else
            Math.logicalOr(decoderFinished, finished)
        }
        if (maximumIterations != null)
          nextFinished = Math.logicalOr(nextFinished, Math.greaterEqual(time + 1, maximumIterations))
        val nextSequenceLengths = Math.select(
          Math.logicalAnd(Math.logicalNot(finished), nextFinished),
          Basic.fill[Int, Long](Basic.shape(sequenceLengths))(time + 1),
          sequenceLengths)

        // Zero out output values past finish and pass through state when appropriate
        val (nextOutputs, nextStates) = {
          if (imputeFinished) {
            val nextOutputs = decoderOutputs.zip(zeroOutputs).map(o => {
              Math.select(finished, o._2, o._1)(TF.fromDataType(o._2.dataType))
            })
            // Passes `decoderStates` through as the next state depending on their corresponding value in `finished` and
            // on their type and shape. Tensor arrays and scalar states are always passed through.
            val states = evStructureDecState.outputs(state)
            val nextStates = decoderStates.zip(states).map(s => {
              s._1.setShape(s._2.shape)
              if (s._1.rank == 0) {
                s._1
              } else {
                Math.select(finished, s._2, s._1)(TF.fromDataType(s._2.dataType))
              }
            })
            (nextOutputs, nextStates)
          } else {
            (decoderOutputs, decoderStates)
          }
        }
        val nextState = evStructureDecState.decodeOutputFromOutput(state, nextStates)._1
        val nextOutputTensorArrays = outputTensorArrays.zip(nextOutputs).map(t => {
          t._1.write(time, t._2)
        })
        (time + 1, nextOutputTensorArrays, nextState, nextInput, nextFinished, nextSequenceLengths)
      }

      val (_, finalOutputTensorArrays, preFinalState, _, _, preFinalSequenceLengths): LoopVariables =
        ControlFlow.whileLoop(
          (loopVariables: LoopVariables) => condition(loopVariables),
          (loopVariables: LoopVariables) => body(loopVariables),
          (initialTime, initialOutputTensorArrays, initialState,
              initialInput, initialFinished, initialSequenceLengths),
          parallelIterations = parallelIterations,
          swapMemory = swapMemory)

      var (finalOutput, finalState, finalSequenceLengths) = finalize(
        evZeroDecOut.structure.decodeOutputFromOutput(zeroOutput, finalOutputTensorArrays.map(_.stack()))._1,
        preFinalState, preFinalSequenceLengths)

      if (!outputTimeMajor) {
        finalOutput = evStructureDecFinalOut.decodeOutputFromOutput(
          finalOutput,
          evStructureDecFinalOut.outputs(finalOutput).map(o => {
            RNN.transposeBatchTime(o)(TF.fromDataType(o.dataType))
          }))._1
      }
      (finalOutput, finalState, finalSequenceLengths)
    }
  }
}
