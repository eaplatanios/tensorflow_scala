/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.ops.rnn.decoder

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.ops.control_flow.{ControlFlow, WhileLoopVariable}
import org.platanios.tensorflow.api.ops.rnn.RNNCell
import org.platanios.tensorflow.api.ops.variables.VariableScope
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, OpSpecification, Output}
import org.platanios.tensorflow.api.types.INT32

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
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNDecoder[T, O, OTA, S](
    val cell: RNNCell.Tuple[T] => RNNCell.Tuple[T],
    val cellOutputSize: Seq[Int],
    val initialState: Seq[Output],
    val name: String = "RNNDecoder"
)(implicit
    whileLoopEvT: WhileLoopVariable.Aux[T, _],
    whileLoopEvO: WhileLoopVariable.Aux[O, _],
    whileLoopEvOTA: WhileLoopVariable.Aux[OTA, _],
    whileLoopEvS: WhileLoopVariable.Aux[S, _]
) {
  /** Scalar `INT32` tensor representing the batch size of the input values. */
  val batchSize: Output

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

  def createZeroOutputTensorArrays(): (O, OTA)
  def writeOutputTensorArrays(time: Output, ta: OTA, o: O): OTA
  def stackOutputTensorArrays(ta: OTA): O
  def transposeOutputBatchTime(output: O): O
  def zeroOutOutputPastFinish(nextOutput: O, zeroOutput: O, finished: Output): O

  /** Passes `nextState` through as the next state depending on the corresponding value in `finished` and on its type
    * and shape. Tensor arrays and scalar states are always passed through.
    *
    * @param  nextState Next decoder state.
    * @param  state     Current decoder state.
    * @param  finished  Boolean tensor indicating whether decoding has finished for each sequence.
    * @return "Filtered" next decoder state.
    */
  def maybeCopyThroughStatePastFinish(nextState: S, state: S, finished: Output): S

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  def initialize(): (Output, T, S)

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) the decoder output for this step, (ii) the next decoder state, (iii) the next input,
    *         and (iv) a scalar `BOOLEAN` tensor specifying whether decoding has finished.
    */
  def next(time: Output, input: T, state: S): (O, S, T, Output)

  /** Performs dynamic decoding using this decoder.
    *
    * This method calls `initialize()` once and `next()` repeatedly.
    */
  def dynamicDecode(
      outputTimeMajor: Boolean = false, imputeFinished: Boolean = false,
      maximumIterations: Output = null, parallelIterations: Int = 32, swapMemory: Boolean = false,
      name: String = s"$name/DynamicRNNDecode"
  ): (O, S, Output) = {
    if (maximumIterations != null && maximumIterations.rank != 0)
      throw InvalidShapeException(s"'maximumIterations' (shape = ${maximumIterations.shape}) must be a scalar.")
    // Create a new variable scope in which the caching device is either determined by the parent scope, or is set to
    // place the cached variables using the same device placement as for the rest of the RNN.
    val currentVariableScope = VariableScope.createWithVariableScope(name)(Op.currentVariableScope)
    val cachingDevice = {
      if (currentVariableScope.cachingDevice == null)
        (opSpecification: OpSpecification) => opSpecification.device
      else
        currentVariableScope.cachingDevice
    }
    VariableScope.createWithUpdatedVariableScope(currentVariableScope, cachingDevice = cachingDevice) {
      var (initialFinished, initialInputs, initialState) = initialize()
      val (zeroOutput, initialOutputTensorArrays) = createZeroOutputTensorArrays()
      if (maximumIterations != null)
        initialFinished = Math.logicalOr(initialFinished, Math.greaterEqual(0, maximumIterations))
      val initialSequenceLengths = Basic.zerosLike(initialFinished, INT32)
      val initialTime = Basic.zeros(INT32, Shape.scalar())

      type LoopVariables = (Output, OTA, S, T, Output, Output)

      def condition(loopVariables: LoopVariables): Output = {
        Math.logicalNot(Math.all(loopVariables._5))
      }

      def body(loopVariables: LoopVariables): LoopVariables = {
        val (time, outputTensorArrays, state, input, finished, sequenceLengths) = loopVariables
        val (decoderOutput, decoderState, nextInput, decoderFinished) = next(time, input, state)
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
          Basic.fill(sequenceLengths.dataType, Basic.shape(sequenceLengths))(time + 1),
          sequenceLengths)

        // Zero out output values past finish and pass through state when appropriate
        val (nextOutput, nextState) = {
          if (imputeFinished) {
            val nextOutput = zeroOutOutputPastFinish(decoderOutput, zeroOutput, finished)
            val nextState = maybeCopyThroughStatePastFinish(decoderState, state, finished)
            (nextOutput, nextState)
          } else
            (decoderOutput, decoderState)
        }
        val nextOutputTensorArrays = writeOutputTensorArrays(time, outputTensorArrays, nextOutput)
        (time + 1, nextOutputTensorArrays, nextState, nextInput, nextFinished, nextSequenceLengths)
      }

      val (_, finalOutputTensorArrays, finalState, _, _, finalSequenceLengths): LoopVariables =
        ControlFlow.whileLoop(
          (loopVariables: LoopVariables) => condition(loopVariables),
          (loopVariables: LoopVariables) => body(loopVariables),
          (initialTime, initialOutputTensorArrays, initialState,
              initialInputs, initialFinished, initialSequenceLengths),
          parallelIterations = parallelIterations,
          swapMemory = swapMemory)

      var finalOutputs = stackOutputTensorArrays(finalOutputTensorArrays)
      // TODO: Add support for "finalize".
      if (!outputTimeMajor)
        finalOutputs = transposeOutputBatchTime(finalOutputs)
      (finalOutputs, finalState, finalSequenceLengths)
    }
  }
}
