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
import org.platanios.tensorflow.api.core.types.{IsInt32OrInt64, IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{NestedStructure, Zero}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.rnn.RNN
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

import scala.language.postfixOps

/** Basic sampling Recurrent Neural Network (RNN) decoder.
  *
  * @param  cell             RNN cell to use for decoding.
  * @param  initialCellState Initial RNN cell state to use for starting the decoding process.
  * @param  helper           Basic RNN decoder helper to use.
  * @param  outputLayer      Output layer to use that is applied at the outputs of the provided RNN cell before
  *                          returning them.
  * @param  name             Name prefix used for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicDecoder[Out, Sample, State](
    override val cell: RNNCell[Out, State],
    val initialCellState: State,
    val helper: BasicDecoder.Helper[Out, Sample, State],
    val outputLayer: Out => Out = (o: Out) => o,
    override val name: String = "BasicRNNDecoder"
) extends Decoder[
    Out, State,
    BasicDecoder.DecoderOutput[Out, Sample], State,
    BasicDecoder.DecoderOutput[Out, Sample], State](cell, name) {
  /** Scalar tensor representing the batch size of the input values. */
  override val batchSize: Output[Int] = {
    helper.batchSize
  }

  override def zeroOutput[OV, OD, OS]()(implicit
      evStructureO: NestedStructure.Aux[Out, OV, OD, OS],
      evZeroO: Zero.Aux[Out, OS]
  ): BasicDecoder.DecoderOutput[Out, Sample] = {
    val zOutput = evZeroO.zero(batchSize, cell.outputShape, "ZeroOutput")
    val zSample = helper.zeroSample(batchSize, "ZeroSample")
    BasicDecoder.DecoderOutput(
      modelOutput = outputLayer(zOutput),
      sample = zSample)
  }

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  override def initialize[OV, OD, OS, SV, SD, SS]()(implicit
      evStructureO: NestedStructure.Aux[Out, OV, OD, OS],
      evStructureS: NestedStructure.Aux[State, SV, SD, SS]
  ): (Output[Boolean], Out, State) = {
    Op.nameScope(s"$name/Initialize") {
      val helperInitialize = helper.initialize()
      (helperInitialize._1, helperInitialize._2, initialCellState)
    }
  }

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) the decoder output, (ii) the next state, (iii) the next inputs, and (iv) a scalar
    *         tensor specifying whether sampling has finished.
    */
  override def next[OV, OD, OS, SV, SD, SS, DSV, DSD, DSS](
      time: Output[Int],
      input: Out,
      state: State
  )(implicit
      evStructureO: NestedStructure.Aux[Out, OV, OD, OS],
      evStructureS: NestedStructure.Aux[State, SV, SD, SS],
      evStructureDS: NestedStructure.Aux[State, DSV, DSD, DSS]
  ): (BasicDecoder.DecoderOutput[Out, Sample], State, Out, Output[Boolean]) = {
    Op.nameScope(s"$name/Next") {
      val nextTuple = cell(Tuple(input, state))(evStructureO, evStructureS)
      val nextTupleOutput = outputLayer(nextTuple.output)
      val sample = helper.sample(time, nextTupleOutput, nextTuple.state)
      val (finished, nextInputs, nextState) = helper.next(time, nextTupleOutput, nextTuple.state, sample)
      (BasicDecoder.DecoderOutput(nextTupleOutput, sample), nextState, nextInputs, finished)
    }
  }

  /** Finalizes the output of the decoding process.
    *
    * @param  output Final output after decoding.
    * @param  state  Final state after decoding.
    * @return Finalized output and state to return from the decoding process.
    */
  override def finalize[SV, SD, SS, DOV, DOD, DOS, DSV, DSD, DSS](
      output: BasicDecoder.DecoderOutput[Out, Sample],
      state: State,
      sequenceLengths: Output[Int]
  )(implicit
      evStructureS: NestedStructure.Aux[State, SV, SD, SS],
      evStructureDO: NestedStructure.Aux[BasicDecoder.DecoderOutput[Out, Sample], DOV, DOD, DOS],
      evStructureDS: NestedStructure.Aux[State, DSV, DSD, DSS]
  ): (BasicDecoder.DecoderOutput[Out, Sample], State, Output[Int]) = {
    (output, state, sequenceLengths)
  }
}

object BasicDecoder {
  def apply[Out, Sample, State](
      cell: RNNCell[Out, State],
      initialCellState: State,
      helper: BasicDecoder.Helper[Out, Sample, State],
      outputLayer: Out => Out = (o: Out) => o,
      name: String = "BasicRNNDecoder"
  )(implicit
      evStructureO: NestedStructure[Out],
      evStructureS: NestedStructure[Sample],
      evStructureDS: NestedStructure[State],
      evZeroOut: Zero[Out]
  ): BasicDecoder[Out, Sample, State] = {
    new BasicDecoder(cell, initialCellState, helper, outputLayer, name)
  }

  case class DecoderOutput[Out, Sample](modelOutput: Out, sample: Sample)

  /** Interface for implementing sampling helpers in sequence-to-sequence decoders. */
  trait Helper[Out, Sample, State] {
    /** Scalar tensor representing the batch size of a tensor returned by `sample()`. */
    val batchSize: Output[Int]

    /** Returns a zero-valued sample for this helper. */
    def zeroSample(
        batchSize: Output[Int],
        name: String = "ZeroSample"
    ): Sample

    /** Returns a tuple containing: (i) a scalar tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    def initialize(): (Output[Boolean], Out)

    /** Returns a sample for the provided time, input, and state. */
    def sample(
        time: Output[Int],
        input: Out,
        state: State
    ): Sample

    /** Returns a tuple containing: (i) a scalar tensor specifying whether sampling has finished, and
      * (ii) the next inputs, and (iii) the next state. */
    def next(
        time: Output[Int],
        input: Out,
        state: State,
        sample: Sample
    ): (Output[Boolean], Out, State)
  }

  /** RNN decoder helper to be used while training. It only reads inputs and the returned sample indexes are the argmax
    * over the RNN output logits. */
  case class TrainingHelper[Out, State](
      input: Out,
      sequenceLengths: Output[Int],
      timeMajor: Boolean = false,
      name: String = "RNNDecoderTrainingHelper"
  )(implicit
      evStructureO: NestedStructure[Out],
      evZeroO: Zero[Out]
  ) extends Helper[Out, Out, State] {
    if (sequenceLengths.rank != 1)
      throw InvalidShapeException(s"'sequenceLengths' (shape = ${sequenceLengths.shape}) must have rank 1.")

    private var inputs: Seq[Output[Any]] = evStructureO.outputs(input)

    private val inputTensorArrays: Seq[TensorArray[Any]] = {
      Op.nameScope(name) {
        if (!timeMajor) {
          // [B, T, D] => [T, B, D]
          inputs = inputs.map(i => {
            RNN.transposeBatchTime(i)(TF.fromDataType(i.dataType))
          })
        }
        inputs.map(input => {
          TensorArray.create(
            size = Basic.shape(input)(TF.fromDataType(input.dataType)).castTo[Int].slice(0),
            elementShape = input.shape(1 ::)
          )(TF.fromDataType(input.dataType)).unstack(input)
        })
      }
    }

    private val zeroInputs: Seq[Output[Any]] = {
      Op.nameScope(name) {
        inputs.map(input => {
          Basic.zerosLike(
            Basic.gather(
              input = input,
              indices = 0
            )(TF.fromDataType(input.dataType), TF[Int], IsInt32OrInt64[Int], IntDefault[Int], TF[Int], IsInt32OrInt64[Int]))
        })
      }
    }

    /** Scalar tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: Output[Int] = {
      Op.nameScope(name) {
        Basic.size(sequenceLengths).castTo[Int]
      }
    }

    /** Returns a zero-valued sample for this helper. */
    def zeroSample(
        batchSize: Output[Int],
        name: String = "ZeroSample"
    ): Out = {
      val shapes = evStructureO.outputs(input).map(_ => Shape.scalar())
      val shape = evStructureO.decodeShapeFromOutput(input, shapes)._1
      evZeroO.zero(batchSize, shape.asInstanceOf[evZeroO.S])
    }

    /** Returns a tuple containing: (i) a scalar tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (Output[Boolean], Out) = {
      Op.nameScope(s"$name/Initialize") {
        val finished = Math.equal(0, sequenceLengths)
        val nextInputs = ControlFlow.cond(
          Math.all(finished),
          () => zeroInputs,
          () => inputTensorArrays.map(_.read(0)))
        (finished, evStructureO.decodeOutputFromOutput(input, nextInputs)._1)
      }
    }

    /** Returns a sample for the provided time, input, and state. */
    override def sample(
        time: Output[Int],
        input: Out,
        state: State
    ): Out = {
      val outputs = evStructureO.outputs(input)
      Op.nameScope(s"$name/Sample") {
        evStructureO.decodeOutputFromOutput(input, outputs.map(output => {

          // TODO: [TYPES] !!! Super hacky. Remove in the future.
          val ev: IsNotQuantized[Any] = null

          Math.argmax(
            output,
            axes = -1,
            outputDataType = Int
          )(TF.fromDataType(output.dataType), ev, TF[Int], IsInt32OrInt64[Int], TF[Int])
        }))._1
      }
    }

    /** Returns a tuple containing: (i) a scalar tensor specifying whether sampling has finished, and
      * (ii) the next inputs, and (iii) the next state. */
    override def next(
        time: Output[Int],
        input: Out,
        state: State,
        sample: Out
    ): (Output[Boolean], Out, State) = {
      Op.nameScope(s"$name/NextInputs") {
        val nextTime = time + 1
        val finished = Math.greaterEqual(nextTime, sequenceLengths)
        val nextInputs = ControlFlow.cond(
          Math.all(finished),
          () => zeroInputs,
          () => inputTensorArrays.map(_.read(nextTime)))
        (finished, evStructureO.decodeOutputFromOutput(input, nextInputs)._1, state)
      }
    }
  }

  /** RNN decoder helper to be used while performing inference. It uses the argmax over the RNN output logits and passes
    * the result through an embedding layer to get the next input.
    *
    * @param  embeddingFn Function that takes an vector of IDs and returns the corresponding embedded values
    *                     that will be passed to the decoder input.
    * @param  beginTokens Vector with length equal to the batch size, which contains the begin-of-sequence token IDs.
    * @param  endToken    Scalar containing the end-of-sequence token ID (i.e., token ID which marks the end of
    *                     decoding).
    */
  case class GreedyEmbeddingHelper[T, State](
      embeddingFn: Output[Int] => Output[T],
      beginTokens: Output[Int],
      endToken: Output[Int],
      name: String = "RNNDecoderGreedyEmbeddingHelper"
  ) extends Helper[Output[T], Output[Int], State] {
    if (beginTokens.rank != 1)
      throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
    if (endToken.rank != 0)
      throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

    private val beginInputs: Output[T] = {
      Op.nameScope(name) {
        embeddingFn(beginTokens)
      }
    }

    /** Scalar tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: Output[Int] = {
      Op.nameScope(name) {
        Basic.size(beginTokens).castTo[Int]
      }
    }

    /** Returns a zero-valued sample for this helper. */
    def zeroSample(
        batchSize: Output[Int],
        name: String = "ZeroSample"
    ): Output[Int] = {
      Op.nameScope(name) {
        Basic.fill[Int, Int](batchSize.expandDims(0))(0)
      }
    }

    /** Returns a tuple containing: (i) a scalar tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (Output[Boolean], Output[T]) = {
      Op.nameScope(s"$name/Initialize") {
        (Basic.tile(Tensor(false), batchSize.expandDims(0)), beginInputs)
      }
    }

    /** Returns a sample for the provided time, input, and state. */
    override def sample(
        time: Output[Int],
        input: Output[T],
        state: State
    ): Output[Int] = {
      Op.nameScope(s"$name/Sample") {

        // TODO: [TYPES] !!! Super hacky. Remove in the future.
        implicit val ev: IsNotQuantized[T] = null
        implicit val evTF: TF[T] = TF.fromDataType(input.dataType)

        Math.argmax(input, axes = -1, outputDataType = endToken.dataType)
      }
    }

    /** Returns a tuple containing: (i) a scalar tensor specifying whether sampling has finished, and
      * (ii) the next RNN cell tuple. */
    override def next(
        time: Output[Int],
        input: Output[T],
        state: State,
        sample: Output[Int]
    ): (Output[Boolean], Output[T], State) = {
      Op.nameScope(s"$name/NextInputs") {
        val finished = Math.equal(sample, endToken)
        val nextInputs = ControlFlow.cond(
          Math.all(finished),
          // If we are finished, the next inputs value does not matter.
          () => beginInputs,
          () => embeddingFn(sample))
        (finished, nextInputs, state)
      }
    }
  }
}
