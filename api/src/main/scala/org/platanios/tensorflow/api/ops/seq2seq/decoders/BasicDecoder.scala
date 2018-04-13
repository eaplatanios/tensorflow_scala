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
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.{ControlFlow, WhileLoopVariable}
import org.platanios.tensorflow.api.ops.rnn.RNN
import org.platanios.tensorflow.api.ops.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32}

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
class BasicDecoder[O, OS, S, SS](
    override val cell: RNNCell[O, OS, S, SS],
    val initialCellState: S,
    val helper: BasicDecoder.Helper[O, S],
    val outputLayer: O => O = (o: O) => o,
    override val name: String = "BasicRNNDecoder"
)(implicit
    evO: WhileLoopVariable.Aux[O, OS],
    evS: WhileLoopVariable.Aux[S, SS]
) extends Decoder[
    O, OS, S, SS,
    BasicDecoder.Output[O, OS], (OS, OS), S, SS,
    BasicDecoder.Output[O, OS], S](cell, name) {
  /** Scalar `INT32` tensor representing the batch size of the input values. */
  override val batchSize: Output = helper.batchSize

  override def zeroOutput(): BasicDecoder.Output[O, OS] = {
    val dataType = evS.outputs(initialCellState).head.dataType
    val zOutput = evO.zero(batchSize, dataType, cell.outputShape, "ZeroOutput")
    val zSample = helper.zeroSample(batchSize, "ZeroSample")
    BasicDecoder.Output(outputLayer(zOutput), zSample)
  }

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  override def initialize(): (Output, O, S) = Op.createWithNameScope(s"$name/Initialize") {
    val helperInitialize = helper.initialize()
    (helperInitialize._1, helperInitialize._2, initialCellState)
  }

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
    *         (ii) the next RNN cell tuple.
    */
  override def next(time: Output, input: O, state: S): (BasicDecoder.Output[O, OS], S, O, Output) = {
    val inputs = evO.outputs(input)
    val states = evS.outputs(state)
    Op.createWithNameScope(s"$name/Next", Set(time.op) ++ inputs.map(_.op).toSet ++ states.map(_.op).toSet) {
      val nextTuple = cell(Tuple(input, state))
      val nextTupleOutput = outputLayer(nextTuple.output)
      val sample = helper.sample(time, nextTupleOutput, nextTuple.state)
      val (finished, nextInputs, nextState) = helper.next(time, nextTupleOutput, nextTuple.state, sample)
      (BasicDecoder.Output(nextTupleOutput, sample), nextState, nextInputs, finished)
    }
  }

  /** Finalizes the output of the decoding process.
    *
    * @param  output Final output after decoding.
    * @param  state  Final state after decoding.
    * @return Finalized output and state to return from the decoding process.
    */
  override def finalize(
      output: BasicDecoder.Output[O, OS],
      state: S,
      sequenceLengths: Output
  ): (BasicDecoder.Output[O, OS], S, Output) = {
    (output, state, sequenceLengths)
  }
}

object BasicDecoder {
  def apply[O, OS, S, SS](
      cell: RNNCell[O, OS, S, SS],
      initialCellState: S,
      helper: BasicDecoder.Helper[O, S],
      outputLayer: O => O = (o: O) => o,
      name: String = "BasicRNNDecoder"
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS]
  ): BasicDecoder[O, OS, S, SS] = {
    new BasicDecoder[O, OS, S, SS](cell, initialCellState, helper, outputLayer, name)
  }

  case class Output[O, OS](rnnOutput: O, sample: O)(implicit whileLoopEvO: WhileLoopVariable.Aux[O, OS])

  object Output {
    implicit def outputWhileLoopVariable[O, OS](implicit
        whileLoopEvO: WhileLoopVariable.Aux[O, OS]
    ): WhileLoopVariable.Aux[Output[O, OS], (OS, OS)] = new WhileLoopVariable[Output[O, OS]] {
      override type ShapeType = (OS, OS)

      override def zero(batchSize: ops.Output, dataType: DataType, shape: (OS, OS), name: String): Output[O, OS] = {
        Output(whileLoopEvO.zero(batchSize, dataType, shape._1), whileLoopEvO.zero(batchSize, dataType, shape._2))
      }

      override def size(output: Output[O, OS]): Int = {
        whileLoopEvO.size(output.rnnOutput) + whileLoopEvO.size(output.sample)
      }

      override def outputs(output: Output[O, OS]): Seq[ops.Output] = {
        whileLoopEvO.outputs(output.rnnOutput) ++ whileLoopEvO.outputs(output.sample)
      }

      override def shapes(shape: (OS, OS)): Seq[Shape] = {
        whileLoopEvO.shapes(shape._1) ++ whileLoopEvO.shapes(shape._2)
      }

      override def segmentOutputs(output: Output[O, OS], values: Seq[ops.Output]): (Output[O, OS], Seq[ops.Output]) = {
        val (rnnOutput, sampleAndTail) = whileLoopEvO.segmentOutputs(output.rnnOutput, values)
        val (sample, tail) = whileLoopEvO.segmentOutputs(output.sample, sampleAndTail)
        (Output(rnnOutput, sample), tail)
      }

      override def segmentShapes(output: Output[O, OS], values: Seq[Shape]): ((OS, OS), Seq[Shape]) = {
        val (rnnOutput, sampleAndTail) = whileLoopEvO.segmentShapes(output.rnnOutput, values)
        val (sample, tail) = whileLoopEvO.segmentShapes(output.sample, sampleAndTail)
        ((rnnOutput, sample), tail)
      }

      override def map(value: Output[O, OS], mapFn: ops.Symbol => ops.Symbol): Output[O, OS] = {
        Output(
          whileLoopEvO.map(value.rnnOutput, mapFn),
          whileLoopEvO.map(value.sample, mapFn))
      }

      override def mapWithShape(
          value: Output[O, OS], shape: (OS, OS), mapFn: (ops.Symbol, Shape) => ops.Symbol
      ): Output[O, OS] = {
        Output(
          whileLoopEvO.mapWithShape(value.rnnOutput, shape._1, mapFn),
          whileLoopEvO.mapWithShape(value.sample, shape._2, mapFn))
      }
    }
  }

  /** Interface for implementing sampling helpers in sequence-to-sequence decoders. */
  trait Helper[O, S] {
    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    val batchSize: ops.Output

    /** Returns a zero-valued sample for this helper. */
    def zeroSample(batchSize: ops.Output, name: String = "ZeroSample"): O

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    def initialize(): (ops.Output, O)

    /** Returns a sample for the provided time, input, and state. */
    def sample(time: ops.Output, input: O, state: S): O

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
      * (ii) the next inputs, and (iii) the next state. */
    def next(time: ops.Output, input: O, state: S, sample: O): (ops.Output, O, S)
  }

  /** RNN decoder helper to be used while training. It only reads inputs and the returned sample indexes are the argmax
    * over the RNN output logits. */
  case class TrainingHelper[O, OS, S, SS](
      input: O,
      sequenceLengths: ops.Output,
      timeMajor: Boolean = false,
      name: String = "RNNDecoderTrainingHelper"
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS]
  ) extends Helper[O, S] {
    if (sequenceLengths.rank != 1)
      throw InvalidShapeException(s"'sequenceLengths' (shape = ${sequenceLengths.shape}) must have rank 1.")

    private[this] var inputs           : Seq[ops.Output]  = evO.outputs(input)
    private[this] val inputTensorArrays: Seq[TensorArray] = Op.createWithNameScope(name, inputs.map(_.op).toSet) {
      if (!timeMajor) {
        // [B, T, D] => [T, B, D]
        inputs = inputs.map(RNN.transposeBatchTime)
      }
      inputs.map(input => {
        TensorArray.create(Basic.shape(input)(0), input.dataType, elementShape = input.shape(1 ::)).unstack(input)
      })
    }

    private[this] val zeroInputs: Seq[ops.Output] = Op.createWithNameScope(name, inputs.map(_.op).toSet) {
      inputs.map(input => Basic.zerosLike(input.gather(0)))
    }

    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: ops.Output = Op.createWithNameScope(name, Set(sequenceLengths.op)) {
      Basic.size(sequenceLengths)
    }

    /** Returns a zero-valued sample for this helper. */
    def zeroSample(batchSize: ops.Output, name: String = "ZeroSample"): O = {
      evO.zero(batchSize, INT32, evO.fromShapes(input, evO.outputs(input).map(_ => Shape.scalar())))
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (ops.Output, O) = {
      Op.createWithNameScope(s"$name/Initialize") {
        val finished = Math.equal(0, sequenceLengths)
        val nextInputs = ControlFlow.cond(
          Math.all(finished),
          () => zeroInputs,
          () => inputTensorArrays.map(_.read(0)))
        (finished, evO.fromOutputs(input, nextInputs))
      }
    }

    /** Returns a sample for the provided time, input, and state. */
    override def sample(time: ops.Output, input: O, state: S): O = {
      val outputs = evO.outputs(input)
      Op.createWithNameScope(s"$name/Sample", Set(time.op) ++ outputs.map(_.op).toSet) {
        evO.fromOutputs(input, outputs.map(output => Math.argmax(output, axes = -1, outputDataType = INT32)))
      }
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
      * (ii) the next inputs, and (iii) the next state. */
    override def next(
        time: ops.Output, input: O, state: S, sample: O): (ops.Output, O, S) = {
      val inputs = evO.outputs(input)
      val states = evS.outputs(state)
      val samples = evO.outputs(sample)
      Op.createWithNameScope(
        s"$name/NextInputs",
        Set(time.op) ++ inputs.map(_.op).toSet ++ states.map(_.op).toSet ++ samples.map(_.op).toSet
      ) {
        val nextTime = time + 1
        val finished = Math.greaterEqual(nextTime, sequenceLengths)
        val nextInputs = ControlFlow.cond(
          Math.all(finished),
          () => zeroInputs,
          () => inputTensorArrays.map(_.read(nextTime)))
        (finished, evO.fromOutputs(input, nextInputs), state)
      }
    }
  }

  /** RNN decoder helper to be used while performing inference. It uses the argmax over the RNN output logits and passes
    * the result through an embedding layer to get the next input.
    *
    * @param  embeddingFn Function that takes an `INT32` vector of IDs and returns the corresponding embedded values
    *                     that will be passed to the decoder input.
    * @param  beginTokens `INT32` vector with length equal to the batch size, which contains the begin-of-sequence
    *                      token IDs.
    * @param  endToken    `INT32` scalar containing the end-of-sequence token ID (i.e., token ID which marks the end of
    *                     decoding).
    */
  case class GreedyEmbeddingHelper[S](
      embeddingFn: ops.Output => ops.Output,
      beginTokens: ops.Output,
      endToken: ops.Output,
      name: String = "RNNDecoderGreedyEmbeddingHelper"
  ) extends Helper[ops.Output, S] {
    if (beginTokens.rank != 1)
      throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
    if (endToken.rank != 0)
      throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

    private[this] val beginInputs: ops.Output = Op.createWithNameScope(name, Set(beginTokens.op)) {
      embeddingFn(beginTokens)
    }

    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: ops.Output = Op.createWithNameScope(name, Set(beginTokens.op)) {
      Basic.size(beginTokens)
    }

    /** Returns a zero-valued sample for this helper. */
    def zeroSample(batchSize: ops.Output, name: String = "ZeroSample"): ops.Output = Op.createWithNameScope(name) {
      Basic.fill(INT32, batchSize.expandDims(0))(0, name)
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (ops.Output, ops.Output) = {
      Op.createWithNameScope(s"$name/Initialize") {
        (Basic.tile(Tensor(false), batchSize.expandDims(0)), beginInputs)
      }
    }

    /** Returns a sample for the provided time, input, and state. */
    override def sample(time: ops.Output, input: ops.Output, state: S): ops.Output = {
      Op.createWithNameScope(s"$name/Sample", Set(time.op, input.op)) {
        Math.argmax(input, axes = -1, outputDataType = INT32)
      }
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
      * (ii) the next RNN cell tuple. */
    override def next(
        time: ops.Output,
        input: ops.Output,
        state: S,
        sample: ops.Output
    ): (ops.Output, ops.Output, S) = {
      Op.createWithNameScope(s"$name/NextInputs", Set(time.op, input.op, sample.op)) {
        val finished = Math.equal(sample, endToken)
        val nextInputs = ControlFlow.cond(
          Math.all(finished),
          // If we are finished, the next inputs value does not matter
          () => beginInputs,
          () => embeddingFn(sample))
        (finished, nextInputs, state)
      }
    }
  }
}
