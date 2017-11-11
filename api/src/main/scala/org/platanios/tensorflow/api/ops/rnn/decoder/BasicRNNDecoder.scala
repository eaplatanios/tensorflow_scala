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
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, TensorArray}
import org.platanios.tensorflow.api.ops.control_flow.{CondOutput, ControlFlow, WhileLoopVariable}
import org.platanios.tensorflow.api.ops.rnn.{RNN, RNNCell}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32}

import scala.language.postfixOps

/** Basic sampling Recurrent Neural Network (RNN) decoder.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicRNNDecoder[T: RNNCell.Output](
    override val cell: RNNCell.Tuple[T] => RNNCell.Tuple[T],
    override val cellOutputSize: Seq[Int],
    override val initialState: Seq[Output],
    val helper: BasicRNNDecoder.Helper[T],
    override val name: String = "BasicRNNDecoder"
)(implicit
    whileLoopEvT: WhileLoopVariable.Aux[T, _],
    whileLoopEvOutputT: WhileLoopVariable.Aux[BasicRNNDecoder.Output[T], _],
    whileLoopEvOutputSeqTensorArray: WhileLoopVariable.Aux[BasicRNNDecoder.Output[Seq[TensorArray]], _]
) extends RNNDecoder[T, BasicRNNDecoder.Output[T], BasicRNNDecoder.Output[Seq[TensorArray]], Seq[Output]](
  cell,
  cellOutputSize,
  initialState,
  name
) {
  /** Scalar `INT32` tensor representing the batch size of the input values. */
  override val batchSize: Output = helper.batchSize

  override def createZeroOutputTensorArrays(): (BasicRNNDecoder.Output[T], BasicRNNDecoder.Output[Seq[TensorArray]]) = {
    val zeroOutputs = cellOutputSize.map(s => Basic.fill(initialState.head.dataType, Basic.stack(Seq(batchSize, s)))(0))
    val rnnCellZeroOutput = RNNCell.Output[T].fromOutputs(zeroOutputs)
    val rnnCellTensorArrays = zeroOutputs.map(output => {
      TensorArray.create(0, output.dataType, dynamicSize = true, elementShape = output.shape)
    })
    val zeroOutput = BasicRNNDecoder.Output[T](rnnCellZeroOutput, rnnCellZeroOutput)
    val tensorArrays = BasicRNNDecoder.Output[Seq[TensorArray]](rnnCellTensorArrays, rnnCellTensorArrays)
    (zeroOutput, tensorArrays)
  }

  def writeOutputTensorArrays(
      time: Output,
      ta: BasicRNNDecoder.Output[Seq[TensorArray]],
      o: BasicRNNDecoder.Output[T]
  ): BasicRNNDecoder.Output[Seq[TensorArray]] = {
    BasicRNNDecoder.Output[Seq[TensorArray]](
      RNNCell.Output[T].outputs(o.rnnOutput)
          .zip(ta.rnnOutput)
          .map(t => t._2.write(time, t._1)),
      RNNCell.Output[T].outputs(o.sample)
          .zip(ta.sample)
          .map(t => t._2.write(time, t._1)))
  }

  def stackOutputTensorArrays(ta: BasicRNNDecoder.Output[Seq[TensorArray]]): BasicRNNDecoder.Output[T] = {
    val rnnOutput = ta.rnnOutput.map(_.stack())
    val sample = ta.sample.map(_.stack())
    BasicRNNDecoder.Output[T](RNNCell.Output[T].fromOutputs(rnnOutput), RNNCell.Output[T].fromOutputs(sample))
  }

  def transposeOutputBatchTime(output: BasicRNNDecoder.Output[T]): BasicRNNDecoder.Output[T] = {
    val rnnOutput = RNNCell.Output[T].outputs(output.rnnOutput).map(RNN.transposeBatchTime)
    val sample = RNNCell.Output[T].outputs(output.sample).map(RNN.transposeBatchTime)
    BasicRNNDecoder.Output[T](RNNCell.Output[T].fromOutputs(rnnOutput), RNNCell.Output[T].fromOutputs(sample))
  }

  def zeroOutOutputPastFinish(
      nextOutput: BasicRNNDecoder.Output[T],
      zeroOutput: BasicRNNDecoder.Output[T],
      finished: Output
  ): BasicRNNDecoder.Output[T] = {
    val zeroRnnOutput = RNNCell.Output[T].outputs(zeroOutput.rnnOutput)
    val nextRnnOutput = RNNCell.Output[T].outputs(nextOutput.rnnOutput)
    val rnnOutput = nextRnnOutput.zip(zeroRnnOutput).map(o => Math.select(finished, o._2, o._1))
    val zeroSample = RNNCell.Output[T].outputs(zeroOutput.sample)
    val nextSample = RNNCell.Output[T].outputs(nextOutput.sample)
    val sample = nextSample.zip(zeroSample).map(o => Math.select(finished, o._2, o._1))
    BasicRNNDecoder.Output[T](RNNCell.Output[T].fromOutputs(rnnOutput), RNNCell.Output[T].fromOutputs(sample))
  }

  /** Passes `nextState` through as the next state depending on the corresponding value in `finished` and on its type
    * and shape. Tensor arrays and scalar states are always passed through.
    *
    * @param  nextState Next decoder state.
    * @param  state     Current decoder state.
    * @param  finished  Boolean tensor indicating whether decoding has finished for each sequence.
    * @return "Filtered" next decoder state.
    */
  def maybeCopyThroughStatePastFinish(nextState: Seq[Output], state: Seq[Output], finished: Output): Seq[Output] = {
    nextState.zip(state).map(s => {
      s._1.setShape(s._2.shape)
      if (s._1.rank == 0)
        s._1
      else
        Math.select(finished, s._2, s._1)
    })
  }

  /** This method is called before any decoding iterations. It computes the initial input values and the initial state.
    *
    * @return Tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished,
    *         (ii) the next input, and (iii) the initial decoder state.
    */
  override def initialize(): (Output, T, Seq[Output]) = Op.createWithNameScope(s"$name/Initialize") {
    val helperInitialize = helper.initialize()
    (helperInitialize._1, helperInitialize._2, initialState)
  }

  /** This method is called once per step of decoding (but only once for dynamic decoding).
    *
    * @return Tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
    *         (ii) the next RNN cell tuple.
    */
  override def next(time: Output, input: T, state: Seq[Output]): (BasicRNNDecoder.Output[T], Seq[Output], T, Output) = {
    val inputs = RNNCell.Output[T].outputs(input)
    Op.createWithNameScope(s"$name/Step", Set(time.op) ++ inputs.map(_.op).toSet ++ state.map(_.op).toSet) {
      val nextTuple = cell(RNNCell.Tuple(input, state))
      val sample = helper.sample(time, nextTuple.output, nextTuple.state)
      val (finished, nextInputs, nextState) = helper.next(time, nextTuple.output, nextTuple.state, sample)
      (BasicRNNDecoder.Output[T](nextTuple.output, sample), nextState, nextInputs, finished)
    }
  }
}

object BasicRNNDecoder {
  case class Output[T](rnnOutput: T, sample: T)(implicit whileLoopEvT: WhileLoopVariable.Aux[T, _])

  /** Interface for implementing sampling helpers in sequence-to-sequence decoders. */
  trait Helper[T] {
    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    val batchSize: ops.Output

    /** Shape of tensor returned by `sample()`, excluding the batch dimension. */
    val sampleShape: Shape

    /** Data type of tensor returned by `sample()`. */
    val sampleDataType: DataType

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    def initialize(): (ops.Output, T)

    /** Returns a sample for the provided time, input, and state. */
    def sample(time: ops.Output, input: T, state: Seq[ops.Output]): T

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
      * (ii) the next inputs, and (iii) the next state. */
    def next(time: ops.Output, input: T, state: Seq[ops.Output], sample: T): (ops.Output, T, Seq[ops.Output])
  }

  /** RNN decoder helper to be used while training. It only reads inputs and the returned sample indexes are the argmax
    * over the RNN output logits. */
  case class TrainingHelper[T: RNNCell.Output](
      input: T,
      sequenceLengths: ops.Output,
      timeMajor: Boolean = false,
      name: String = "RNNDecoderTrainingHelper"
  )(implicit
    condEvT: CondOutput.Aux[T, _]
  ) extends Helper[T] {
    if (sequenceLengths.rank != 1)
      throw InvalidShapeException(s"'sequenceLengths' (shape = ${sequenceLengths.shape}) must have rank 1.")

    private[this] var inputs           : Seq[ops.Output]      = RNNCell.Output[T].outputs(input)
    private[this] val inputTensorArrays: Seq[TensorArray] = Op.createWithNameScope(name, inputs.map(_.op).toSet) {
      if (!timeMajor) {
        // [B, T, D] => [T, B, D]
        inputs = inputs.map(RNN.transposeBatchTime)
      }
      inputs.map(input => {
        TensorArray.create(Basic.shape(input)(0), input.dataType, elementShape = input.shape(1 ::)).unstack(input)
      })
    }

    private[this] val zeroInput: T = Op.createWithNameScope(name, inputs.map(_.op).toSet) {
      RNNCell.Output[T].fromOutputs(inputs.map(input => Basic.zerosLike(input(0))))
    }

    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: ops.Output = Op.createWithNameScope(name, Set(sequenceLengths.op)) {
      Basic.size(sequenceLengths)
    }

    /** Shape of tensor returned by `sample()`, excluding the batch dimension. */
    override val sampleShape: Shape = Shape.scalar()

    /** Data type of tensor returned by `sample()`. */
    override val sampleDataType: DataType = INT32

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (ops.Output, T) = {
      Op.createWithNameScope(s"$name/Initialize") {
        val finished = Math.all(Math.equal(0, sequenceLengths))
        val nextInputs = ControlFlow.cond(
          finished,
          () => zeroInput,
          () => RNNCell.Output[T].fromOutputs(inputTensorArrays.map(_.read(0))))
        (finished, nextInputs)
      }
    }

    /** Returns a sample for the provided time, input, and state. */
    override def sample(time: ops.Output, input: T, state: Seq[ops.Output]): T = {
      val outputs = RNNCell.Output[T].outputs(input)
      Op.createWithNameScope(s"$name/Sample", Set(time.op) ++ outputs.map(_.op).toSet) {
        RNNCell.Output[T].fromOutputs(outputs.map(output => Math.cast(Math.argmax(output, axes = -1), INT32)))
      }
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
      * (ii) the next inputs, and (iii) the next state. */
    override def next(
        time: ops.Output, input: T, state: Seq[ops.Output], sample: T): (ops.Output, T, Seq[ops.Output]) = {
      val inputs = RNNCell.Output[T].outputs(input)
      val samples = RNNCell.Output[T].outputs(sample)
      Op.createWithNameScope(
        s"$name/NextInputs",
        Set(time.op) ++ inputs.map(_.op).toSet ++ state.map(_.op).toSet ++ samples.map(_.op).toSet
      ) {
        val nextTime = time + 1
        val finished = Math.all(Math.greaterEqual(nextTime, sequenceLengths))
        val nextInputs = ControlFlow.cond(
          finished,
          () => zeroInput,
          () => RNNCell.Output[T].fromOutputs(inputTensorArrays.map(_.read(nextTime))))
        (finished, nextInputs, state)
      }
    }
  }

  /** RNN decoder helper to be used while performing inference. It uses the argmax over the RNN output logits and passes
    * the result through an embedding layer to get the next input.
    *
    * @param  embeddingFn Function that takes an `INT32` vector of IDs and returns the corresponding embedded values
    *                     that will be passed to the decoder input.
    * @param  beginTokens `INT32` vector with length equal to the batch size, which contains the being token IDs.
    * @param  endToken    `INT32` scalar containing the end token ID (i.e., token ID which marks the end of decoding).
    */
  case class GreedyEmbeddingHelper(
      embeddingFn: (ops.Output) => ops.Output,
      beginTokens: ops.Output,
      endToken: ops.Output,
      name: String = "RNNDecoderGreedyEmbeddingHelper"
  ) extends Helper[ops.Output] {
    if (beginTokens.rank != 1)
      throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
    if (endToken.rank != 0)
      throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

    private[this] val beginInputs: ops.Output = embeddingFn(beginTokens)

    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: ops.Output = Op.createWithNameScope(name, Set(beginTokens.op)) {
      Basic.size(beginTokens)
    }

    /** Shape of tensor returned by `sample()`, excluding the batch dimension. */
    override val sampleShape: Shape = Shape.scalar()

    /** Data type of tensor returned by `sample()`. */
    override val sampleDataType: DataType = INT32

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (ops.Output, ops.Output) = {
      Op.createWithNameScope(s"$name/Initialize") {
        (Basic.tile(Tensor(false), batchSize.expandDims(0)), beginInputs)
      }
    }

    /** Returns a sample for the provided time, input, and state. */
    override def sample(time: ops.Output, input: ops.Output, state: Seq[ops.Output]): ops.Output = {
      Op.createWithNameScope(s"$name/Sample", Set(time.op, input.op)) {
        Math.cast(Math.argmax(input, axes = -1), INT32)
      }
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished, and
      * (ii) the next RNN cell tuple. */
    override def next(
        time: ops.Output,
        input: ops.Output,
        state: Seq[ops.Output],
        sample: ops.Output
    ): (ops.Output, ops.Output, Seq[ops.Output]) = {
      Op.createWithNameScope(
        s"$name/NextInputs", Set(time.op, input.op, sample.op) ++ state.map(_.op).toSet) {
        val finished = Math.all(Math.equal(sample, endToken))
        val nextInputs = ControlFlow.cond(
          finished,
          // If we are finished, the next inputs value does not matter
          () => beginInputs,
          () => embeddingFn(sample))
        (finished, nextInputs, state)
      }
    }
  }
}
