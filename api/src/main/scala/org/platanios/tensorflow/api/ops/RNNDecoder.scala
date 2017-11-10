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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32}

import scala.language.postfixOps

/**
  * @author Emmanouil Antonios Platanios
  */
object RNNDecoder {
  /** Interface for implementing sampling helpers in sequence-to-sequence decoders. */
  trait Helper[T] {
    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    val batchSize: Output

    /** Shape of tensor returned by `sample()`, excluding the batch dimension. */
    val sampleShape: Shape

    /** Data type of tensor returned by `sample()`. */
    val sampleDataType: DataType

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    def initialize(): (Output, T)

    /** Returns a sample for the provided `time`, `outputs`, and `state`. */
    def sample(time: Output, outputs: Output, state: Output): Output

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished,
      * (ii) the next input, and (iii) a tensor containing the next state. */
    def nextInput(time: Output, outputs: Output, state: Output, sample: Output): (Output, T, Output)
  }

  /** RNN decoder helper to be used while training. It only reads inputs and the returned sample indexes are the argmax
    * over the RNN output logits. */
  case class TrainingHelper[T: RNNCell.Output](
      input: T,
      sequenceLengths: Output,
      timeMajor: Boolean = false,
      name: String = "RNNDecoderTrainingHelper"
  ) extends Helper[T] {
    if (sequenceLengths.rank != 1)
      throw InvalidShapeException(s"'sequenceLengths' (shape = ${sequenceLengths.shape}) must have rank 1.")

    private[this] var inputs           : Seq[Output]      = RNNCell.Output[T].outputs(input)
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
      RNNCell.Output[T].fromOutputs(inputs.map(input => Basic.zerosLike(input(0, ::))))
    }

    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: Output = Op.createWithNameScope(name, Set(sequenceLengths.op)) {
      Basic.size(sequenceLengths)
    }

    /** Shape of tensor returned by `sample()`, excluding the batch dimension. */
    override val sampleShape: Shape = Shape.scalar()

    /** Data type of tensor returned by `sample()`. */
    override val sampleDataType: DataType = INT32

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (Output, T) = {
      Op.createWithNameScope(s"$name/Initialize") {
        val finished = Math.all(Math.equal(0, sequenceLengths))
        val nextInputs = ControlFlow.cond(
          finished,
          () => zeroInput,
          () => RNNCell.Output[T].fromOutputs(inputTensorArrays.map(_.read(0))))
        (finished, nextInputs)
      }
    }

    /** Returns a sample for the provided `time`, `outputs`, and `state`. */
    override def sample(time: Output, outputs: Output, state: Output): Output = {
      Op.createWithNameScope(s"$name/Sample", Set(time.op, outputs.op)) {
        Math.cast(Math.argmax(outputs, axes = -1), INT32)
      }
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished,
      * (ii) the next input, and (iii) a tensor containing the next state. */
    override def nextInput(time: Output, outputs: Output, state: Output, sample: Output): (Output, T, Output) = {
      Op.createWithNameScope(s"$name/NextInputs", Set(time.op, outputs.op, state.op)) {
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
      embeddingFn: (Output) => Output,
      beginTokens: Output,
      endToken: Output,
      name: String = "RNNDecoderGreedyEmbeddingHelper"
  ) extends Helper[Output] {
    if (beginTokens.rank != 1)
      throw InvalidShapeException(s"'beginTokens' (shape = ${beginTokens.shape}) must have rank 1.")
    if (endToken.rank != 0)
      throw InvalidShapeException(s"'endToken' (shape = ${endToken.shape}) must have rank 0.")

    private[this] val beginInputs: Output = embeddingFn(beginTokens)

    /** Scalar `INT32` tensor representing the batch size of a tensor returned by `sample()`. */
    override val batchSize: Output = Op.createWithNameScope(name, Set(beginTokens.op)) {
      Basic.size(beginTokens)
    }

    /** Shape of tensor returned by `sample()`, excluding the batch dimension. */
    override val sampleShape: Shape = Shape.scalar()

    /** Data type of tensor returned by `sample()`. */
    override val sampleDataType: DataType = INT32

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether initialization has finished, and
      * (ii) the next input. */
    override def initialize(): (Output, Output) = {
      Op.createWithNameScope(s"$name/Initialize") {
        (Basic.tile(Tensor(false), batchSize.expandDims(0)), beginInputs)
      }
    }

    /** Returns a sample for the provided `time`, `outputs`, and `state`. */
    override def sample(time: Output, outputs: Output, state: Output): Output = {
      Op.createWithNameScope(s"$name/Sample", Set(time.op, outputs.op)) {
        Math.cast(Math.argmax(outputs, axes = -1), INT32)
      }
    }

    /** Returns a tuple containing: (i) a scalar `BOOLEAN` tensor specifying whether sampling has finished,
      * (ii) the next input, and (iii) a tensor containing the next state. */
    override def nextInput(time: Output, outputs: Output, state: Output, sample: Output): (Output, Output, Output) = {
      Op.createWithNameScope(s"$name/NextInputs", Set(time.op, outputs.op, state.op)) {
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
