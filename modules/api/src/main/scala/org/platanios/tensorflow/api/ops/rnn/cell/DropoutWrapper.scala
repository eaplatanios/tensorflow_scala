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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types.{IsHalfOrFloatOrDouble, IsIntOrLong, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToShape, SparseShape}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/** RNN cell that applies dropout to the provided RNN cell.
  *
  * Note that currently, a different dropout mask is used for each time step in an RNN (i.e., not using the variational
  * recurrent dropout method described in
  * ["A Theoretically Grounded Application of Dropout in Recurrent Neural Networks"](https://arxiv.org/abs/1512.05287).
  *
  * Note also that for LSTM cells, no dropout is applied to the memory tensor of the state. It is only applied to the
  * state tensor.
  *
  * @param  cell                  RNN cell on which to perform dropout.
  * @param  inputKeepProbability  Keep probability for the input of the RNN cell.
  * @param  outputKeepProbability Keep probability for the output of the RNN cell.
  * @param  stateKeepProbability  Keep probability for the output state of the RNN cell.
  * @param  seed                  Optional random seed, used to generate a random seed pair for the random number
  *                               generator, when combined with the graph-level seed.
  * @param  name                  Name prefix used for all new ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class DropoutWrapper[Out: OutputStructure, State: OutputStructure] protected (
    val cell: RNNCell[Out, State],
    val inputKeepProbability: Output[Float] = 1.0f,
    val outputKeepProbability: Output[Float] = 1.0f,
    val stateKeepProbability: Output[Float] = 1.0f,
    val seed: Option[Int] = None,
    val name: String = "DropoutWrapper"
) extends RNNCell[Out, State]() {
  type OutShape = cell.OutShape
  type StateShape = cell.StateShape

  override def evOutputToShapeOut: OutputToShape.Aux[Out, OutShape] = cell.evOutputToShapeOut
  override def evOutputToShapeState: OutputToShape.Aux[State, StateShape] = cell.evOutputToShapeState

  override def outputShape: OutShape = {
    cell.outputShape
  }

  override def stateShape: StateShape = {
    cell.stateShape
  }

  override def forward(input: Tuple[Out, State]): Tuple[Out, State] = {
    Op.nameScope(name) {
      val dropoutInput = OutputStructure[Out].map(
        input.output, DropoutWrapper.DropoutConverter(inputKeepProbability, "input", seed))
      val nextTuple = cell(Tuple(dropoutInput, input.state))
      val nextState = OutputStructure[State].map(
        nextTuple.state, DropoutWrapper.DropoutConverter(stateKeepProbability, "state", seed))
      val nextOutput = OutputStructure[Out].map(
        nextTuple.output, DropoutWrapper.DropoutConverter(outputKeepProbability, "output", seed))
      Tuple(nextOutput, nextState)
    }
  }
}

object DropoutWrapper {
  def apply[Out: OutputStructure, State: OutputStructure](
      cell: RNNCell[Out, State],
      inputKeepProbability: Output[Float] = 1.0f,
      outputKeepProbability: Output[Float] = 1.0f,
      stateKeepProbability: Output[Float] = 1.0f,
      seed: Option[Int] = None,
      name: String = "DropoutWrapper"
  ): DropoutWrapper[Out, State] = {
    new DropoutWrapper(
      cell, inputKeepProbability, outputKeepProbability,
      stateKeepProbability, seed, name)
  }

  private def generateSeed(
      saltPrefix: String,
      seed: Option[Int],
      index: Int
  ): Option[Int] = {
    // TODO: [OPS] !!! What about the index?

    seed.map(s => {
      val md5 = MessageDigest.getInstance("MD5")
          .digest(s"$s${saltPrefix}_$index".getBytes(StandardCharsets.UTF_8))
      ByteBuffer.wrap(md5.take(8)).getInt() & 0x7fffffff
    })
  }

  private[DropoutWrapper] case class DropoutConverter(
      keepProbability: Output[Float],
      saltPrefix: String,
      seed: Option[Int]
  ) extends OutputStructure.Converter {
    // TODO: [IMPLICITS] !!! Handle OutputIndexedSlices and SparseOutput.

    override def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = {

      // TODO: [TYPES] !!! Super hacky. Remove in the future.
      val ev: IsHalfOrFloatOrDouble[T] = null

      NN.dynamicDropout(
        value,
        keepProbability.castTo[T](TF.fromDataType(value.dataType)),
        seed = generateSeed(saltPrefix, seed, index = 0)
      )(TF.fromDataType(value.dataType), ev, IntDefault[Int], TF[Int], IsIntOrLong[Int])
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: OutputIndexedSlices[T], shape: Option[SparseShape]): OutputIndexedSlices[T] = {
      throw InvalidArgumentException("Tensor indexed slices are not supported in the dropout wrapper.")
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: SparseOutput[T], shape: Option[SparseShape]): SparseOutput[T] = {
      throw InvalidArgumentException("Sparse tensors are not supported in the dropout wrapper.")
    }

    @throws[InvalidArgumentException]
    override def apply[T](value: Dataset[T], shape: Option[Shape]): Dataset[T] = {
      throw InvalidArgumentException("Unsupported argument type for use with the dropout wrapper.")
    }
  }
}
