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

package org.platanios.tensorflow.api.learn.layers.rnn

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.learn.layers.rnn.cell.RNNCell
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

import scala.collection.mutable

/** Creates a bidirectional dynamic RNN layer.
  *
  * $OpDocRNNBidirectionalDynamicRNN
  *
  * @param  cellFw             RNN cell to use for the forward direction.
  * @param  cellBw             RNN cell to use for the backward direction.
  * @param  initialStateFw     Initial state to use for the forward RNN, which is a sequence of tensors with shapes
  *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
  *                            Defaults to a zero state.
  * @param  initialStateBw     Initial state to use for the backward RNN, which is a sequence of tensors with shapes
  *                            `[batchSize, stateSize(i)]`, where `i` corresponds to the index in that sequence.
  *                            Defaults to a zero state.
  * @param  timeMajor          Boolean value indicating whether the inputs are provided in time-major format (i.e.,
  *                            have shape `[time, batch, depth]`) or in batch-major format (i.e., have shape
  *                            `[batch, time, depth]`).
  * @param  parallelIterations Number of RNN loop iterations allowed to run in parallel.
  * @param  swapMemory         If `true`, GPU-CPU memory swapping support is enabled for the RNN loop.
  * @param  sequenceLengths    Optional `INT32` tensor with shape `[batchSize]` containing the sequence lengths for
  *                            each row in the batch.
  * @param  name               Desired name for this layer (note that this name will be made unique by potentially
  *                            appending a number to it, if it has been used before for another layer).
  *
  * @author Emmanouil Antonios Platanios
  */
class BidirectionalRNN[T: RNNCell.Output] private[rnn] (
    cellFw: RNNCell[T],
    cellBw: RNNCell[T],
    initialStateFw: Seq[Tensor] = null,
    initialStateBw: Seq[Tensor] = null,
    timeMajor: Boolean = false,
    parallelIterations: Int = 32,
    swapMemory: Boolean = false,
    sequenceLengths: Tensor = null,
    override protected val name: String = "BidirectionalRNN"
) extends Layer[T, (RNNCell.Tuple[T], RNNCell.Tuple[T])](name) {
  override val layerType: String = "BidirectionalRNN"

  override def forward(input: T, mode: Mode): LayerInstance[T, (RNNCell.Tuple[T], RNNCell.Tuple[T])] = {
    val stateFw = if (initialStateFw == null) null else initialStateFw.map(ops.Basic.constant(_))
    val stateBw = if (initialStateBw == null) null else initialStateBw.map(ops.Basic.constant(_))
    val lengths = if (sequenceLengths == null) null else ops.Basic.constant(sequenceLengths)
    val trainableVariables = mutable.Set.empty[Variable]
    val nonTrainableVariables = mutable.Set.empty[Variable]

    def cellFwFunction(input: RNNCell.Tuple[T]): RNNCell.Tuple[T] = {
      val instance = cellFw.forward(input, mode)
      trainableVariables ++= instance.trainableVariables
      nonTrainableVariables ++= instance.nonTrainableVariables
      instance.output
    }

    def cellBwFunction(input: RNNCell.Tuple[T]): RNNCell.Tuple[T] = {
      val instance = cellBw.forward(input, mode)
      trainableVariables ++= instance.trainableVariables
      nonTrainableVariables ++= instance.nonTrainableVariables
      instance.output
    }

    LayerInstance(
      input,
      ops.rnn.RNN.bidirectionalDynamicRNN(
        cellFwFunction, cellFw.outputSize,
        cellBwFunction, cellBw.outputSize,
        input,
        stateFw, cellFw.zeroState,
        stateBw, cellBw.zeroState,
        timeMajor, parallelIterations, swapMemory, lengths, uniquifiedName),
      trainableVariables.toSet,
      nonTrainableVariables.toSet)
  }
}

object BidirectionalRNN {
  def apply[T: RNNCell.Output](
      cellFw: RNNCell[T],
      cellBw: RNNCell[T],
      initialStateFw: Seq[Tensor] = null,
      initialStateBw: Seq[Tensor] = null,
      timeMajor: Boolean = false,
      parallelIterations: Int = 32,
      swapMemory: Boolean = false,
      sequenceLengths: Tensor = null,
      name: String = "BidirectionalRNN"): BidirectionalRNN[T] = {
    new BidirectionalRNN[T](
      cellFw, cellBw,
      initialStateFw, initialStateBw,
      timeMajor, parallelIterations, swapMemory, sequenceLengths, name)
  }
}
