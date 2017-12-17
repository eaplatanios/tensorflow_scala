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
import org.platanios.tensorflow.api.learn.layers.rnn.cell.{RNNCell, Tuple}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Basic
import org.platanios.tensorflow.api.tensors.Tensor

/** Creates a bidirectional dynamic RNN layer.
  *
  * $OpDocRNNBidirectionalDynamicRNN
  *
  * @param  cellFw             RNN cell to use for the forward direction.
  * @param  cellBw             RNN cell to use for the backward direction.
  * @param  initialStateFw     Initial state to use for the forward RNN, which is a structure over tensors with shapes
  *                            `[batchSize, stateShape(i)(0), stateShape(i)(1), ...]`, where `i` corresponds to the
  *                            index of the corresponding state. Defaults to a zero state.
  * @param  initialStateBw     Initial state to use for the backward RNN, which is a structure over tensors with shapes
  *                            `[batchSize, stateShape(i)(0), stateShape(i)(1), ...]`, where `i` corresponds to the
  *                            index of the corresponding state. Defaults to a zero state.
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
class BidirectionalRNN[O, OS, S, SS] private[rnn] (
    val cellFw: RNNCell[O, OS, S, SS],
    val cellBw: RNNCell[O, OS, S, SS],
    val initialStateFw: () => S = null,
    val initialStateBw: () => S = null,
    val timeMajor: Boolean = false,
    val parallelIterations: Int = 32,
    val swapMemory: Boolean = false,
    val sequenceLengths: Tensor = null,
    override protected val name: String = "BidirectionalRNN"
)(implicit
    evO: ops.control_flow.WhileLoopVariable.Aux[O, OS],
    evS: ops.control_flow.WhileLoopVariable.Aux[S, SS]
) extends Layer[O, (Tuple[O, S], Tuple[O, S])](name) {
  override val layerType: String = "BidirectionalRNN"

  override protected def forward(input: O, mode: Mode): LayerInstance[O, (Tuple[O, S], Tuple[O, S])] = {
    val stateFw = if (initialStateFw == null) null.asInstanceOf[S] else initialStateFw()
    val stateBw = if (initialStateBw == null) null.asInstanceOf[S] else initialStateBw()
    val lengths = if (sequenceLengths == null) null else ops.Basic.constant(sequenceLengths)
    val cellInstanceFw = cellFw.createCell(mode, evO.fromShapes(input, evO.outputs(input).map(_.shape)))
    val cellInstanceBw = cellBw.createCell(mode, evO.fromShapes(input, evO.outputs(input).map(_.shape)))
    LayerInstance(
      input,
      ops.rnn.RNN.bidirectionalDynamicRNN(
        cellInstanceFw.cell, cellInstanceBw.cell, input, stateFw, stateBw,
        timeMajor, parallelIterations, swapMemory, lengths, uniquifiedName)(evO, evS),
      cellInstanceFw.trainableVariables ++ cellInstanceBw.trainableVariables,
      cellInstanceFw.nonTrainableVariables ++ cellInstanceBw.nonTrainableVariables)
  }

  def withConcatenatedOutputs: Layer[O, Tuple[O, (S, S)]] = {
    new Layer[O, Tuple[O, (S, S)]](s"$name/ConcatenatedOutputs") {
      override val layerType: String = "BidirectionalRNNWithConcatenatedOutputs"

      override protected def forward(input: O, mode: Mode): LayerInstance[O, Tuple[O, (S, S)]] = {
        val raw = BidirectionalRNN.this(input, mode)
        val output = evO.fromOutputs(
          raw.output._1.output, evO.outputs(raw.output._1.output).zip(evO.outputs(raw.output._2.output)).map(o => {
            Basic.concatenate(Seq(o._1, o._2), -1)
          }))
        LayerInstance(
          input,
          Tuple(output, (raw.output._1.state, raw.output._2.state)),
          raw.trainableVariables,
          raw.nonTrainableVariables)
      }
    }
  }
}

object BidirectionalRNN {
  def apply[O, OS, S, SS](
      cellFw: RNNCell[O, OS, S, SS],
      cellBw: RNNCell[O, OS, S, SS],
      initialStateFw: () => S = null,
      initialStateBw: () => S = null,
      timeMajor: Boolean = false,
      parallelIterations: Int = 32,
      swapMemory: Boolean = false,
      sequenceLengths: Tensor = null,
      name: String = "BidirectionalRNN"
  )(implicit
      evO: ops.control_flow.WhileLoopVariable.Aux[O, OS],
      evS: ops.control_flow.WhileLoopVariable.Aux[S, SS]
  ): BidirectionalRNN[O, OS, S, SS] = {
    new BidirectionalRNN(
      cellFw, cellBw, initialStateFw, initialStateBw,
      timeMajor, parallelIterations, swapMemory, sequenceLengths, name)(evO, evS)
  }
}
