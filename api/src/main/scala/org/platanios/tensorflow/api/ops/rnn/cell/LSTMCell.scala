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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Math, Output}

/** $OpDocRNNCellLSTMCell
  *
  * @group RNNCellOps
  * @param  kernel           Kernel matrix to use.
  * @param  bias             Bias vector to use.
  * @param  cellClip         If different than `-1`, then the cell state is clipped by this value prior to the cell
  *                          output activation.
  * @param  wfDiag           If not `null`, then diagonal peep-hole connections are added from the forget gate to the
  *                          state, using these weights.
  * @param  wiDiag           If not `null`, then diagonal peep-hole connections are added from the input gate to the
  *                          state, using these weights.
  * @param  woDiag           If not `null`, then diagonal peep-hole connections are added from the output gate to the
  *                          state, using these weights.
  * @param  projectionKernel If not `null`, then this matrix is used to project the cell output.
  * @param  projectionClip   If different than `-1` and `projectionKernel` not `null`, then the projected output is
  *                          clipped by this value.
  * @param  activation       Activation function to use.
  * @param  forgetBias       Forget bias added to the forget gate.
  * @param  name             Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class LSTMCell private[cell] (
    val kernel: Output,
    val bias: Output,
    val cellClip: Float = -1,
    val wfDiag: Output = null,
    val wiDiag: Output = null,
    val woDiag: Output = null,
    val projectionKernel: Output = null,
    val projectionClip: Float = -1,
    val activation: Output => Output = Math.tanh(_),
    val forgetBias: Float = 1.0f,
    val name: String = "LSTMCell"
) extends RNNCell.LSTMCell {
  private[this] val numUnits = bias.shape(0) / 4

  override def outputShape: Shape = {
    if (projectionKernel != null)
      Shape(projectionKernel.shape(1))
    else
      Shape(numUnits)
  }

  override def stateShape: (Shape, Shape) = {
    if (projectionKernel != null)
      (Shape(numUnits), Shape(projectionKernel.shape(1)))
    else
      (Shape(numUnits), Shape(numUnits))
  }

  override def forward(input: RNNCell.LSTMTuple): RNNCell.LSTMTuple = {
    RNNCell.lstmCell(
      input, kernel, bias, cellClip, wfDiag, wiDiag, woDiag, projectionKernel, projectionClip, activation, forgetBias,
      name)
  }
}

object LSTMCell {
  def apply(
      kernel: Output, bias: Output, cellClip: Float = -1,
      wfDiag: Output = null, wiDiag: Output = null, woDiag: Output = null,
      projectionKernel: Output = null, projectionClip: Float = -1,
      activation: Output => Output = Math.tanh(_), forgetBias: Float = 1.0f,
      name: String = "LSTMCell"): LSTMCell = {
    new LSTMCell(
      kernel, bias, cellClip, wfDiag, wiDiag, woDiag, projectionKernel, projectionClip, activation, forgetBias, name)
  }
}
