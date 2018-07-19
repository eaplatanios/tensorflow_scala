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

package org.platanios.tensorflow.api.learn.layers.rnn.cell

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.{Initializer, ZerosInitializer}

/** $OpDocRNNCellLSTMCell
  *
  * @param  name              Name scope (also acting as variable scope) for this layer.
  * @param  numUnits          Number of units in the LSTM cell.
  * @param  dataType          Data type for the parameters of this cell.
  * @param  forgetBias        Forget bias added to the forget gate.
  * @param  usePeepholes      Boolean value indicating whether or not to use diagonal/peephole connections.
  * @param  cellClip          If different than `-1`, then the cell state is clipped by this value prior to the cell
  *                           output activation.
  * @param  projectionSize    If different than `-1`, then a projection to that size is added at the output.
  * @param  projectionClip    If different than `-1`, then the projected output is clipped by this value.
  * @param  activation        Activation function used by this LSTM cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  *
  * @author Emmanouil Antonios Platanios
  */
class LSTMCell(
    override val name: String,
    val numUnits: Int,
    val dataType: DataType,
    val forgetBias: Float = 1.0f,
    val usePeepholes: Boolean = false,
    val cellClip: Float = -1,
    val projectionSize: Int = -1,
    val projectionClip: Float = -1,
    val activation: Output => Output = ops.Math.tanh(_),
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output, Shape, LSTMState, (Shape, Shape)](name) {
  override val layerType: String = "LSTMCell"

  override def createCellWithoutContext(mode: Mode, inputShape: Shape): ops.rnn.cell.LSTMCell = {
    val hiddenDepth = if (projectionSize != -1) projectionSize else numUnits
    val kernel = getParameter(
      KERNEL_NAME, dataType, Shape(inputShape(-1) + hiddenDepth, 4 * numUnits), kernelInitializer)
    val bias = getParameter(BIAS_NAME, dataType, Shape(4 * numUnits), biasInitializer)
    val (wfDiag, wiDiag, woDiag) = {
      if (usePeepholes) {
        val wfDiag = getParameter("Peepholes/ForgetKernelDiag", dataType, Shape(numUnits), kernelInitializer)
        val wiDiag = getParameter("Peepholes/InputKernelDiag", dataType, Shape(numUnits), kernelInitializer)
        val woDiag = getParameter("Peepholes/OutputKernelDiag", dataType, Shape(numUnits), kernelInitializer)
        (wfDiag, wiDiag, woDiag)
      } else {
        (null, null, null)
      }
    }
    val projectionKernel = {
      if (projectionSize != -1) {
        val projectionKernel = getParameter(
          s"Projection/$KERNEL_NAME", dataType, Shape(numUnits, projectionSize), kernelInitializer)
        projectionKernel
      } else {
        null
      }
    }
    ops.rnn.cell.LSTMCell(
      kernel, bias, cellClip, wfDiag, wiDiag, woDiag, projectionKernel, projectionClip,
      activation, forgetBias, name)
  }
}

object LSTMCell {
  def apply(
      variableScope: String,
      numUnits: Int,
      dataType: DataType,
      forgetBias: Float = 1.0f,
      usePeepholes: Boolean = false,
      cellClip: Float = -1,
      projectionSize: Int = -1,
      projectionClip: Float = -1,
      activation: Output => Output = ops.Math.tanh(_),
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): LSTMCell = {
    new LSTMCell(
      variableScope, numUnits, dataType, forgetBias, usePeepholes, cellClip, projectionSize, projectionClip, activation,
      kernelInitializer, biasInitializer)
  }
}
