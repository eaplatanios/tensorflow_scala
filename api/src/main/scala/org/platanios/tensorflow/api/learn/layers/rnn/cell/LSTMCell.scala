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

package org.platanios.tensorflow.api.learn.layers.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.types.DataType

import scala.collection.mutable

/** $OpDocRNNCellLSTMCell
  *
  * @param  numUnits          Number of units in the LSTM cell.
  * @param  dataType          Data type for the parameters of this cell.
  * @param  inputShape        Shape of inputs to this cell.
  * @param  forgetBias        Forget bias added to the forget gate.
  * @param  usePeepholes      Boolean value indicating whether or not to use diagonal/peephole connections.
  * @param  cellClip          If different than `-1`, then the cell state is clipped by this value prior to the cell
  *                           output activation.
  * @param  projectionSize    If different than `-1`, then a projection to that size is added at the output.
  * @param  projectionClip    If different than `-1`, then the projected output is clipped by this value.
  * @param  activation        Activation function used by this LSTM cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  * @param  name              Desired name for this layer (note that this name will be made unique by potentially
  *                           appending a number to it, if it has been used before for another layer).
  *
  * @author Emmanouil Antonios Platanios
  */
class LSTMCell private[cell] (
    numUnits: Int,
    val dataType: DataType,
    val inputShape: Shape,
    forgetBias: Float = 1.0f,
    usePeepholes: Boolean = false,
    cellClip: Float = -1,
    projectionSize: Int = -1,
    projectionClip: Float = -1,
    activation: Output => Output = ops.Math.tanh(_),
    kernelInitializer: Initializer = null,
    biasInitializer: Initializer = ZerosInitializer,
    override protected val name: String = "LSTMCell"
) extends RNNCell[Output, Shape, (Output, Output), (Shape, Shape)](name) {
  override val layerType: String = "LSTMCell"

  override def createCell(mode: Mode): LSTMCellInstance = {
    val trainableVariables: mutable.Set[Variable] = mutable.Set[Variable]()
    val hiddenDepth = if (projectionSize != -1) projectionSize else numUnits
    val kernel = variable(
      KERNEL_NAME, dataType, Shape(inputShape(-1) + hiddenDepth, 4 * numUnits), kernelInitializer)
    val bias = variable(BIAS_NAME, dataType, Shape(4 * numUnits), biasInitializer)
    trainableVariables += kernel
    trainableVariables += bias
    val (wfDiag, wiDiag, woDiag) = {
      if (usePeepholes) {
        val wfDiag = variable("Peepholes/ForgetKernelDiag", dataType, Shape(numUnits), kernelInitializer)
        val wiDiag = variable("Peepholes/InputKernelDiag", dataType, Shape(numUnits), kernelInitializer)
        val woDiag = variable("Peepholes/OutputKernelDiag", dataType, Shape(numUnits), kernelInitializer)
        trainableVariables += wfDiag
        trainableVariables += wiDiag
        trainableVariables += woDiag
        (wfDiag.value, wiDiag.value, woDiag.value)
      } else {
        (null, null, null)
      }
    }
    val projectionKernel = {
      if (projectionSize != -1) {
        val projectionKernel = variable(
          s"Projection/$KERNEL_NAME",
          dataType,
          Shape(numUnits, projectionSize),
          kernelInitializer)
        trainableVariables += projectionKernel
        projectionKernel.value
      } else {
        null
      }
    }
    val cell = ops.rnn.cell.LSTMCell(
      kernel.value, bias.value, cellClip, wfDiag, wiDiag, woDiag, projectionKernel, projectionClip,
      activation, forgetBias, name)
    LSTMCellInstance(cell, trainableVariables.toSet)
  }
}

object LSTMCell {
  def apply(
      numUnits: Int,
      dataType: DataType,
      inputShape: Shape,
      forgetBias: Float = 1.0f,
      usePeepholes: Boolean = false,
      cellClip: Float = -1,
      projectionSize: Int = -1,
      projectionClip: Float = -1,
      activation: Output => Output = ops.Math.tanh(_),
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer,
      name: String = "LSTMCell"
  ): LSTMCell = {
    new LSTMCell(
      numUnits, dataType, inputShape, forgetBias, usePeepholes, cellClip, projectionSize, projectionClip, activation,
      kernelInitializer, biasInitializer, name)
  }
}
