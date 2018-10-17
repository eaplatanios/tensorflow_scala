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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.{Initializer, ZerosInitializer}

/** $OpDocRNNCellLSTMCell
  *
  * @param  name              Name scope (also acting as variable scope) for this layer.
  * @param  numUnits          Number of units in the LSTM cell.
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
class LSTMCell[T: TF : IsNotQuantized](
    override val name: String,
    val numUnits: Int,
    val activation: Output[T] => Output[T],
    val forgetBias: Float = 1.0f,
    val usePeepholes: Boolean = false,
    val cellClip: Float = -1,
    val projectionSize: Int = -1,
    val projectionClip: Float = -1,
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output[T], LSTMState[T]](name) {
  override val layerType: String = "LSTMCell"

  override def createCellWithoutContext[OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureO: NestedStructure.Aux[Output[T], _, _, OS]): ops.rnn.cell.LSTMCell[T] = {
    val shape = inputShape.asInstanceOf[Shape]
    val hiddenDepth = if (projectionSize != -1) projectionSize else numUnits
    val kernel = getParameter[T](KERNEL_NAME, Shape(shape(-1) + hiddenDepth, 4 * numUnits), kernelInitializer)
    val bias = getParameter[T](BIAS_NAME, Shape(4 * numUnits), biasInitializer)
    val (wfDiag, wiDiag, woDiag) = {
      if (usePeepholes) {
        val wfDiag = getParameter[T]("Peepholes/ForgetKernelDiag", Shape(numUnits), kernelInitializer)
        val wiDiag = getParameter[T]("Peepholes/InputKernelDiag", Shape(numUnits), kernelInitializer)
        val woDiag = getParameter[T]("Peepholes/OutputKernelDiag", Shape(numUnits), kernelInitializer)
        (wfDiag, wiDiag, woDiag)
      } else {
        (null, null, null)
      }
    }
    val projectionKernel = {
      if (projectionSize != -1) {
        getParameter[T](s"Projection/$KERNEL_NAME", Shape(numUnits, projectionSize), kernelInitializer)
      } else {
        null
      }
    }
    ops.rnn.cell.LSTMCell(
      kernel, bias, activation, cellClip, wfDiag, wiDiag, woDiag, projectionKernel, projectionClip,
      forgetBias, name)
  }
}

object LSTMCell {
  def apply[T: TF : IsNotQuantized](
      variableScope: String,
      numUnits: Int,
      activation: Output[T] => Output[T],
      forgetBias: Float = 1.0f,
      usePeepholes: Boolean = false,
      cellClip: Float = -1,
      projectionSize: Int = -1,
      projectionClip: Float = -1,
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): LSTMCell[T] = {
    new LSTMCell(
      variableScope, numUnits, activation, forgetBias, usePeepholes, cellClip,
      projectionSize, projectionClip, kernelInitializer, biasInitializer)
  }
}
