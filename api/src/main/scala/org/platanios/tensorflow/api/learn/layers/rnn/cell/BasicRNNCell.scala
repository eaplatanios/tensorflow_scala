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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.{Initializer, ZerosInitializer}

/** $OpDocRNNCellBasicRNNCell
  *
  * @param  name              Name scope (also acting as variable scope) for this layer.
  * @param  numUnits          Number of units in the RNN cell.
  * @param  dataType          Data type for the parameters of this cell.
  * @param  activation        Activation function used by this RNN cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicRNNCell(
    override val name: String,
    val numUnits: Int,
    val dataType: DataType,
    val activation: Output => Output = ops.Math.tanh(_),
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output, Shape, Output, Shape](name) {
  override val layerType: String = "BasicRNNCell"

  override def createCellWithoutContext(mode: Mode, inputShape: Shape): ops.rnn.cell.BasicRNNCell = {
    val kernel = tf.variable(KERNEL_NAME, dataType, Shape(inputShape(-1) + numUnits, numUnits), kernelInitializer)
    val bias = tf.variable(BIAS_NAME, dataType, Shape(numUnits), biasInitializer)
    ops.rnn.cell.BasicRNNCell(kernel, bias, activation, name)
  }
}

object BasicRNNCell {
  def apply(
      variableScope: String,
      numUnits: Int,
      dataType: DataType,
      activation: Output => Output = ops.Math.tanh(_),
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): BasicRNNCell = {
    new BasicRNNCell(variableScope, numUnits, dataType, activation, kernelInitializer, biasInitializer)
  }
}
