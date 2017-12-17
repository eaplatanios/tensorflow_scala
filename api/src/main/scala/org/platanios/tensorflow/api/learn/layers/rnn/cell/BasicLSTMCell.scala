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

/** $OpDocRNNCellBasicLSTMCell
  *
  * @param  variableScope     Variable scope (also acting as name scope) for this layer.
  * @param  numUnits          Number of units in the LSTM cell.
  * @param  dataType          Data type for the parameters of this cell.
  * @param  forgetBias        Forget bias added to the forget gate.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicLSTMCell(
    override val variableScope: String,
    val numUnits: Int,
    val dataType: DataType,
    val forgetBias: Float = 1.0f,
    val activation: Output => Output = ops.Math.tanh(_),
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output, Shape, LSTMState, (Shape, Shape)](variableScope) {
  override val layerType: String = "BasicLSTMCell"

  override def createCell(mode: Mode, inputShape: Shape): LSTMCellInstance = {
    val kernel = tf.variable(
      KERNEL_NAME, dataType, Shape(inputShape(-1) + numUnits, 4 * numUnits), kernelInitializer)
    val bias = tf.variable(BIAS_NAME, dataType, Shape(4 * numUnits), biasInitializer)
    val cell = ops.rnn.cell.BasicLSTMCell(kernel, bias, activation, forgetBias, variableScope)
    LSTMCellInstance(cell, Set(kernel, bias))
  }
}

object BasicLSTMCell {
  def apply(
      variableScope: String,
      numUnits: Int,
      dataType: DataType,
      forgetBias: Float = 1.0f,
      activation: Output => Output = ops.Math.tanh(_),
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): BasicLSTMCell = {
    new BasicLSTMCell(variableScope, numUnits, dataType, forgetBias, activation, kernelInitializer, biasInitializer)
  }
}
