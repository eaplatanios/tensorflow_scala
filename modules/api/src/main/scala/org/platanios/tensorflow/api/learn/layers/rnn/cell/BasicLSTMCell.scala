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
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.{Initializer, ZerosInitializer}

/** $OpDocRNNCellBasicLSTMCell
  *
  * @param  name              Name scope (also acting as variable scope) for this layer.
  * @param  numUnits          Number of units in the LSTM cell.
  * @param  forgetBias        Forget bias added to the forget gate.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicLSTMCell[T: TF : IsNotQuantized](
    override val name: String,
    val numUnits: Int,
    val activation: Output[T] => Output[T],
    val forgetBias: Float = 1.0f,
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output[T], Shape, LSTMState[T], (Shape, Shape)](name) {
  override val layerType: String = "BasicLSTMCell"

  override def createCellWithoutContext(
      mode: Mode,
      inputShape: Shape
  ): ops.rnn.cell.BasicLSTMCell[T] = {
    val kernel = getParameter[T](KERNEL_NAME, Shape(inputShape(-1) + numUnits, 4 * numUnits), kernelInitializer)
    val bias = getParameter[T](BIAS_NAME, Shape(4 * numUnits), biasInitializer)
    ops.rnn.cell.BasicLSTMCell(kernel, bias, activation, forgetBias, name)
  }
}

object BasicLSTMCell {
  def apply[T: TF : IsNotQuantized](
      variableScope: String,
      numUnits: Int,
      activation: Output[T] => Output[T],
      forgetBias: Float = 1.0f,
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): BasicLSTMCell[T] = {
    new BasicLSTMCell(variableScope, numUnits, activation, forgetBias, kernelInitializer, biasInitializer)
  }
}
