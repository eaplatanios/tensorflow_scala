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

/** $OpDocRNNCellBasicLSTMCell
  *
  * @group RNNCellOps
  * @param  kernel     Kernel matrix to use.
  * @param  bias       Bias vector to use.
  * @param  activation Activation function to use.
  * @param  forgetBias Forget bias added to the forget gate.
  * @param  name       Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicLSTMCell private[cell] (
    val kernel: Output,
    val bias: Output,
    val activation: Output => Output = Math.tanh(_),
    val forgetBias: Float = 1.0f,
    val name: String = "BasicLSTMCell"
) extends RNNCell[Output, Shape, (Output, Output), (Shape, Shape)] {
  private[this] val numUnits = bias.shape(0) / 4

  override def outputShape: Shape = Shape(numUnits)
  override def stateShape: (Shape, Shape) = (Shape(numUnits), Shape(numUnits))

  override def forward(input: LSTMTuple): LSTMTuple = {
    RNNCell.basicLSTMCell(input, kernel, bias, activation, forgetBias, name)
  }
}

object BasicLSTMCell {
  def apply(
      kernel: Output, bias: Output, activation: Output => Output = Math.tanh(_), forgetBias: Float = 1.0f,
      name: String = "BasicLSTMCell"): BasicLSTMCell = {
    new BasicLSTMCell(kernel, bias, activation, forgetBias, name)
  }
}
