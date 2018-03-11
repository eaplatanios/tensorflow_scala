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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Math, Output}

/** $OpDocRNNCellBasicRNNCell
  *
  * @group RNNCellOps
  * @param  kernel     Kernel matrix to use.
  * @param  bias       Bias vector to use.
  * @param  activation Activation function to use.
  * @param  name       Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicRNNCell private[cell] (
    val kernel: Output,
    val bias: Output,
    val activation: Output => Output = Math.tanh(_),
    val name: String = "BasicRNNCell"
) extends RNNCell[Output, Shape, Output, Shape] {
  override def outputShape: Shape = bias.shape
  override def stateShape: Shape = bias.shape

  override def forward(input: BasicTuple): BasicTuple = {
    RNNCell.basicRNNCell(input, kernel, bias, activation, name)
  }
}

object BasicRNNCell {
  def apply(
      kernel: Output, bias: Output, activation: Output => Output = Math.tanh(_),
      name: String = "BasicRNNCell"): BasicRNNCell = {
    new BasicRNNCell(kernel, bias, activation, name)
  }
}
