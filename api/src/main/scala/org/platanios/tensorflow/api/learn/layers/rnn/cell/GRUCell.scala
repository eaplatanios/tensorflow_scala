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

/** $OpDocRNNCellGRUCell
  *
  * @param  name              Name scope (also acting as variable scope) for this layer.
  * @param  numUnits          Number of units in the GRU cell.
  * @param  dataType          Data type for the parameters of this cell.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  *
  * @author Emmanouil Antonios Platanios
  */
class GRUCell(
    override val name: String,
    val numUnits: Int,
    val dataType: DataType,
    val activation: Output => Output = ops.Math.tanh(_),
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output, Shape, Output, Shape](name) {
  override val layerType: String = "GRUCell"

  override def createCellWithoutContext(mode: Mode, inputShape: Shape): ops.rnn.cell.GRUCell = {
    val gateKernel = tf.variable(
      s"Gate/$KERNEL_NAME",
      dataType,
      Shape(inputShape(-1) + numUnits, 2 * numUnits),
      kernelInitializer)
    val gateBias = tf.variable(
      s"Gate/$BIAS_NAME",
      dataType, Shape(2 * numUnits),
      biasInitializer)
    val candidateKernel = tf.variable(
      s"Candidate/$KERNEL_NAME",
      dataType,
      Shape(inputShape(-1) + numUnits, numUnits),
      kernelInitializer)
    val candidateBias = tf.variable(
      s"Candidate/$BIAS_NAME",
      dataType,
      Shape(numUnits),
      biasInitializer)
    ops.rnn.cell.GRUCell(gateKernel, gateBias, candidateKernel, candidateBias, activation, name)
  }
}

object GRUCell {
  def apply(
      variableScope: String,
      numUnits: Int,
      dataType: DataType,
      activation: Output => Output = ops.Math.tanh(_),
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): GRUCell = {
    new GRUCell(variableScope, numUnits, dataType, activation, kernelInitializer, biasInitializer)
  }
}
