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

/** $OpDocRNNCellGRUCell
  *
  * @param  name              Name scope (also acting as variable scope) for this layer.
  * @param  numUnits          Number of units in the GRU cell.
  * @param  activation        Activation function used by this GRU cell.
  * @param  kernelInitializer Variable initializer for kernel matrices.
  * @param  biasInitializer   Variable initializer for the bias vectors.
  *
  * @author Emmanouil Antonios Platanios
  */
class GRUCell[T: TF : IsNotQuantized](
    override val name: String,
    val numUnits: Int,
    val activation: Output[T] => Output[T],
    val kernelInitializer: Initializer = null,
    val biasInitializer: Initializer = ZerosInitializer
) extends RNNCell[Output[T], Output[T]](name) {
  override val layerType: String = "GRUCell"

  override def createCellWithoutContext[OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureO: NestedStructure.Aux[Output[T], _, _, OS]): ops.rnn.cell.GRUCell[T] = {
    val shape = inputShape.asInstanceOf[Shape]
    val gateKernel = getParameter[T](
      s"Gate/$KERNEL_NAME",
      Shape(shape(-1) + numUnits, 2 * numUnits),
      kernelInitializer)
    val gateBias = getParameter[T](
      s"Gate/$BIAS_NAME",
      Shape(2 * numUnits),
      biasInitializer)
    val candidateKernel = getParameter[T](
      s"Candidate/$KERNEL_NAME",
      Shape(shape(-1) + numUnits, numUnits),
      kernelInitializer)
    val candidateBias = getParameter[T](
      s"Candidate/$BIAS_NAME",
      Shape(numUnits),
      biasInitializer)
    ops.rnn.cell.GRUCell(gateKernel, gateBias, candidateKernel, candidateBias, activation, name)
  }
}

object GRUCell {
  def apply[T: TF : IsNotQuantized](
      variableScope: String,
      numUnits: Int,
      activation: Output[T] => Output[T],
      kernelInitializer: Initializer = null,
      biasInitializer: Initializer = ZerosInitializer
  ): GRUCell[T] = {
    new GRUCell(variableScope, numUnits, activation, kernelInitializer, biasInitializer)
  }
}
