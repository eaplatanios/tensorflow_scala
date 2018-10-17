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

import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops

/** RNN cell that creates a residual connection (i.e., combining the cell inputs and its outputs) over another RNN cell.
  *
  * @param  name       Name scope (also acting as variable scope) for this layer.
  * @param  cell       RNN cell being wrapped.
  * @param  residualFn Residual function to use that maps from a tuple of cell input and cell output to the new cell
  *                    output. Common choices include the addition and the concatenation functions.
  *
  * @author Emmanouil Antonios Platanios
  */
class ResidualWrapper[O, S](
    override val name: String,
    val cell: RNNCell[O, S],
    val residualFn: (O, O) => O
)(implicit
    override protected val evStructureO: NestedStructure.Aux[O, _, _, _]
) extends RNNCell[O, S](name) {
  override val layerType: String = "ResidualWrapper"

  override def createCellWithoutContext[OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureO: NestedStructure.Aux[O, _, _, OS]): ops.rnn.cell.RNNCell[O, S] = {
    val createdCell = cell.createCellWithoutContext(mode, inputShape)
    ops.rnn.cell.ResidualWrapper(createdCell, residualFn)
  }
}

object ResidualWrapper {
  def apply[O, S](
      variableScope: String,
      cell: RNNCell[O, S],
      residualFn: (O, O) => O
  )(implicit
      evStructureO: NestedStructure.Aux[O, _, _, _]
  ): ResidualWrapper[O, S] = {
    new ResidualWrapper(variableScope, cell, residualFn)
  }
}
