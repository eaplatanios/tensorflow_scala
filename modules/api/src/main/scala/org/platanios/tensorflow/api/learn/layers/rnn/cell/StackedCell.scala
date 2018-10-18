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
import org.platanios.tensorflow.api.ops.variables.VariableScope

/** RNN cell that is composed by applying a sequence of RNN cells in order.
  *
  * This will create a different set of variables for each layer in the stacked LSTM cell (i.e., no variable sharing).
  *
  * @param  name  Name scope (also acting as variable scope) for this layer.
  * @param  cells Cells being stacked together.
  *
  * @author Emmanouil Antonios Platanios
  */
class StackedCell[O: NestedStructure, S: NestedStructure](
    override val name: String,
    val cells: Seq[RNNCell[O, S]]
) extends RNNCell[O, Seq[S]](name) {
  override val layerType: String = "StackedCell"

  override def createCellWithoutContext[OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureOAux: NestedStructure.Aux[O, _, _, OS]): ops.rnn.cell.RNNCell[O, Seq[S]] = {
    val createdCells = cells.zipWithIndex.foldLeft(Seq.empty[ops.rnn.cell.RNNCell[O, S]])((seq, cell) => {
      VariableScope.scope(s"Cell${cell._2}") {
        if (seq.isEmpty)
          seq :+ cell._1.createCellWithoutContext(mode, inputShape)
        else
          seq :+ cell._1.createCellWithoutContext(mode, seq.last.outputShape)
      }
    })
    ops.rnn.cell.StackedCell(createdCells)
  }
}

object StackedCell {
  def apply[O: NestedStructure, S: NestedStructure](
      variableScope: String,
      cells: Seq[RNNCell[O, S]]
  ): StackedCell[O, S] = {
    new StackedCell(variableScope, cells)
  }
}
