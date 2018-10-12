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
class StackedCell[O, OS, S, SS](
    override val name: String,
    val cells: Seq[RNNCell[O, OS, S, SS]]
)(implicit
    evStructureO: NestedStructure.Aux[O, _, OS],
    evStructureS: NestedStructure.Aux[S, _, SS]
) extends RNNCell[O, OS, Seq[S], Seq[SS]](name) {
  override val layerType: String = "StackedCell"

  override def createCellWithoutContext(mode: Mode, inputShape: OS): ops.rnn.cell.RNNCell[O, OS, Seq[S], Seq[SS]] = {
    val createdCells = cells.zipWithIndex.foldLeft(Seq.empty[ops.rnn.cell.RNNCell[O, OS, S, SS]])((seq, cell) => {
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
  def apply[O, OS, S, SS](
      variableScope: String,
      cells: Seq[RNNCell[O, OS, S, SS]]
  )(implicit
      evStructureO: NestedStructure.Aux[O, _, OS],
      evStructureS: NestedStructure.Aux[S, _, SS]
  ): StackedCell[O, OS, S, SS] = {
    new StackedCell(variableScope, cells)
  }
}
