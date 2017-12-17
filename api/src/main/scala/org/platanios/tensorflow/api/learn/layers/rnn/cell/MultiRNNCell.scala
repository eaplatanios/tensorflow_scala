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

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.variables.VariableScope

/** RNN cell that is composed by applying a sequence of RNN cells in order.
  *
  * This will create a different set of variables for each layer in the stacked LSTM cell (i.e., no variable sharing).
  *
  * @param  variableScope Variable scope (also acting as name scope) for this layer.
  * @param  cells         Cells being stacked together.
  *
  * @author Emmanouil Antonios Platanios
  */
class MultiRNNCell[O, OS, S, SS](
    override val variableScope: String,
    val cells: Seq[RNNCell[O, OS, S, SS]]
)(implicit
    evO: WhileLoopVariable.Aux[O, OS],
    evS: WhileLoopVariable.Aux[S, SS]
) extends RNNCell[O, OS, Seq[S], Seq[SS]](variableScope) {
  override val layerType: String = "MultiRNNCell"

  override def createCell(mode: Mode, inputShape: OS): CellInstance[O, OS, Seq[S], Seq[SS]] = {
    Op.createWithNameScope(variableScope) {
      val cellInstances = cells.zipWithIndex.foldLeft(Seq.empty[CellInstance[O, OS, S, SS]])((seq, cell) => {
        VariableScope.createWithVariableScope(s"Cell${cell._2}") {
          if (seq.isEmpty)
            seq :+ cell._1.createCell(mode, inputShape)
          else
            seq :+ cell._1.createCell(mode, seq.last.cell.outputShape)
        }
      })
      val cell = ops.rnn.cell.MultiRNNCell(cellInstances.map(_.cell))(evO, evS)
      CellInstance(
        cell,
        cellInstances.flatMap(_.trainableVariables).toSet,
        cellInstances.flatMap(_.nonTrainableVariables).toSet)
    }
  }
}

object MultiRNNCell {
  def apply[O, OS, S, SS](
      variableScope: String,
      cells: Seq[RNNCell[O, OS, S, SS]]
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS]
  ): MultiRNNCell[O, OS, S, SS] = {
    new MultiRNNCell(variableScope, cells)(evO, evS)
  }
}
