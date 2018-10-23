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

import org.platanios.tensorflow.api.implicits.helpers.OutputToShape
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
class ResidualWrapper[Out, State, OutShape, StateShape](
    override val name: String,
    val cell: RNNCell[Out, State, OutShape, StateShape],
    val residualFn: (Out, Out) => Out
)(implicit
    evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
    evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends RNNCell[Out, State, OutShape, StateShape](name) {
  override val layerType: String = "ResidualWrapper"

  override def createCellWithoutContext(
      mode: Mode,
      inputShape: OutShape
  ): ops.rnn.cell.RNNCell[Out, State, OutShape, StateShape] = {
    val createdCell = cell.createCellWithoutContext(mode, inputShape)
    ops.rnn.cell.ResidualWrapper(createdCell, residualFn)
  }
}

object ResidualWrapper {
  def apply[Out, State, OutShape, StateShape](
      variableScope: String,
      cell: RNNCell[Out, State, OutShape, StateShape],
      residualFn: (Out, Out) => Out
  )(implicit
      evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
      evOutputToShapeState: OutputToShape.Aux[State, StateShape]
  ): ResidualWrapper[Out, State, OutShape, StateShape] = {
    new ResidualWrapper(variableScope, cell, residualFn)
  }
}
