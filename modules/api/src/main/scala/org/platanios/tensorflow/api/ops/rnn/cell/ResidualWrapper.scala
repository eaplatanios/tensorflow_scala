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

import org.platanios.tensorflow.api.implicits.helpers.OutputToShape

/** RNN cell that creates a residual connection (i.e., combining the cell inputs and its outputs) over another RNN cell.
  *
  * @param  cell       RNN cell being wrapped.
  * @param  residualFn Residual function to use that maps from a tuple of cell input and cell output to the new cell
  *                    output. Common choices include the addition and the concatenation functions.
  *
  * @author Emmanouil Antonios Platanios
  */
class ResidualWrapper[Out, State] protected (
    val cell: RNNCell[Out, State],
    val residualFn: (Out, Out) => Out
) extends RNNCell[Out, State]() {
  type OutShape = cell.OutShape
  type StateShape = cell.StateShape

  override def evOutputToShapeOut: OutputToShape.Aux[Out, OutShape] = cell.evOutputToShapeOut
  override def evOutputToShapeState: OutputToShape.Aux[State, StateShape] = cell.evOutputToShapeState

  override def outputShape: OutShape = {
    cell.outputShape
  }

  override def stateShape: StateShape = {
    cell.stateShape
  }

  override def forward(input: Tuple[Out, State]): Tuple[Out, State] = {
    val nextTuple = cell.forward(input)
    val nextOutput = residualFn(input.output, nextTuple.output)
    Tuple(nextOutput, nextTuple.state)
  }
}

object ResidualWrapper {
  def apply[Out, State](
      cell: RNNCell[Out, State],
      residualFn: (Out, Out) => Out
  ): ResidualWrapper[Out, State] = {
    new ResidualWrapper(cell, residualFn)
  }
}
