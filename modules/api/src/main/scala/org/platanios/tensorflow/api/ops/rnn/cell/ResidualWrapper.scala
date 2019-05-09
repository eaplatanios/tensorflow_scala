/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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
class ResidualWrapper[Out, State, OutShape, StateShape] protected (
    val cell: RNNCell[Out, State, OutShape, StateShape],
    val residualFn: (Out, Out) => Out
)(implicit
    evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
    evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends RNNCell[Out, State, OutShape, StateShape] {
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
  def apply[Out, State, OutShape, StateShape](
      cell: RNNCell[Out, State, OutShape, StateShape],
      residualFn: (Out, Out) => Out
  )(implicit
      evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
      evOutputToShapeState: OutputToShape.Aux[State, StateShape]
  ): ResidualWrapper[Out, State, OutShape, StateShape] = {
    new ResidualWrapper(cell, residualFn)
  }
}
