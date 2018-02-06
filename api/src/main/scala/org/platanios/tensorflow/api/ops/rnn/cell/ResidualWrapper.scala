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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/** RNN cell that creates a residual connection (i.e., combining the cell inputs and its outputs) over another RNN cell.
  *
  * @param  cell       RNN cell being wrapped.
  * @param  residualFn Residual function to use that maps from a tuple of cell input and cell output to the new cell
  *                    output. Common choices include the addition and the concatenation functions.
  *
  * @author Emmanouil Antonios Platanios
  */
class ResidualWrapper[O, OS, S, SS](
    val cell: RNNCell[O, OS, S, SS],
    val residualFn: (O, O) => O
)(implicit
    evO: WhileLoopVariable.Aux[O, OS],
    evS: WhileLoopVariable.Aux[S, SS]
) extends RNNCell[O, OS, S, SS]()(evO, evS) {
  override def outputShape: OS = cell.outputShape
  override def stateShape: SS = cell.stateShape

  override def forward(input: Tuple[O, S]): Tuple[O, S] = {
    val nextTuple = cell.forward(input)
    val nextOutput = residualFn(input.output, nextTuple.output)
    Tuple(nextOutput, nextTuple.state)
  }
}

object ResidualWrapper {
  def apply[O, OS, S, SS](
      cell: RNNCell[O, OS, S, SS],
      residualFn: (O, O) => O
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS]
  ): ResidualWrapper[O, OS, S, SS] = {
    new ResidualWrapper(cell, residualFn)(evO, evS)
  }
}
