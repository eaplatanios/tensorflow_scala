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

import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/** RNN cell that is composed by applying a sequence of RNN cells in order.
  *
  * This means that the output of each RNN is fed to the next one as input, while the states remain separate.
  *
  * Note that this class does no variable management at all. Variable sharing should be handled based on the RNN cells
  * the caller provides to this class. The learn API provides a layer version of this class that also does some
  * management of the variables involved.
  *
  * @param  cells Cells being stacked together.
  * @param  name  Name prefix used for all new ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class MultiCell[O, OS, S, SS] protected (
    val cells: Seq[RNNCell[O, OS, S, SS]],
    val name: String = "MultiCell"
)(implicit
    evO: WhileLoopVariable.Aux[O, OS],
    evS: WhileLoopVariable.Aux[S, SS]
) extends RNNCell[O, OS, Seq[S], Seq[SS]] {
  override def outputShape: OS = cells.last.outputShape
  override def stateShape: Seq[SS] = cells.map(_.stateShape)

  override def forward(input: Tuple[O, Seq[S]]): Tuple[O, Seq[S]] = {
    Op.nameScope(name) {
      var currentInput = input.output
      val state = cells.zip(input.state).map {
        case (cell, s) =>
          val nextTuple = cell(Tuple(currentInput, s))
          currentInput = nextTuple.output
          nextTuple.state
      }
      Tuple(currentInput, state)
    }
  }
}

object MultiCell {
  def apply[O, OS, S, SS](
      cells: Seq[RNNCell[O, OS, S, SS]],
      name: String = "MultiCell"
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS]
  ): MultiCell[O, OS, S, SS] = {
    new MultiCell(cells, name)
  }
}
