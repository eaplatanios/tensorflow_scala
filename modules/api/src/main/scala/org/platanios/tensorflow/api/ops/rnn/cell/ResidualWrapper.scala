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

import org.platanios.tensorflow.api.implicits.helpers.NestedStructure

/** RNN cell that creates a residual connection (i.e., combining the cell inputs and its outputs) over another RNN cell.
  *
  * @param  cell       RNN cell being wrapped.
  * @param  residualFn Residual function to use that maps from a tuple of cell input and cell output to the new cell
  *                    output. Common choices include the addition and the concatenation functions.
  *
  * @author Emmanouil Antonios Platanios
  */
class ResidualWrapper[O, S] protected (
    val cell: RNNCell[O, S],
    val residualFn: (O, O) => O
) extends RNNCell[O, S]() {
  override def outputShape[OS](implicit evStructureO: NestedStructure.Aux[O, _, _, OS]): OS = {
    cell.outputShape
  }

  override def stateShape[SS](implicit evStructureS: NestedStructure.Aux[S, _, _, SS]): SS = {
    cell.stateShape
  }

  override def forward(input: Tuple[O, S]): Tuple[O, S] = {
    val nextTuple = cell.forward(input)
    val nextOutput = residualFn(input.output, nextTuple.output)
    Tuple(nextOutput, nextTuple.state)
  }
}

object ResidualWrapper {
  def apply[O, S](
      cell: RNNCell[O, S],
      residualFn: (O, O) => O
  ): ResidualWrapper[O, S] = {
    new ResidualWrapper(cell, residualFn)
  }
}
