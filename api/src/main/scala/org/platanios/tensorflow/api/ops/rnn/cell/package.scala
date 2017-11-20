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

package org.platanios.tensorflow.api.ops.rnn

import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
package object cell {
  class Tuple[O, S](val output: O, val state: S)

  object Tuple {
    def apply[O, S](output: O, state: S): Tuple[O, S] = new Tuple(output, state)
  }

  type BasicTuple = Tuple[Output, Output]
  type LSTMTuple = Tuple[Output, (Output, Output)]

  def LSTMTuple(output: Output, state: (Output, Output)): LSTMTuple = Tuple(output, state)

  private[rnn] trait API {
    type RNNCell[O, OS, S, SS] = cell.RNNCell[O, OS, S, SS]
    type BasicRNNCell = cell.BasicRNNCell
    type GRUCell = cell.GRUCell
    type BasicLSTMCell = cell.BasicLSTMCell
    type LSTMCell = cell.LSTMCell
    type MultiRNNCell[O, OS, S, SS] = cell.MultiRNNCell[O, OS, S, SS]

    val BasicRNNCell : cell.BasicRNNCell.type  = cell.BasicRNNCell
    val GRUCell      : cell.GRUCell.type       = cell.GRUCell
    val BasicLSTMCell: cell.BasicLSTMCell.type = cell.BasicLSTMCell
    val LSTMCell     : cell.LSTMCell.type      = cell.LSTMCell
    val MultiRNNCell : cell.MultiRNNCell.type  = cell.MultiRNNCell

    type RNNTuple[O, S] = cell.Tuple[O, S]
    type BasicTuple = cell.Tuple[Output, Output]
    type LSTMTuple = cell.Tuple[Output, (Output, Output)]

    val RNNTuple: cell.Tuple.type = cell.Tuple

    def LSTMTuple(output: Output, state: (Output, Output)): LSTMTuple = cell.Tuple(output, state)
  }
}
