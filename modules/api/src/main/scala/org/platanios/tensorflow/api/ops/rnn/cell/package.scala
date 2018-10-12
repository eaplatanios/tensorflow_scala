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

  type BasicTuple[T] = Tuple[Output[T], Output[T]]

  /** LSTM state tuple.
    *
    * @param  c Memory state tensor (i.e., previous output).
    * @param  m State tensor.
    */
  case class LSTMState[T](c: Output[T], m: Output[T])

  type LSTMTuple[T] = Tuple[Output[T], LSTMState[T]]

  def LSTMTuple[T](
      output: Output[T],
      state: LSTMState[T]
  ): LSTMTuple[T] = {
    Tuple(output, state)
  }

  private[rnn] trait API {
    type RNNCell[O, OS, S, SS] = cell.RNNCell[O, OS, S, SS]
    type BasicRNNCell[T] = cell.BasicRNNCell[T]
    type GRUCell[T] = cell.GRUCell[T]
    type BasicLSTMCell[T] = cell.BasicLSTMCell[T]
    type LSTMCell[T] = cell.LSTMCell[T]
    type DeviceWrapper[O, OS, S, SS] = cell.DeviceWrapper[O, OS, S, SS]
    type DropoutWrapper[O, OS, S, SS] = cell.DropoutWrapper[O, OS, S, SS]
    type ResidualWrapper[O, OS, S, SS] = cell.ResidualWrapper[O, OS, S, SS]
    type MultiCell[O, OS, S, SS] = cell.StackedCell[O, OS, S, SS]

    val BasicRNNCell   : cell.BasicRNNCell.type    = cell.BasicRNNCell
    val GRUCell        : cell.GRUCell.type         = cell.GRUCell
    val BasicLSTMCell  : cell.BasicLSTMCell.type   = cell.BasicLSTMCell
    val LSTMCell       : cell.LSTMCell.type        = cell.LSTMCell
    val DeviceWrapper  : cell.DeviceWrapper.type   = cell.DeviceWrapper
    val DropoutWrapper : cell.DropoutWrapper.type  = cell.DropoutWrapper
    val ResidualWrapper: cell.ResidualWrapper.type = cell.ResidualWrapper
    val MultiCell      : cell.StackedCell.type     = cell.StackedCell

    type LSTMState[T] = cell.LSTMState[T]

    val LSTMState: cell.LSTMState.type = cell.LSTMState

    type RNNTuple[O, S] = cell.Tuple[O, S]
    type BasicTuple[T] = cell.Tuple[Output[T], Output[T]]
    type LSTMTuple[T] = cell.Tuple[Output[T], LSTMState[T]]

    val RNNTuple: cell.Tuple.type = cell.Tuple

    def LSTMTuple[T](output: Output[T], state: LSTMState[T]): LSTMTuple[T] = {
      cell.Tuple(output, state)
    }
  }
}
