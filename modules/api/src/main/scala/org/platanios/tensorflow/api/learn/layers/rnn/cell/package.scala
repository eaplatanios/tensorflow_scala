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

package org.platanios.tensorflow.api.learn.layers.rnn

import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
package object cell {
  private[cell] val KERNEL_NAME: String = "Weights"
  private[cell] val BIAS_NAME  : String = "Bias"

  type Tuple[O, S] = ops.rnn.cell.Tuple[O, S]
  type BasicTuple[T] = ops.rnn.cell.BasicTuple[T]

  val Tuple: ops.rnn.cell.Tuple.type = ops.rnn.cell.Tuple

  type LSTMState[T] = ops.rnn.cell.LSTMState[T]

  val LSTMState: ops.rnn.cell.LSTMState.type = ops.rnn.cell.LSTMState

  type LSTMTuple[T] = ops.rnn.cell.LSTMTuple[T]

  def LSTMTuple[T](output: Output[T], state: LSTMState[T]): LSTMTuple[T] = {
    ops.rnn.cell.LSTMTuple(output, state)
  }

  private[rnn] trait API {
    type RNNCell[O, S] = cell.RNNCell[O, S]
    type BasicRNNCell[T] = cell.BasicRNNCell[T]
    type GRUCell[T] = cell.GRUCell[T]
    type BasicLSTMCell[T] = cell.BasicLSTMCell[T]
    type LSTMCell[T] = cell.LSTMCell[T]
    type DeviceWrapper[O, S] = cell.DeviceWrapper[O, S]
    type DropoutWrapper[O, S] = cell.DropoutWrapper[O, S]
    type ResidualWrapper[O, S] = cell.ResidualWrapper[O, S]
    type StackedCell[O, S] = cell.StackedCell[O, S]

    val BasicRNNCell   : cell.BasicRNNCell.type    = cell.BasicRNNCell
    val GRUCell        : cell.GRUCell.type         = cell.GRUCell
    val BasicLSTMCell  : cell.BasicLSTMCell.type   = cell.BasicLSTMCell
    val LSTMCell       : cell.LSTMCell.type        = cell.LSTMCell
    val DeviceWrapper  : cell.DeviceWrapper.type   = cell.DeviceWrapper
    val DropoutWrapper : cell.DropoutWrapper.type  = cell.DropoutWrapper
    val ResidualWrapper: cell.ResidualWrapper.type = cell.ResidualWrapper
    val StackedCell    : cell.StackedCell.type     = cell.StackedCell

    type RNNTuple[O, S] = cell.Tuple[O, S]
    type BasicTuple[T] = cell.BasicTuple[T]
    type LSTMTuple[T] = cell.LSTMTuple[T]

    type LSTMState[T] = cell.LSTMState[T]

    val LSTMState: cell.LSTMState.type = cell.LSTMState

    val RNNTuple: cell.Tuple.type = cell.Tuple

    def LSTMTuple[T](output: Output[T], state: LSTMState[T]): LSTMTuple[T] = {
      cell.LSTMTuple(output, state)
    }
  }
}
