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
  type BasicTuple = Tuple[Output, Output]

  val Tuple: ops.rnn.cell.Tuple.type = ops.rnn.cell.Tuple

  type LSTMState = ops.rnn.cell.LSTMState

  val LSTMState: ops.rnn.cell.LSTMState.type = ops.rnn.cell.LSTMState

  type LSTMTuple = ops.rnn.cell.LSTMTuple

  def LSTMTuple(output: Output, state: LSTMState): LSTMTuple = ops.rnn.cell.LSTMTuple(output, state)

  private[rnn] trait API {
    type RNNCell[O, OS, S, SS] = cell.RNNCell[O, OS, S, SS]
    type BasicRNNCell = cell.BasicRNNCell
    type GRUCell = cell.GRUCell
    type BasicLSTMCell = cell.BasicLSTMCell
    type LSTMCell = cell.LSTMCell
    type DeviceWrapper[O, OS, S, SS] = cell.DeviceWrapper[O, OS, S, SS]
    type DropoutWrapper[O, OS, S, SS] = cell.DropoutWrapper[O, OS, S, SS]
    type ResidualWrapper[O, OS, S, SS] = cell.ResidualWrapper[O, OS, S, SS]
    type MultiCell[O, OS, S, SS] = cell.MultiCell[O, OS, S, SS]

    val BasicRNNCell   : cell.BasicRNNCell.type    = cell.BasicRNNCell
    val GRUCell        : cell.GRUCell.type         = cell.GRUCell
    val BasicLSTMCell  : cell.BasicLSTMCell.type   = cell.BasicLSTMCell
    val LSTMCell       : cell.LSTMCell.type        = cell.LSTMCell
    val DeviceWrapper  : cell.DeviceWrapper.type   = cell.DeviceWrapper
    val DropoutWrapper : cell.DropoutWrapper.type  = cell.DropoutWrapper
    val ResidualWrapper: cell.ResidualWrapper.type = cell.ResidualWrapper
    val MultiCell      : cell.MultiCell.type       = cell.MultiCell

    type RNNTuple[O, S] = cell.Tuple[O, S]
    type BasicTuple = cell.Tuple[Output, Output]

    type LSTMState = cell.LSTMState

    val LSTMState: cell.LSTMState.type = cell.LSTMState

    type LSTMTuple = cell.Tuple[Output, LSTMState]

    val RNNTuple: cell.Tuple.type = cell.Tuple

    def LSTMTuple(output: Output, state: LSTMState): LSTMTuple = cell.Tuple(output, state)
  }
}
