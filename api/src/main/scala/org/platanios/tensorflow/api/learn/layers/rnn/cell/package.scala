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

package org.platanios.tensorflow.api.learn.layers.rnn

/**
  * @author Emmanouil Antonios Platanios
  */
package object cell {
  private[cell] val KERNEL_NAME: String = "weights"
  private[cell] val BIAS_NAME  : String = "bias"

  private[rnn] trait API {
    type RNNCell[O, OS, S, SS] = cell.RNNCell[O, OS, S, SS]
    type BasicRNNCell = cell.BasicRNNCell
    type GRUCell = cell.GRUCell
    type BasicLSTMCell = cell.BasicLSTMCell
    type LSTMCell = cell.LSTMCell

    val RNNCell      : cell.RNNCell.type       = cell.RNNCell
    val BasicRNNCell : cell.BasicRNNCell.type  = cell.BasicRNNCell
    val GRUCell      : cell.GRUCell.type       = cell.GRUCell
    val BasicLSTMCell: cell.BasicLSTMCell.type = cell.BasicLSTMCell
    val LSTMCell     : cell.LSTMCell.type      = cell.LSTMCell
  }
}
