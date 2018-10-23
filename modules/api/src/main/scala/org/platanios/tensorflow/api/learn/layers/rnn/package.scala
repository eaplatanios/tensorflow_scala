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

package org.platanios.tensorflow.api.learn.layers

/**
  * @author Emmanouil Antonios Platanios
  */
package object rnn {
  private[layers] trait API
      extends rnn.cell.API {
    type RNN[Out, State, OutShape, StateShape] = rnn.RNN[Out, State, OutShape, StateShape]
    type BidirectionalRNN[Out, State, OutShape, StateShape] = rnn.BidirectionalRNN[Out, State, OutShape, StateShape]

    val RNN             : rnn.RNN.type              = rnn.RNN
    val BidirectionalRNN: rnn.BidirectionalRNN.type = rnn.BidirectionalRNN
  }
}
