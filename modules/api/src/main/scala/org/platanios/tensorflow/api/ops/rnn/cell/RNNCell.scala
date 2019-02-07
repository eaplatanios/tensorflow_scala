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

import org.platanios.tensorflow.api.implicits.helpers.{OutputToShape, Zero}
import org.platanios.tensorflow.api.ops.Output

/** Contains functions for constructing ops related to recurrent neural network (RNN) cells.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNCell[Out, State, OutShape, StateShape](implicit
    val evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
    val evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) {
  def outputShape: OutShape
  def stateShape: StateShape

  def zeroOutput(
      batchSize: Output[Int],
      name: String = "ZeroOutput"
  )(implicit evZero: Zero.Aux[Out, OutShape]): Out = {
    evZero.zero(batchSize, outputShape, name)
  }

  def zeroState(
      batchSize: Output[Int],
      name: String = "ZeroState"
  )(implicit evZero: Zero.Aux[State, StateShape]): State = {
    evZero.zero(batchSize, stateShape, name)
  }

  @throws[IllegalArgumentException]
  def forward(input: Tuple[Out, State]): Tuple[Out, State]

  def apply(input: Tuple[Out, State]): Tuple[Out, State] = {
    forward(input)
  }
}
