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

import org.platanios.tensorflow.api.implicits.helpers.{OutputToShape, Zero}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, Output}

/** RNN cell that ensures another RNN cell runs on a specific device.
  *
  * @param  cell           RNN cell being wrapped.
  * @param  device         Device to use.
  * @param  deviceFunction Device function to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class DeviceWrapper[Out, State, OutShape, StateShape] protected (
    val cell: RNNCell[Out, State, OutShape, StateShape],
    val device: String = "",
    val deviceFunction: Option[OpSpecification => String] = None
)(implicit
    evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
    evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends RNNCell[Out, State, OutShape, StateShape] {
  override def outputShape: OutShape = {
    cell.outputShape
  }

  override def stateShape: StateShape = {
    cell.stateShape
  }

  override def zeroState(
      batchSize: Output[Int],
      name: String
  )(implicit evZeroS: Zero.Aux[State, StateShape]): State = {
    Op.device(device, deviceFunction) {
      cell.zeroState(batchSize, name)
    }
  }

  override def forward(input: Tuple[Out, State]): Tuple[Out, State] = {
    Op.device(device, deviceFunction) {
      cell.forward(input)
    }
  }
}

object DeviceWrapper {
  def apply[Out, State, OutShape, StateShape](
      cell: RNNCell[Out, State, OutShape, StateShape],
      device: String = "",
      deviceFunction: Option[OpSpecification => String] = None
  )(implicit
      evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
      evOutputToShapeState: OutputToShape.Aux[State, StateShape]
  ): DeviceWrapper[Out, State, OutShape, StateShape] = {
    new DeviceWrapper(cell, device, deviceFunction)
  }
}
