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

package org.platanios.tensorflow.api.learn.layers.rnn.cell

import org.platanios.tensorflow.api.implicits.helpers.OutputToShape
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}

/** RNN cell that ensures another RNN cell runs on a specific device.
  *
  * @param  name           Name scope (also acting as variable scope) for this layer.
  * @param  cell           RNN cell being wrapped.
  * @param  device         Device to use.
  * @param  deviceFunction Device function to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class DeviceWrapper[Out, State](
    override val name: String,
    val cell: RNNCell[Out, State],
    val device: String = "",
    val deviceFunction: OpSpecification => String = _.device
) extends RNNCell[Out, State](name) {
  type OutShape = cell.OutShape
  type StateShape = cell.StateShape

  override def evOutputToShapeOut: OutputToShape.Aux[Out, OutShape] = cell.evOutputToShapeOut
  override def evOutputToShapeState: OutputToShape.Aux[State, StateShape] = cell.evOutputToShapeState

  override val layerType: String = "DeviceWrapper"

  override def createCellWithoutContext(
      mode: Mode,
      inputShape: OutShape
  ): ops.rnn.cell.RNNCell[Out, State] = {
    Op.createWith(device = device, deviceFunction = deviceFunction) {
      ops.rnn.cell.DeviceWrapper(cell.createCellWithoutContext(mode, inputShape), device, deviceFunction)
    }
  }
}

object DeviceWrapper {
  def apply[Out, State](
      variableScope: String,
      cell: RNNCell[Out, State],
      device: String = "",
      deviceFunction: OpSpecification => String = _.device
  ): DeviceWrapper[Out, State] = {
    new DeviceWrapper(variableScope, cell, device, deviceFunction)
  }
}
