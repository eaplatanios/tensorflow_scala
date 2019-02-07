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
class DeviceWrapper[Out, State, OutShape, StateShape](
    override val name: String,
    val cell: RNNCell[Out, State, OutShape, StateShape],
    val device: String = "",
    val deviceFunction: Option[OpSpecification => String] = None
)(implicit
    evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
    evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends RNNCell[Out, State, OutShape, StateShape](name) {
  override val layerType: String = "DeviceWrapper"

  override def createCellWithoutContext(
      mode: Mode,
      inputShape: OutShape
  ): ops.rnn.cell.RNNCell[Out, State, OutShape, StateShape] = {
    Op.createWith(device = device, deviceFunction = deviceFunction) {
      ops.rnn.cell.DeviceWrapper(cell.createCellWithoutContext(mode, inputShape), device, deviceFunction)
    }
  }
}

object DeviceWrapper {
  def apply[Out, State, OutShape, StateShape](
      variableScope: String,
      cell: RNNCell[Out, State, OutShape, StateShape],
      device: String = "",
      deviceFunction: Option[OpSpecification => String] = None
  )(implicit
      evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
      evOutputToShapeState: OutputToShape.Aux[State, StateShape]
  ): DeviceWrapper[Out, State, OutShape, StateShape] = {
    new DeviceWrapper(variableScope, cell, device, deviceFunction)
  }
}
