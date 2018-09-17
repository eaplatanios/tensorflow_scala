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

import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, Output}
import org.platanios.tensorflow.api.types.DataType

/** RNN cell that ensures another RNN cell runs on a specific device.
  *
  * @param  cell           RNN cell being wrapped.
  * @param  device         Device to use.
  * @param  deviceFunction Device function to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class DeviceWrapper[O, OS, S, SS] protected (
    val cell: RNNCell[O, OS, S, SS],
    val device: String = "",
    val deviceFunction: OpSpecification => String = _.device
)(implicit
    evO: WhileLoopVariable.Aux[O, OS],
    evS: WhileLoopVariable.Aux[S, SS]
) extends RNNCell[O, OS, S, SS]()(evO, evS) {
  override def outputShape: OS = cell.outputShape
  override def stateShape: SS = cell.stateShape

  override def zeroState(batchSize: Output, dataType: DataType, shape: SS, name: String = "ZeroState"): S = {
    Op.createWith(device = device, deviceFunction = deviceFunction) {
      super.zeroState(batchSize, dataType, shape, name)
    }
  }

  override def forward(input: Tuple[O, S]): Tuple[O, S] = {
    Op.createWith(device = device, deviceFunction = deviceFunction) {
      cell.forward(input)
    }
  }
}

object DeviceWrapper {
  def apply[O, OS, S, SS](
      cell: RNNCell[O, OS, S, SS], device: String = "", deviceFunction: OpSpecification => String = _.device
  )(implicit
      evO: WhileLoopVariable.Aux[O, OS],
      evS: WhileLoopVariable.Aux[S, SS]
  ): DeviceWrapper[O, OS, S, SS] = {
    new DeviceWrapper(cell, device, deviceFunction)(evO, evS)
  }
}
