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

import org.platanios.tensorflow.api.implicits.helpers.{NestedStructure, Zero}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification, Output}

/** RNN cell that ensures another RNN cell runs on a specific device.
  *
  * @param  cell           RNN cell being wrapped.
  * @param  device         Device to use.
  * @param  deviceFunction Device function to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class DeviceWrapper[O, S] protected (
    val cell: RNNCell[O, S],
    val device: String = "",
    val deviceFunction: OpSpecification => String = _.device
) extends RNNCell[O, S]() {
  override def outputShape[OS](implicit evStructureO: NestedStructure.Aux[O, _, _, OS]): OS = {
    cell.outputShape
  }

  override def stateShape[SS](implicit evStructureS: NestedStructure.Aux[S, _, _, SS]): SS = {
    cell.stateShape
  }

  override def zeroState(
      batchSize: Output[Int],
      name: String
  )(implicit evZeroS: Zero[S]): S = {
    Op.device(device, deviceFunction) {
      super.zeroState(batchSize, name)
    }
  }

  override def forward(input: Tuple[O, S]): Tuple[O, S] = {
    Op.device(device, deviceFunction) {
      cell.forward(input)
    }
  }
}

object DeviceWrapper {
  def apply[O, S](
      cell: RNNCell[O, S],
      device: String = "",
      deviceFunction: OpSpecification => String = _.device
  ): DeviceWrapper[O, S] = {
    new DeviceWrapper(cell, device, deviceFunction)
  }
}
