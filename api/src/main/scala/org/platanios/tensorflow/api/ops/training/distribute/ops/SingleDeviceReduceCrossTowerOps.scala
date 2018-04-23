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

package org.platanios.tensorflow.api.ops.training.distribute.ops

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.values.{MirroredValue, PerDeviceValue}
import org.platanios.tensorflow.api.ops.{Math, Output}

/** Cross-tower ops that always perform a reduction to one device first and then do broadcasting.
  *
  * Batch reduction is done by reduction on each element one by one.
  *
  * @param  device       Intermediate device to reduce to. If `None`, reduce to the first device in the provided
  *                      destination of the `reduce()` method.
  * @param  accumulateFn Accumulation function to use.
  *
  * @author Emmanouil Antonios Platanios
  */
class SingleDeviceReduceCrossTowerOps protected(
    val device: Option[DeviceSpecification],
    val accumulateFn: Seq[Output] => Output
) extends CrossTowerOps {
  /** Reduces `value` to `destination`.
    *
    * It runs the reduction operation defined by `reduction` and puts the result on `destination`.
    *
    * @param  reduction   Reduction method.
    * @param  value       Per-device value to be reduced.
    * @param  destination Reduction destination.
    * @return Reduced mirrored value.
    */
  def reduce[D: Destination](
      reduction: Reduction,
      value: PerDeviceValue[Output],
      destination: Option[D]
  ): MirroredValue[Output] = {
    val devices = destination match {
      case Some(d) => Destination.devicesFrom(d)
      case None => value.devices
    }
    val reduceToDevice = device.getOrElse(devices.head)
    val reduced = CrossTowerOps.simpleReduce(value, reduceToDevice, reduction)
    broadcast(reduced, devices)
  }

  /** Reduces a batch per-device values to the corresponding provided destinations.
    *
    * For each tuple in `valueDestinationPairs`, this method reduces each first element to each second element which
    * indicates the corresponding destination.
    *
    * @param  reduction             Reduction method.
    * @param  valueDestinationPairs Sequence of per-device values and destinations pairs. If a destination is `None`,
    *                               then the destination is set to match the devices of the corresponding per-device
    *                               value.
    * @return Sequence of reduced mirrored values.
    */
  def batchReduce[D: Destination](
      reduction: Reduction,
      valueDestinationPairs: Seq[(PerDeviceValue[Output], Option[D])]
  ): Seq[MirroredValue[Output]] = {
    valueDestinationPairs.map(p => reduce(reduction, p._1, p._2))
  }
}

object SingleDeviceReduceCrossTowerOps {
  def apply(
      device: Option[DeviceSpecification] = None,
      accumulateFn: Seq[Output] => Output = Math.addN(_, name = "ReductionAccumulate")
  ): SingleDeviceReduceCrossTowerOps = {
    new SingleDeviceReduceCrossTowerOps(device, accumulateFn)
  }
}
