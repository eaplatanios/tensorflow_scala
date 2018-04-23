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

package org.platanios.tensorflow.api.ops.training.distribute.values

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.training.distribute
import org.platanios.tensorflow.api.ops.training.distribute._
import org.platanios.tensorflow.api.ops.training.distribute.strategies.{DistributionContext, InTowerContext}

/** Holds a map from devices to values.
  *
  * @param  index            Index map from devices to values.
  * @param  distributionType Type of this distributed value (e.g., per-device or mirrored).
  *
  * @author Emmanouil Antonios Platanios
  */
class DistributedValue[T: Distributable] protected (
    val index: Map[DeviceSpecification, T],
    val distributionType: DistributedValue.Type
) {
  /** Returns the devices on which this value is distributed. */
  def devices: Seq[DeviceSpecification] = index.keys.toSeq

  /** Returns the value on the specified device (defaults to the current device, if not provided. */
  def get(device: String = "current")(implicit context: DistributionContext): T = {
    if (device == "current")
      index(DeviceSpecification.fromString(DistributedValue.currentDevice))
    else
      index(DeviceSpecification.fromString(device))
  }

  /** Returns true if the values are distributed on the provided device. */
  def onDevice(device: String): Boolean = {
    index.contains(DeviceSpecification.fromString(device))
  }
}

object DistributedValue {
  def apply[T: Distributable](
      index: Map[DeviceSpecification, T],
      distributionType: DistributedValue.Type
  ): DistributedValue[T] = {
    new DistributedValue(index, distributionType)
  }

  def mirrored[T: Distributable](
      index: Map[DeviceSpecification, T]
  ): DistributedValue[T] = {
    MirroredValue(index)
  }

  def perDevice[T: Distributable](
      index: Map[DeviceSpecification, T]
  ): DistributedValue[T] = {
    PerDeviceValue(index)
  }

  private[distribute] def currentDevice(implicit context: DistributionContext): String = {
    context match {
      case _: InTowerContext => distribute.currentDevice
      case _ => currentUpdateDevice.getOrElse(distribute.currentDevice)
    }
  }

  /** Represents the type of a distributed value. */
  sealed trait Type

  /** Represents a non-synchronized distributed value. */
  case object PerDevice extends Type

  /** Represents a synchronized distributed value. */
  case object Mirrored extends Type
}
