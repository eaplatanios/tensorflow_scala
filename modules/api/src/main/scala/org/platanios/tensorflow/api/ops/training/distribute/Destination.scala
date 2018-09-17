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

package org.platanios.tensorflow.api.ops.training.distribute

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.{Op, OutputLike}
import org.platanios.tensorflow.api.ops.training.distribute.values.{DistributedValue, PerDeviceValue}

/**
  * @author Emmanouil Antonios Platanios
  */
trait Destination[T] {
  /** Returns all the devices used or represented by the provided value. */
  def devices(value: T): Seq[DeviceSpecification]
}

object Destination {
  implicit val stringDestination: Destination[String] = new Destination[String] {
    override def devices(value: String): Seq[DeviceSpecification] = {
      Seq(DeviceSpecification.fromString(value))
    }
  }

  implicit val deviceSpecificationDestination: Destination[DeviceSpecification] = {
    new Destination[DeviceSpecification] {
      override def devices(value: DeviceSpecification): Seq[DeviceSpecification] = {
        Seq(value)
      }
    }
  }

  implicit val opDestination: Destination[Op] = new Destination[Op] {
    override def devices(value: Op): Seq[DeviceSpecification] = {
      Seq(DeviceSpecification.fromString(value.device))
    }
  }

  implicit def outputLikeDestination[O](implicit ev: O => OutputLike): Destination[O] = new Destination[O] {
    override def devices(value: O): Seq[DeviceSpecification] = {
      Seq(DeviceSpecification.fromString(ev(value).device))
    }
  }

  implicit def distributedValueDestination[T, V](implicit ev: V => DistributedValue[T]): Destination[V] = {
    new Destination[V] {
      override def devices(value: V): Seq[DeviceSpecification] = {
        ev(value).devices
      }
    }
  }

  implicit def seqDestination[T: Destination]: Destination[Seq[T]] = {
    new Destination[Seq[T]] {
      /** Returns all the devices used or represented by the provided value. */
      override def devices(value: Seq[T]): Seq[DeviceSpecification] = {
        value.flatMap(implicitly[Destination[T]].devices(_))
      }
    }
  }

  /** Returns all the devices used or represented by the provided values. */
  def devicesFrom[D: Destination](value: D): Seq[DeviceSpecification] = {
    implicitly[Destination[D]].devices(value)
  }

  /** Returns `true` if the devices used or represented by the two provided values are the same. */
  def devicesMatch[D1: Destination, D2: Destination](value1: D1, value2: D2): Boolean = {
    devicesFrom(value1).toSet == devicesFrom(value2).toSet
  }

  /** Returns `true` if the devices used or represented by the provided values and destinations. */
  def allDevicesMatch[T: Distributable, D: Destination](
      valueDestinationPairs: Seq[(PerDeviceValue[T], Option[D])]
  ): Boolean = {
    valueDestinationPairs.forall(pair => pair._2.isEmpty || devicesMatch(pair._1, pair._2.get)) &&
        valueDestinationPairs.tail.map(_._1).forall(value => devicesMatch(value, valueDestinationPairs.head._1))
  }
}
