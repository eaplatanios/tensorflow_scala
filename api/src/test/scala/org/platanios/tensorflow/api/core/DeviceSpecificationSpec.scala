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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.tf.InvalidDeviceException

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class DeviceSpecificationSpec extends FlatSpec with Matchers {
  "An device specification" must "be represented using an empty string when empty" in {
    assert(DeviceSpecification().toString === "")
    assert(DeviceSpecification.fromString("").toString === "")
  }

  it must "have a correctly functioning toString method implementation" in {
    assert(DeviceSpecification(job = "foo").toString === "/job:foo/replica:0/task:0/device:CPU:0")
    assert(DeviceSpecification(job = "foo", task = 3).toString === "/job:foo/replica:0/task:3/device:CPU:0")
    assert(DeviceSpecification(task = 3, deviceType = "CPU", deviceIndex = 1).toString
        === "/job:localhost/replica:0/task:3/device:CPU:1")
    assert(DeviceSpecification(job = "foo", replica = 12).toString === "/job:foo/replica:12/task:0/device:CPU:0")
    assert(DeviceSpecification(job = "foo", task = 3, deviceType = "CPU", deviceIndex = 0).toString
               === "/job:foo/replica:0/task:3/device:CPU:0")
    assert(DeviceSpecification(job = "foo", replica = 12, deviceType = "CPU", deviceIndex = 0).toString
               === "/job:foo/replica:12/task:0/device:CPU:0")
    assert(DeviceSpecification(job = "foo", replica = 12, deviceType = "GPU", deviceIndex = 2).toString
               === "/job:foo/replica:12/task:0/device:GPU:2")
    assert(DeviceSpecification(job = "foo", replica = 12, task = 3, deviceType = "GPU").toString
               === "/job:foo/replica:12/task:3/device:GPU:*")
  }

  it must "be correctly parsed from valid strings" in {
    assert(DeviceSpecification(job = "foo", replica = 0)
               === DeviceSpecification.fromString("/job:foo/replica:0"))
    assert(DeviceSpecification(job = "foo", task = 3, deviceType = "CPU")
               === DeviceSpecification.fromString("/job:foo/task:3/device:CPU:*"))
    assert(DeviceSpecification(job = "foo", replica = 12, task = 3, deviceType = "GPU", deviceIndex = 2)
               === DeviceSpecification.fromString("/job:foo/replica:12/task:3/device:GPU:2"))
    assert(DeviceSpecification(job = "foo", replica = 12, task = 3, deviceType = "GPU", deviceIndex = 2)
               === DeviceSpecification.fromString("/job:foo/replica:12/task:3/GPU:2"))
    assert(DeviceSpecification(deviceType = "GPU", deviceIndex = 1)
               === DeviceSpecification.fromString("/GPU:1"))
    assert(DeviceSpecification(deviceType = "CPU", deviceIndex = 0)
               === DeviceSpecification.fromString("/device:CPU:0"))
  }

  it must "be merged correctly to other device specifications" in {
    val dev1 = DeviceSpecification.fromString("/job:foo/replica:0")
    val dev2 = DeviceSpecification.fromString("/task:1/GPU:2")
    assert(DeviceSpecification.merge(dev1, dev2).toString === "/job:foo/replica:0/task:1/device:GPU:2")
    assert(dev1.toString === "/job:foo/replica:0/task:0/device:CPU:0")
    assert(dev2.toString === "/job:localhost/replica:0/task:1/device:GPU:2")
    val dev3 = DeviceSpecification()
    val dev4 = DeviceSpecification.merge(dev3, DeviceSpecification.fromString("/task:1/CPU:0"))
    assert(dev4.toString === "/job:localhost/replica:0/task:1/device:CPU:0")
    val dev5 = DeviceSpecification.merge(dev4, DeviceSpecification.fromString("/job:boo/GPU:0"))
    assert(dev5.toString === "/job:boo/replica:0/task:1/device:GPU:0")
    val dev6 = DeviceSpecification.merge(dev5, DeviceSpecification.fromString("/job:muu/CPU:2"))
    assert(dev6.toString === "/job:muu/replica:0/task:1/device:CPU:2")
    val dev7 = DeviceSpecification.merge(dev6, DeviceSpecification.fromString("/job:muu/device:MyFunnyDevice:2"))
    assert(dev7.toString === "/job:muu/replica:0/task:1/device:MYFUNNYDEVICE:2")
  }

  "An InvalidDeviceSpecificationException" must "be thrown when attempting to parse invalid strings" in {
    assertThrows[InvalidDeviceException](DeviceSpecification.fromString("/job:j/replica:foo"))
    assertThrows[InvalidDeviceException](DeviceSpecification.fromString("/job:j/task:bar"))
    assertThrows[InvalidDeviceException](DeviceSpecification.fromString("/bar:muu/baz:2"))
    assertThrows[InvalidDeviceException](DeviceSpecification.fromString("/CPU:0/GPU:2"))
  }
}
