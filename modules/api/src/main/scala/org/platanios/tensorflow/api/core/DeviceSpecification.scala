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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.exception.InvalidDeviceException

import scala.util.matching.Regex

/** Represents a (possibly partial) specification for a TensorFlow device.
  *
  * Device specifications are used throughout TensorFlow to describe where state is stored and computations occur.
  *
  * @example {{{
  *   createWith(device = "/GPU:0") {
  *     // All ops constructed in this code block will be assigned to '/device:GPU:0'
  *     val c = constant(1.0)
  *     assert(c.device == "/device:GPU:0")
  *   }
  * }}}
  *
  * If a [[DeviceSpecification]] is partially specified, it will be merged with other [[DeviceSpecification]]s according
  * to the scope in which it is defined. [[DeviceSpecification]] components defined in inner scopes take precedence over
  * those defined in outer scopes.
  *
  * @example {{{
  *   createWith(device = "/GPU:0") {
  *     // All ops constructed in this code block will be assigned to '/device:GPU:0'
  *     val c1 = constant(1.0)
  *     assert(c1.device == "/device:GPU:0")
  *
  *     // Reset the device being used
  *     createWith(device = "/job:ps") {
  *       // All ops constructed in this code block will be assigned to '/job:ps/device:GPU:0'
  *       val c2 = constant(2.0)
  *       assert(c2.device == "/job:ps/device:GPU:0")
  *       createWith(device = "/job:train/device:GPU:1") {
  *         // All ops constructed in this code block will be assigned to '/job:train/device:GPU:1'
  *         val c3 = constant(3.0)
  *         assert(c3.device == "/job:train/device:GPU:1")
  *       }
  *     }
  *   }
  * }}}
  *
  * A [[DeviceSpecification]] consists of 5 components, each of which is optionally specified.
  *
  * @param  job         Job name.
  * @param  replica     Replica index.
  * @param  task        Task index.
  * @param  deviceType  Device type string (e.g., "CPU" or "GPU").
  * @param  deviceIndex Device index.
  * @return Constructed device specification.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] case class DeviceSpecification(
    job: String = null,
    replica: Int = -1,
    task: Int = -1,
    deviceType: String = null,
    deviceIndex: Int = -1
) {
  /** Returns a string representation of this device, of the form:
    *
    * `/job:<name>/replica:<id>/task:<id>/device:<device_type>:<id>`.
    *
    * @return String representation of the device.
    */
  override def toString: String = {
    val stringBuilder: StringBuilder = new StringBuilder()
    if (job != null)
      stringBuilder ++= s"/job:$job"
    if (replica != -1)
      stringBuilder ++= s"/replica:$replica"
    if (task != -1)
      stringBuilder ++= s"/task:$task"
    if (deviceType != null) {
      if (deviceIndex != -1)
        stringBuilder ++= s"/device:$deviceType:$deviceIndex"
      else
        stringBuilder ++= s"/device:$deviceType:*"
    }
    stringBuilder.toString()
  }
}

/** Contains helper functions for dealing with TensorFlow device specifications.
  *
  * Using [[DeviceSpecification]] allows you to parse device specification strings, verify the validity of device
  * specifications, merge them, or compose their string representation programmatically.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] object DeviceSpecification {
  private val deviceStringRegex: Regex = {
    """
      #^(?:/job:([^/:]+))?
      #(?:/replica:([[0-9]&&[^\:]]+))?
      #(?:/task:([[0-9]&&[^\:]]+))?
      #(?:(?:/device:([^/:]+):([[0-9*]&&[^\:]]+))
      #|(?:/?([^/:]+):([[0-9*]&&[^\:]]+)))?$"""
        .stripMargin('#')
        .replaceAll("\n", "")
        .r("job", "replica", "task", "deviceType", "deviceIndex", "deviceTypeShort", "deviceIndexShort")
  }

  /** Parses a [[DeviceSpecification]] specification from the provided string.
    *
    * The string being parsed must comply with the following regular expression (otherwise an
    * [[InvalidDeviceException]] exception is thrown):
    *
    * {{{
    *   ^(?:/job:([^/:]+))?
    *   (?:/replica:([[0-9]&&[^\:]]+))?
    *   (?:/task:([[0-9]&&[^\:]]+))?
    *   (?:(?:/device:([^/:]+):([[0-9*]&&[^\:]]+))|(?:/([^/:]+):([[0-9*]&&[^\:]]+)))?$
    * }}}
    *
    * Valid string examples:
    *   - `/job:job1/replica:1/task:22/device:CPU:0`
    *   - `/task:22/device:GPU:0`
    *   - `/CPU:1`
    *
    * @param  device String to parse.
    * @return Device specification parsed from string.
    * @throws InvalidDeviceException If the provided string does not match the appropriate regular expression.
    */
  @throws[InvalidDeviceException]
  private[api] def fromString(device: String): DeviceSpecification = {
    device match {
      case deviceStringRegex(job, replica, task, deviceType, deviceIndex, deviceTypeShort, deviceIndexShort)
        if (replica == null || replica.matches("\\d+")) &&
            (task == null || task.matches("\\d+")) &&
            (deviceIndex == null || deviceIndex == "*" || deviceIndex.matches("""\d+""")) &&
            (deviceIndexShort == null || deviceIndexShort == "*" || deviceIndexShort.matches("""\d+""")) =>
        DeviceSpecification(
          job = job,
          replica = if (replica == null) -1 else replica.toInt,
          task = if (task == null) -1 else task.toInt,
          deviceType = {
            if (deviceType == null && deviceTypeShort == null)
              null
            else if (deviceType != null)
              deviceType.toUpperCase
            else
              deviceTypeShort.toUpperCase
          },
          deviceIndex = {
            if ((deviceIndex == null && deviceIndexShort == null) || deviceIndex == "*" || deviceIndexShort == "*")
              -1
            else if (deviceIndex == null)
              deviceIndexShort.toInt
            else
              deviceIndex.toInt
          }
        )
      case _ => throw InvalidDeviceException(s"Invalid device specification '$device'.")
    }
  }

  /** Merges the properties of `spec2` into those of `spec1` and returns the result as a new [[DeviceSpecification]].
    *
    * @param  dev1 First device specification being merged.
    * @param  dev2 Second device specification being merged.
    * @return Merged device specification.
    */
  private[api] def merge(dev1: DeviceSpecification, dev2: DeviceSpecification): DeviceSpecification = {
    DeviceSpecification(
      job = if (dev2.job != null) dev2.job else dev1.job,
      replica = if (dev2.replica != -1) dev2.replica else dev1.replica,
      task = if (dev2.task != -1) dev2.task else dev1.task,
      deviceType = if (dev2.deviceType != null) dev2.deviceType else dev1.deviceType,
      deviceIndex = if (dev2.deviceIndex != -1) dev2.deviceIndex else dev1.deviceIndex
    )
  }
}
