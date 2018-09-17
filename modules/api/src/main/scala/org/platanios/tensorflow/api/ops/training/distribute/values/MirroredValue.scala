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
import org.platanios.tensorflow.api.ops.training.distribute.Distributable

/** Represents a synchronous value mirrored among multiple devices.
  *
  * @author Emmanouil Antonios Platanios
  */
class MirroredValue[+T: Distributable] protected (
    override val index: Map[DeviceSpecification, T]
) extends DistributedValue[T](index, DistributedValue.Mirrored)

object MirroredValue {
  def apply[T: Distributable](index: Map[DeviceSpecification, T]): MirroredValue[T] = {
    new MirroredValue[T](index)
  }
}
