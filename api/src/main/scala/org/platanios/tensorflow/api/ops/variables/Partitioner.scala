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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.types.DataType

/** A variable partitioner is simply a function that accepts the `DataType` and the fully defined `Shape` of the
  * variable to be created, and returns an array of integers corresponding to the number of partitions for each axis
  * (currently only one axis can be partitioned).
  *
  * @author Emmanouil Antonios Platanios
  */
trait Partitioner {
  def apply(dataType: DataType, shape: Shape): Array[Int]
}
