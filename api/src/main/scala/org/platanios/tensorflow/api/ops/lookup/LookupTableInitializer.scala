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

package org.platanios.tensorflow.api.ops.lookup

import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.types.DataType

/** Lookup table initializer.
  *
  * @param  keysDataType   Data type of the table keys.
  * @param  valuesDataType Data type of the table values.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class LookupTableInitializer(
    val keysDataType: DataType,
    val valuesDataType: DataType
) {
  /** Creates and returns an op that initializes the provided table.
    *
    * @param  table Table to initialize.
    * @return Created initialization op for `table`.
    */
  def initialize(table: InitializableLookupTable, name: String = "Initialize"): Op
}
