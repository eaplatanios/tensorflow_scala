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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.{Op, Output, OutputLike, OutputOps}
import org.platanios.tensorflow.api.types.DataType

/** Lookup table that persists across different session runs.
  *
  * @param  keysDataType   Data type of the table keys.
  * @param  valuesDataType Data type of the table values.
  * @param  name          Name of this lookup table.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class LookupTable(val keysDataType: DataType, val valuesDataType: DataType, val name: String) {
  /** Creates an op used to initialize this table.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def initialize(name: String = "Initialize"): Op

  /** Creates an op that computes the number of elements in this table.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def size(name: String = "Size"): Output

  /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
    *
    * @param  keys Tensor containing the keys to look up.
    * @param  name Name for the created op.
    * @return Created op output.
    * @throws InvalidDataTypeException If the provided keys data types does not match the keys data type of this table.
    */
  @throws[InvalidDataTypeException]
  def lookup[T <: OutputLike : OutputOps](keys: T, name: String = "Lookup"): T

  /** Checks that the provided keys and values data types match those expected for this lookup table and throws an
    * `InvalidDataTypeException` if they do not.
    *
    * @param  keysDataType   Provided keys data type to check.
    * @param  valuesDataType Provided values data type to check.
    * @throws InvalidDataTypeException If any of the provided data type does not match the corresponding expected type.
    */
  @throws[InvalidDataTypeException]
  def checkDataTypes(keysDataType: DataType, valuesDataType: DataType): Unit = {
    if (keysDataType != this.keysDataType)
      throw InvalidDataTypeException(s"Invalid keys data type $keysDataType (expected ${this.keysDataType}).")
    if (valuesDataType != this.valuesDataType)
      throw InvalidDataTypeException(s"Invalid values data type $valuesDataType (expected ${this.valuesDataType}).")
  }
}

/** Initializable lookup table that is constructed from an existing lookup table handle.
  *
  * Note that even though the caller needs to provide an initializer for this table, the caller also needs to make sure
  * to execute the initialization op.
  *
  * @param  handle       Resource handle to a lookup table.
  * @param  initializer  Lookup table initializer to use.
  * @param  defaultValue Default value to use if a key is missing from the table.
  */
abstract class InitializableLookupTable private[lookup] (
    val handle: Output,
    protected val initializer: LookupTableInitializer,
    val defaultValue: Output
) extends LookupTable(initializer.keysDataType, initializer.valuesDataType, handle.op.name.split("/").last) {
  // Make sure that the provided default value is a scalar
  defaultValue.shape.mergeWith(Shape.scalar())
  initialize()

  /** Creates and returns an op used to initialize this table.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def initialize(name: String = "Initialize"): Op = initializer.initialize(this, s"${this.name}/$name")

  /** Creates an op that computes the number of elements in this table.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def size(name: String = "Size"): Output = Lookup.lookupTableSize(handle, s"${this.name}/$name")

  /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
    *
    * @param  keys Tensor containing the keys to look up.
    * @param  name Name for the created op.
    * @return Created op output.
    * @throws InvalidDataTypeException If the provided keys data types does not match the keys data type of this table.
    */
  @throws[InvalidDataTypeException]
  def lookup[T <: OutputLike : OutputOps](keys: T, name: String = "Lookup"): T = Op.createWithNameScope(name) {
    if (keys.dataType != keysDataType)
      throw InvalidDataTypeException(s"Invalid keys data type ${keys.dataType} (expected $keysDataType).")
    implicitly[OutputOps[T]].applyUnary(keys, o => {
      val values = Lookup.lookupTableFind(handle, o, defaultValue)
      values.setShape(o.shape)
      values
    })
  }
}

/** Generic hash table implementation for lookup tables.
  *
  * The constructor creates a hash table, specifying the type of its keys and values. Before using the table the caller
  * will have to initialize it. After initialization the table will be immutable.
  *
  * Example usage:
  * {{{
  *   val table = HashTable(LookupTableTensorInitializer(keys, values), -1)
  *   val output = table.lookup(input)
  *   // Can now run evaluate the `output` tensor after executing the table initializer.
  * }}}
  *
  * @param  initializer        Lookup table initializer to use.
  * @param  defaultValue       Default value to use if a key is missing from the table.
  * @param  container          If non-empty, the created table is placed in the given container. Otherwise, a
  *                            default container is used.
  * @param  sharedName         If non-empty, the created table is named in the given bucket with this shared name.
  *                            Otherwise, the op name is used, instead.
  * @param  useNodeNameSharing If set to `true` and `sharedName` is empty, the table is shared using the node name.
  * @param  name               Name for the created table.
  */
case class HashTable(
    override protected val initializer: LookupTableInitializer,
    override val defaultValue: Output,
    container: String = "",
    sharedName: String = "",
    useNodeNameSharing: Boolean = false,
    override val name: String = "HashTable"
) extends InitializableLookupTable(
  Lookup.createHashTable(
    initializer.keysDataType, initializer.valuesDataType, container, sharedName, useNodeNameSharing, name),
  initializer, defaultValue)
