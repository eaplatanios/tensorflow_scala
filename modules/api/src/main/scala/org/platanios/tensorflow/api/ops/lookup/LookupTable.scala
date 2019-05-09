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

package org.platanios.tensorflow.api.ops.lookup

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.{DataType, Resource, TF}
import org.platanios.tensorflow.api.ops.{Op, Output, OutputLike, OutputOps, UntypedOp}

/** Lookup table that persists across different session runs.
  *
  * @param  keysDataType   Data type of the table keys.
  * @param  valuesDataType Data type of the table values.
  * @param  name          Name of this lookup table.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class LookupTable[K: TF, V: TF](
    val keysDataType: DataType[K],
    val valuesDataType: DataType[V],
    val name: String
) {
  /** Creates an op used to initialize this table.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def initialize(name: String = "Initialize"): UntypedOp

  /** Creates an op that computes the number of elements in this table.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def size(name: String = "Size"): Output[Long]

  /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
    *
    * @param  keys Tensor containing the keys to look up.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def lookup[OL[A] <: OutputLike[A]](
      keys: OL[K],
      name: String = "Lookup"
  )(implicit ev: OutputOps.Aux[OL, K]): OL[V]
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
abstract class InitializableLookupTable[K: TF, V: TF] private[lookup](
    val handle: Output[Resource],
    protected val initializer: LookupTableInitializer[K, V],
    val defaultValue: Output[V]
) extends LookupTable(
  initializer.keysDataType,
  initializer.valuesDataType,
  handle.op.name.split("/").last
) {
  // Make sure that the provided default value is a scalar
  defaultValue.shape.mergeWith(Shape.scalar())
  initialize()

  /** Creates and returns an op used to initialize this table.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  override def initialize(name: String = "Initialize"): UntypedOp = {
    initializer.initialize(this, name = s"${this.name}/$name")
  }

  /** Creates an op that computes the number of elements in this table.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  override def size(name: String = "Size"): Output[Long] = {
    Op.Builder[Output[Resource], Output[Long]](
      opType = "LookupTableSizeV2",
      name = s"${this.name}/$name",
      input = handle
    ).build().output
  }

  /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
    *
    * @param  keys Tensor containing the keys to look up.
    * @param  name Name for the created op.
    * @return Created op output.
    */
  override def lookup[OL[A] <: OutputLike[A]](
      keys: OL[K],
      name: String = "Lookup"
  )(implicit ev: OutputOps.Aux[OL, K]): OL[V] = {
    Op.nameScope(name) {
      ev.applyUnary(keys, o => {
        val values = Op.Builder[(Output[Resource], Output[K], Output[V]), Output[V]](
          opType = "LookupTableFindV2",
          name = name,
          input = (handle, o, defaultValue)
        ).build().output
        values.setShape(o.shape)
        values
      })
    }
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
case class HashTable[K: TF, V: TF](
    override protected val initializer: LookupTableInitializer[K, V],
    override val defaultValue: Output[V],
    container: String = "",
    sharedName: String = "",
    useNodeNameSharing: Boolean = false,
    override val name: String = "HashTable"
) extends InitializableLookupTable(
  handle = HashTable.createHashTable(
    initializer.keysDataType,
    initializer.valuesDataType,
    container,
    sharedName,
    useNodeNameSharing,
    name),
  initializer = initializer,
  defaultValue = defaultValue)

object HashTable {
  /** Creates an op that creates a non-initialized hash table.
    *
    * The op creates a hash table, specifying the type of its keys and values. Before using the table the caller will
    * have to initialize it. After initialization the table will be immutable.
    *
    * @param  keysDataType       Data type for the keys of the table.
    * @param  valuesDataType     Data type for the values of the table.
    * @param  container          If non-empty, the created table is placed in the given container. Otherwise, a
    *                            default container is used.
    * @param  sharedName         If non-empty, the created table is named in the given bucket with this shared name.
    *                            Otherwise, the op name is used, instead.
    * @param  useNodeNameSharing If set to `true` and `sharedName` is empty, the table is shared using the node name.
    * @param  name               Name for the created op.
    * @return Created op.
    */
  private[lookup] def createHashTable[K: TF, V: TF](
      keysDataType: DataType[K],
      valuesDataType: DataType[V],
      container: String = "",
      sharedName: String = "",
      useNodeNameSharing: Boolean = false,
      name: String = "HashTable"
  ): Output[Resource] = {
    Op.Builder[Unit, Output[Resource]](
      opType = "HashTableV2",
      name = name,
      input = ()
    ).setAttribute("key_dtype", keysDataType)
        .setAttribute("value_dtype", valuesDataType)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .setAttribute("use_node_name_sharing", useNodeNameSharing)
        .build().output
  }
}
