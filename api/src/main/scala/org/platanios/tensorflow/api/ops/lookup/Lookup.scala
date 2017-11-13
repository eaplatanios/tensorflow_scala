/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.types.{DataType, INT64, STRING}

/**
  * @author Emmanouil Antonios Platanios
  */
private[lookup] trait Lookup {
  /** Creates an op that initializes all tables with initializers in the provided graph collection.
    *
    * After you launch the graph in a session, you can run the returned op to initialize tables. This op runs all the
    * initializers of the tables in the specified collection, in parallel.
    *
    * Calling `tablesInitializer` is equivalent to passing the list of initializers to [[ControlFlow.group]].
    *
    * If no table initializers are found in the provided collection, the method still returns an op that can be run.
    * That op has no effect (i.e., it is a [[ControlFlow.noOp]]).
    *
    * @param  key  Collection key from which to collect the table initializers.
    * @param  name Name for the created op.
    * @return Created op.
    */
  def tablesInitializer(key: Graph.Key[Op] = Graph.Keys.TABLE_INITIALIZERS, name: String = "TablesInitializer"): Op = {
    val initializers = Op.currentGraph.getCollection(key)
    if (initializers.nonEmpty)
      ControlFlow.group(initializers, name)
    else
      ControlFlow.noOp(name)
  }

  /** Creates a lookup table that converts string tensors into integer IDs.
    *
    * This operation constructs a lookup table to convert tensors of strings into tensors of `INT64` IDs. The mapping
    * is initialized from a vocabulary file specified in `filename`, where the whole line is the key and the zero-based
    * line number is the ID.
    *
    * The underlying table must be initialized by executing the `tf.tablesInitializer()` op or the op returned by
    * `table.initialize()`.
    *
    * Example usage:
    *
    * If we have a vocabulary file `"test.txt"` with the following content:
    * {{{
    *   emerson
    *   lake
    *   palmer
    * }}}
    * Then, we can use the following code to create a table mapping `"emerson" -> 0`, `"lake" -> 1`, and
    * `"palmer" -> 2`:
    * {{{
    *   val table = tf.indexTableFromFile("test.txt"))
    * }}}
    *
    * @param  filename       Filename of the text file to be used for initialization. The path must be accessible from
    *                        wherever the graph is initialized (e.g., trainer or evaluation workers).
    * @param  delimiter      Delimiter to use in case a `TextFileColumn` extractor is being used.
    * @param  vocabularySize Number of elements in the file, if known. If not known, set to `-1` (the default value).
    * @param  defaultValue   Default value to use if a key is missing from the table.
    * @param  keysDataType   Data type of the table keys.
    * @param  name           Name for the created table.
    * @return Created table.
    */
  def indexTableFromFile(
      filename: String, delimiter: String = "\t", vocabularySize: Int = -1, defaultValue: Int = -1,
      keysDataType: DataType = STRING, name: String = "IndexTableFromFile"): LookupTable = {
    Op.createWithNameScope(name) {
      Op.createWithNameScope("HashTable") {
        val sharedName = {
          if (vocabularySize != -1)
            s"hash_table_${filename}_${vocabularySize}_${TextFileWholeLine}_$TextFileLineNumber"
          else
            s"hash_table_${filename}_${TextFileWholeLine}_$TextFileLineNumber"
        }
        val initializer = LookupTableTextFileInitializer(
          filename, if (keysDataType.isInteger) INT64 else keysDataType, INT64,
          TextFileWholeLine, TextFileLineNumber, delimiter, vocabularySize)
        val table = HashTable(initializer, defaultValue, sharedName, name = "Table")
        // TODO: [LOOKUP] Add support for out-of-vocabulary buckets.
        table
      }
    }
  }
}

object Lookup extends Lookup {
  /** Creates an op that computes the number of elements in the table referenced by `handle`.
    *
    * @param  handle Resource handle to a lookup table.
    * @param  name   Name for the created op.
    * @return Created op output.
    */
  private[lookup] def lookupTableSize(handle: Output, name: String = "LookupTableSize"): Output = {
    Op.Builder("LookupTableSizeV2", name)
        .addInput(handle)
        .build().outputs(0)
  }

  /** Creates an op that looks up `keys` in the table referenced by `handle` and outputs the corresponding values.
    *
    * `keys` must of the same data type as the keys of the table. The output `values` is of the data type of the table
    * values.
    *
    * The scalar `defaultValue` is the value that will be returned for keys not present in the table. It must also be of
    * the same data type as the table values.
    *
    * @param  handle       Resource handle to a lookup table.
    * @param  keys         Tensor containing the keys to look up.
    * @param  defaultValue Default value to return for keys that cannot be found in the table.
    * @param  name         Name for the created op.
    * @return Created op output.
    */
  private[lookup] def lookupTableFind(
      handle: Output, keys: Output, defaultValue: Output, name: String = "LookupTableFind"): Output = {
    Op.Builder("LookupTableFindV2", name)
        .addInput(handle)
        .addInput(keys)
        .addInput(defaultValue)
        .build().outputs(0)
  }

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
  private[lookup] def createHashTable(
      keysDataType: DataType, valuesDataType: DataType, container: String = "", sharedName: String = "",
      useNodeNameSharing: Boolean = false, name: String = "HashTable"): Output = {
    Op.Builder("HashTableV2", name)
        .setAttribute("key_dtype", keysDataType)
        .setAttribute("value_dtype", valuesDataType)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .setAttribute("use_node_name_sharing", useNodeNameSharing)
        .build().outputs(0)
  }

  /** Creates an op that initializes the table referenced by `handle` to the provided keys and values.
    *
    * @param  handle Resource handle to a lookup table which will be initialized.
    * @param  keys   Tensor containing the lookup keys.
    * @param  values Tensor containing the lookup values.
    * @param  name   Name for the created op.
    * @return Created op.
    */
  private[lookup] def createLookupTableTensorInitializer(
      handle: Output, keys: Output, values: Output, name: String = "InitializeLookupTable"): Op = {
    Op.Builder("InitializeTableV2", name)
        .addInput(handle)
        .addInput(keys)
        .addInput(values)
        .build()
  }

  /** Creates an op that initializes the table referenced by `handle` from a text file.
    *
    * The op inserts one key-value pair into the table for each line in the file. The key and value are extracted from
    * the whole line content, elements from the split line based on `delimiter`, or the line number (starting at zero).
    * `keyIndex` and `valueIndex` specify where to extract the key and value from within a line:
    *
    *   - A value of `-1` means to use the line number (starting at zero -- expects `INT64` data type).
    *   - A value of `-2` means to use the whole line content (expects `STRING` data type).
    *   - A value of `>= 0` means to use the index (starting at zero) of the split line based on `delimiter`.
    *
    * @param  handle         Resource handle to a lookup table which will be initialized.
    * @param  filename       Tensor containing the filename.
    * @param  keyIndex       Key index value (described above).
    * @param  valueIndex     Value index value (described above).
    * @param  vocabularySize Number of elements in the file.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  private[lookup] def createLookupTableTextFileInitializer(
      handle: Output, filename: Output, keyIndex: Int = -2, valueIndex: Int = -2, vocabularySize: Int = -1,
      delimiter: String = "\t", name: String = "InitializeLookupTableFromTextFile"): Op = {
    Op.Builder("InitializeTableFromTextFileV2", name)
        .addInput(handle)
        .addInput(filename)
        .setAttribute("key_index", keyIndex)
        .setAttribute("value_index", valueIndex)
        .setAttribute("vocab_size", vocabularySize)
        .setAttribute("delimiter", delimiter)
        .build()
  }

  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("LookupTableFind")
    GradientsRegistry.registerNonDifferentiable("LookupTableFindV2")
    GradientsRegistry.registerNonDifferentiable("LookupTableInsert")
    GradientsRegistry.registerNonDifferentiable("LookupTableInsertV2")
    GradientsRegistry.registerNonDifferentiable("LookupTableSize")
    GradientsRegistry.registerNonDifferentiable("LookupTableSizeV2")
    GradientsRegistry.registerNonDifferentiable("HashTable")
    GradientsRegistry.registerNonDifferentiable("HashTableV2")
    GradientsRegistry.registerNonDifferentiable("InitializeTable")
    GradientsRegistry.registerNonDifferentiable("InitializeTableV2")
    GradientsRegistry.registerNonDifferentiable("InitializeTableFromTextFile")
    GradientsRegistry.registerNonDifferentiable("InitializeTableFromTextFileV2")
    GradientsRegistry.registerNonDifferentiable("MutableDenseHashTable")
    GradientsRegistry.registerNonDifferentiable("MutableDenseHashTableV2")
    GradientsRegistry.registerNonDifferentiable("MutableHashTable")
    GradientsRegistry.registerNonDifferentiable("MutableHashTableV2")
    GradientsRegistry.registerNonDifferentiable("MutableHashTableOfTensors")
    GradientsRegistry.registerNonDifferentiable("MutableHashTableOfTensorsV2")
  }
}
