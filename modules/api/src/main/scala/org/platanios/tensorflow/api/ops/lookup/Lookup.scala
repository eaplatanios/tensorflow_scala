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

import org.platanios.tensorflow.api.core.types.{DataType, INT64, TF, IsStringOrIntOrUInt}
import org.platanios.tensorflow.api.ops.Op

/** Contains functions for constructing ops related to lookup tables.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Lookup {
  /** Creates a lookup table that converts string tensors into integer IDs.
    *
    * This operation constructs a lookup table to convert tensors of strings into tensors of `INT64` IDs. The mapping
    * is initialized from a vocabulary file specified in `filename`, where the whole line is the key and the zero-based
    * line number is the ID.
    *
    * Any lookup of an out-of-vocabulary token will return a bucket ID based on its hash if `numOOVBuckets` is greater
    * than zero. Otherwise it is assigned the `defaultValue`. The bucket ID range is:
    * `[vocabularySize, vocabularySize + numOOVBuckets - 1]`.
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
    * @param  filename          Filename of the text file to be used for initialization. The path must be accessible
    *                           from wherever the graph is initialized (e.g., trainer or evaluation workers).
    * @param  delimiter         Delimiter to use in case a `TextFileColumn` extractor is being used.
    * @param  vocabularySize    Number of elements in the file, if known. If not known, set to `-1` (the default value).
    * @param  defaultValue      Default value to use if a key is missing from the table.
    * @param  numOOVBuckets     Number of out-of-vocabulary buckets.
    * @param  hashSpecification Hashing function specification to use.
    * @param  keysDataType      Data type of the table keys.
    * @param  name              Name for the created table.
    * @return Created table.
    */
  def indexTableFromFile[K: TF : IsStringOrIntOrUInt](
      filename: String,
      keysDataType: DataType[K],
      delimiter: String = "\t",
      vocabularySize: Int = -1,
      defaultValue: Long = -1L,
      numOOVBuckets: Int = 0,
      hashSpecification: HashSpecification = FAST_HASH,
      name: String = "IndexTableFromFile"
  ): LookupTable[K, Long] = {
    Op.nameScope(s"$name/HashTable") {
      val sharedName = {
        if (vocabularySize != -1)
          s"hash_table_${filename}_${vocabularySize}_${TextFileWholeLine}_$TextFileLineNumber"
        else
          s"hash_table_${filename}_${TextFileWholeLine}_$TextFileLineNumber"
      }
      val initializer = LookupTableTextFileInitializer(
        filename = filename,
        keysDataType = keysDataType,
        valuesDataType = INT64,
        keysExtractor = TextFileWholeLine[K],
        valuesExtractor = TextFileLineNumber,
        delimiter = delimiter,
        vocabularySize = vocabularySize)
      val table = HashTable(initializer, defaultValue, sharedName = sharedName, name = "Table")
      if (numOOVBuckets > 0)
        IDLookupTableWithHashBuckets(table, numOOVBuckets, hashSpecification)
      else
        table
    }
  }
}

object Lookup extends Lookup
