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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types.{DataType, Resource, TF, IsStringOrIntOrUInt}
import org.platanios.tensorflow.api.ops.{Op, Output, UntypedOp}

/** Lookup table initializer that uses a text file.
  *
  * This initializer assigns one entry in the table for each line in the file. The key and value types of the table to
  * initialize are given by `keysDataType` and `valuesDataType`.
  *
  * The key and value content to extract from each line is specified by `keysExtractor` and `valuesExtractor`:
  *
  *   - `TextFileLineNumber`: Use the line number (starting at zero -- expects `INT64` data type).
  *   - `TextFileWholeLine`: Use the whole line content.
  *   - `TextFileColumn(i)`: Use the `i`th element of the split line based on `delimiter`.
  *
  * For example if we have a file with the following content:
  * {{{
  *   emerson 10
  *   lake 20
  *   palmer 30
  * }}}
  *
  * The following code creates an op that initializes a table with the first column as keys and second column as values:
  * {{{
  *   val table = HashTable(LookupTableTextFileInitializer(
  *     "text.txt", STRING, INT64, TextFileColumn(0), TextFileColumn(1), " "))
  * }}}
  *
  * Similarly to initialize the whole line as keys and the line number as values:
  * {{{
  *   val table = HashTable(LookupTableTextFileInitializer(
  *     "text.txt", STRING, INT64, TextFileWholeLine, TextFileLineNumber, " "))
  * }}}
  *
  * @param  filename        Scalar tensor containing the filename of the text file to be used for initialization.
  *                         The path must be accessible from wherever the graph is initialized (e.g.,
  *                         trainer or evaluation workers).
  * @param  keysDataType    Data type of the table keys.
  * @param  valuesDataType  Data type of the table values.
  * @param  keysExtractor   Text file field extractor to use for the keys (e.g., `TextFileLineNumber`).
  * @param  valuesExtractor Text file field extractor to use for the values (e.g., `TextFileWholeLine`).
  * @param  delimiter       Delimiter to use in case a `TextFileColumn` extractor is being used.
  * @param  vocabularySize  Number of elements in the file, if known. If not known, set to `-1` (the default value).
  *
  * @author Emmanouil Antonios Platanios
  */
class LookupTableTextFileInitializer[K: TF, V: TF] protected (
    val filename: Output[String],
    override val keysDataType: DataType[K],
    override val valuesDataType: DataType[V],
    val keysExtractor: TextFileFieldExtractor[K],
    val valuesExtractor: TextFileFieldExtractor[V],
    val delimiter: String = "\t",
    val vocabularySize: Int = -1
) extends LookupTableInitializer(keysDataType, valuesDataType) {
  if (vocabularySize != -1 && vocabularySize <= 0)
    throw InvalidArgumentException("The vocabulary size must be positive, if provided.")

  override def initialize(
      table: InitializableLookupTable[K, V],
      name: String = "LookupTableTextFileInitialize"
  )(implicit evVTF: TF[V]): UntypedOp = {
    Op.nameScope(name) {
      val initializationOp = Op.Builder[(Output[Resource], Output[String]), Unit](
        opType = "InitializeTableFromTextFileV2",
        name = name,
        input = (table.handle, filename)
      ).setAttribute("key_index", keysExtractor.value)
          .setAttribute("value_index", valuesExtractor.value)
          .setAttribute("vocab_size", vocabularySize)
          .setAttribute("delimiter", delimiter)
          .build()
      Op.currentGraph.addToCollection(Graph.Keys.TABLE_INITIALIZERS)(initializationOp)
      // If the filename asset tensor is anything other than a string constant
      // (e.g., if it is a placeholder), then it does not make sense to track
      // it as an asset.
      if (filename.op.opType == "Const")
        Op.currentGraph.addToCollection(Graph.Keys.ASSET_FILEPATHS)(filename)
      initializationOp
    }
  }
}

object LookupTableTextFileInitializer {
  def apply[K: TF, V: TF](
      filename: Output[String],
      keysDataType: DataType[K],
      valuesDataType: DataType[V],
      keysExtractor: TextFileFieldExtractor[K],
      valuesExtractor: TextFileFieldExtractor[V],
      delimiter: String = "\t",
      vocabularySize: Int = -1
  ): LookupTableTextFileInitializer[K, V] = {
    new LookupTableTextFileInitializer(
      filename, keysDataType, valuesDataType,
      keysExtractor, valuesExtractor,
      delimiter, vocabularySize)
  }
}

/** Represents a field extractor from a text file. */
sealed trait TextFileFieldExtractor[+K] {
  val name : String
  val value: Int

  override def toString: String = name
}

/** Text file field extractor that extracts the line number as the field (starting at zero). */
case object TextFileLineNumber extends TextFileFieldExtractor[Long] {
  override val name : String = "LINE_NUMBER"
  override val value: Int    = -1
}

/** Text file field extractor that extracts the whole line as a field. */
case class TextFileWholeLine[+K: TF]()(implicit
    ev: IsStringOrIntOrUInt[K]
) extends TextFileFieldExtractor[K] {
  override val name : String = "WHOLE_LINE"
  override val value: Int    = -2
}

/** Text file field extractor that extracts a column from a line as a field.
  *
  * @param  index Column index.
  */
case class TextFileColumn[+K: TF](index: Int) extends TextFileFieldExtractor[K] {
  override val name : String = s"COLUMN[$index]"
  override val value: Int    = index
}
