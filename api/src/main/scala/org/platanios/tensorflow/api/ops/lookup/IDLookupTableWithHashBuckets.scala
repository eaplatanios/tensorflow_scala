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
import org.platanios.tensorflow.api.core.exception.{InvalidArgumentException, InvalidDataTypeException}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.types.{DataType, INT64, STRING}

/** String to ID lookup table wrapper that assigns out-of-vocabulary keys to buckets.
  *
  * For example, if an instance of `IDLookupTableWithHashBuckets` is initialized with a string-to-ID table that maps:
  * {{{
  *   emerson -> 0
  *   lake -> 1
  *   palmer -> 2
  * }}}
  * The `IDLookupTableWithHashBuckets` object will perform the following mapping:
  * {{{
  *   emerson -> 0
  *   lake -> 1
  *   palmer -> 2
  *   <other term> -> bucket ID between 3 and 3 + numOOVBuckets - 1, calculated by hash(<term>) % numOOVBuckets + vocabularySize
  * }}}
  *
  * If the input tensor is `["emerson", "lake", "palmer", "king", "crimson"]`, the lookup result is `[0, 1, 2, 4, 7]`.
  * If `table` is `null`, only out-of-vocabulary buckets are used.
  *
  * Example usage:
  * {{{
  *   val numOOVBuckets = 3
  *   val input = Tensor("emerson", "lake", "palmer", "king", "crimson")
  *   val table = IDLookupTableWithHashBuckets(
  *     HashTable(LookupTableTextFileInitializer(filename), defaultValue), numOOVBuckets)
  *   val output = table.lookup(input)
  * }}}
  *
  * The hash function used for generating out-of-vocabulary buckets ID is defined by `hashSpecification`.
  *
  * @param  table             Lookup table to wrap.
  * @param  numOOVBuckets     Number of out-of-vocabulary buckets.
  * @param  hashSpecification Hashing function specification to use.
  * @param  keysDataType      Data type of the table keys.
  * @param  name              Name of this lookup table.
  *
  * @author Emmanouil Antonios Platanios
  */
class IDLookupTableWithHashBuckets(
    val table: HashTable,
    val numOOVBuckets: Int,
    val hashSpecification: HashSpecification = FAST_HASH,
    override val keysDataType: DataType = null,
    override val name: String = "IDLookupTableWithHashBuckets"
) extends LookupTable(keysDataType, INT64, name) {
  private[this] var inferredKeysDataType: DataType = keysDataType
  if (table != null) {
    if (inferredKeysDataType == null)
      inferredKeysDataType = table.keysDataType
    if (table.keysDataType != STRING && table.keysDataType != INT64)
      throw InvalidDataTypeException(
        s"Expected table key data type to be either STRING or INT64, but got ${table.keysDataType}.")
    if (table.keysDataType.isInteger && !inferredKeysDataType.isInteger)
      throw InvalidDataTypeException("Expected non-integer table key data type but got integer.")
    else if (!table.keysDataType.isInteger && inferredKeysDataType.isInteger)
      throw InvalidDataTypeException("Expected integer table key data type but got non-integer.")
    if (table.valuesDataType != INT64)
      throw InvalidDataTypeException(s"Expected INT64 table value data type but got ${table.valuesDataType}.")
  } else {
    if (numOOVBuckets <= 0)
      throw InvalidArgumentException(s"'numOOVBuckets' must be > 0 if no table is provided, but was $numOOVBuckets.")
    if (inferredKeysDataType == null)
      inferredKeysDataType = STRING
  }
  if (!inferredKeysDataType.isInteger && inferredKeysDataType != STRING)
    throw InvalidDataTypeException(
      s"Expected key data type to be either STRING or INT64, but got ${table.keysDataType}.")

  /** Creates an op used to initialize this table.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  override def initialize(name: String): Op = {
    if (table == null)
      ControlFlow.noOp(name)
    else
      table.initialize(name)
  }

  /** Creates an op that computes the number of elements in this table.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  override def size(name: String): Output = {
    if (table == null)
      Basic.constant(numOOVBuckets, INT64, Shape.scalar())
    else
      Op.createWithNameScope(name)(table.size(name) + numOOVBuckets)
  }

  /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
    *
    * @param  keys Tensor containing the keys to look up.
    * @param  name Name for the created op.
    * @return Created op output.
    * @throws InvalidDataTypeException If the provided keys data types does not match the keys data type of this table.
    */
  @throws[InvalidDataTypeException]
  override def lookup[T <: OutputLike : OutputOps](keys: T, name: String): T = Op.createWithNameScope(name) {
    if (keys.dataType != keysDataType)
      throw InvalidDataTypeException(s"Invalid keys data type ${keys.dataType} (expected $keysDataType).")
    implicitly[OutputOps[T]].applyUnary(keys, o => {
      val castedKeys = if (table != null && table.keysDataType == INT64) o.cast(INT64) else o
      if (numOOVBuckets == 0) {
        table.lookup(castedKeys)
      } else {
        var buckets = hashSpecification.stringToHashBucket(castedKeys.cast(STRING), numOOVBuckets)
        if (table == null) {
          buckets
        } else {
          val ids = table.lookup(castedKeys)
          buckets = Math.add(buckets, table.size())
          Math.select(Math.notEqual(ids, table.defaultValue), ids, buckets)
        }
      }
    })
  }
}

object IDLookupTableWithHashBuckets {
  def apply(
      table: HashTable, numOOVBuckets: Int, hashSpecification: HashSpecification = FAST_HASH,
      keysDataType: DataType = null, name: String = "IDLookupTableWithHashBuckets"): IDLookupTableWithHashBuckets = {
    new IDLookupTableWithHashBuckets(table, numOOVBuckets, hashSpecification, keysDataType, name)
  }
}

/** Hash specification for use with `IDLookupTableWithHashBuckets`. */
sealed trait HashSpecification {
  def stringToHashBucket(input: Output, numBuckets: Int, name: String = "StringToHashBucket"): Output
}

@deprecated("It is recommended to use `FAST_HASH` or `STRONG_HASH` instead.", "0.1.0")
case object LEGACY_HASH extends HashSpecification {
  override def stringToHashBucket(input: Output, numBuckets: Int, name: String): Output = {
    Text.stringToHashBucket(input, numBuckets, name)
  }
}

case object FAST_HASH extends HashSpecification {
  override def stringToHashBucket(input: Output, numBuckets: Int, name: String): Output = {
    Text.stringToHashBucketFast(input, numBuckets, name)
  }
}

case class STRONG_HASH(key1: Long, key2: Long) extends HashSpecification {
  override def stringToHashBucket(input: Output, numBuckets: Int, name: String): Output = {
    Text.stringToHashBucketStrong(input, numBuckets, key1, key2, name)
  }
}
