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

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.core.types.{DataType, INT64, TF, IsStringOrIntOrUInt}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow

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
  * @param  name              Name of this lookup table.
  *
  * @author Emmanouil Antonios Platanios
  */
class IDLookupTableWithHashBuckets[K: TF : IsStringOrIntOrUInt] private[IDLookupTableWithHashBuckets](
    val table: Option[HashTable[K, Long]],
    override val keysDataType: DataType[K],
    val numOOVBuckets: Int,
    val hashSpecification: HashSpecification = FAST_HASH,
    override val name: String = "IDLookupTableWithHashBuckets"
) extends LookupTable(keysDataType, INT64, name) {
  /** Creates an op used to initialize this table.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  override def initialize(name: String): UntypedOp = {
    table.map(_.initialize(name)).getOrElse(ControlFlow.noOp(name))
  }

  /** Creates an op that computes the number of elements in this table.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  override def size(name: String): Output[Long] = {
    Op.nameScope(name) {
      table.map(t => t.size(name) + numOOVBuckets.toLong)
          .getOrElse(Basic.constant(numOOVBuckets.toLong, name = name))
    }
  }

  /** Creates an op that looks up the provided keys in this table and returns the corresponding values.
    *
    * @param  keys Tensor containing the keys to look up.
    * @param  name Name for the created op.
    * @return Created op output.
    * @throws InvalidDataTypeException If the provided keys data types does not match the keys data type of this table.
    */
  override def lookup[OL[A] <: OutputLike[A]](
      keys: OL[K],
      name: String = "Lookup"
  )(implicit ev: OutputOps.Aux[OL, K]): OL[Long] = {
    Op.nameScope(name) {
      ev.applyUnary(keys, o => {
        if (numOOVBuckets == 0) {
          table.get.lookup(o)
        } else {
          var buckets = hashSpecification.stringToHashBucket(o.castTo[String], numOOVBuckets)
          table.map(t => {
            val ids = t.lookup(o)
            buckets = Math.add(buckets, t.size())
            Math.select(Math.notEqual(ids, t.defaultValue), ids, buckets)
          }).getOrElse(buckets)
        }
      })
    }
  }
}

object IDLookupTableWithHashBuckets {
  def apply[K: TF : IsStringOrIntOrUInt](
      table: HashTable[K, Long],
      numOOVBuckets: Int,
      hashSpecification: HashSpecification = FAST_HASH,
      name: String = "IDLookupTableWithHashBuckets"
  ): IDLookupTableWithHashBuckets[K] = {
    new IDLookupTableWithHashBuckets(
      Some(table), table.keysDataType, numOOVBuckets, hashSpecification, name)
  }

  @throws[IllegalArgumentException]
  def empty[K: TF : IsStringOrIntOrUInt](
      keysDataType: DataType[K],
      numOOVBuckets: Int,
      hashSpecification: HashSpecification = FAST_HASH,
      name: String = "IDLookupTableWithHashBuckets"
  ): IDLookupTableWithHashBuckets[K] = {
    require(
      numOOVBuckets > 0,
      "When no hash table is provided, the number of out-of-vocabulary buckets must be > 0.")
    new IDLookupTableWithHashBuckets(
      None, keysDataType, numOOVBuckets, hashSpecification, name)
  }
}

/** Hash specification for use with `IDLookupTableWithHashBuckets`. */
sealed trait HashSpecification {
  def stringToHashBucket(
      input: Output[String],
      numBuckets: Int,
      name: String = "StringToHashBucket"
  ): Output[Long]
}

@deprecated("It is recommended to use `FAST_HASH` or `STRONG_HASH` instead.", "0.1.0")
case object LEGACY_HASH extends HashSpecification {
  override def stringToHashBucket(
      input: Output[String],
      numBuckets: Int,
      name: String = "StringToHashBucket"
  ): Output[Long] = {
    Text.stringToHashBucket(input, numBuckets, name)
  }
}

case object FAST_HASH extends HashSpecification {
  override def stringToHashBucket(
      input: Output[String],
      numBuckets: Int,
      name: String = "StringToHashBucket"
  ): Output[Long] = {
    Text.stringToHashBucketFast(input, numBuckets, name)
  }
}

case class STRONG_HASH(key1: Long, key2: Long) extends HashSpecification {
  override def stringToHashBucket(
      input: Output[String],
      numBuckets: Int,
      name: String = "StringToHashBucket"
  ): Output[Long] = {
    Text.stringToHashBucketStrong(input, numBuckets, key1, key2, name)
  }
}
