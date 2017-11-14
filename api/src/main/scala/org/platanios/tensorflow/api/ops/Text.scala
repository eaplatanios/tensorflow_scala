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

package org.platanios.tensorflow.api.ops

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Text {
  /** $OpDocTextStringJoin
    *
    * @group TextOps
    * @param  inputs    Sequence of string tensors that will be joined. The tensors must all have the same shape, or be
    *                   scalars. Scalars may be mixed in; these will be broadcast to the shape of the non-scalar inputs.
    * @param  separator Separator string.
    */
  def stringJoin(inputs: Seq[Output], separator: String = "", name: String = "StringJoin"): Output = {
    Op.Builder(opType = "StringJoin", name = name)
        .addInputList(inputs)
        .setAttribute("separator", separator)
        .build().outputs(0)
  }

  /** $OpDocTextStringToHashBucket
    *
    * @group TextOps
    * @param  input      `STRING` tensor containing the strings to assign to each bucket.
    * @param  numBuckets Number of buckets.
    * @param  name       Name for the created op.
    * @return Created op output, which has the same shape as `input`.
    */
  @deprecated("It is recommended to use `stringToHashBucketFast` or `stringToHashBucketStrong`.", "0.1.0")
  def stringToHashBucket(input: Output, numBuckets: Int, name: String = "StringToHashBucket"): Output = {
    Op.Builder(opType = "StringToHashBucket", name = name)
        .addInput(input)
        .setAttribute("num_buckets", numBuckets)
        .build().outputs(0)
  }

  /** $OpDocTextStringToHashBucketFast
    *
    * @group TextOps
    * @param  input      `STRING` tensor containing the strings to assign to each bucket.
    * @param  numBuckets Number of buckets.
    * @param  name       Name for the created op.
    * @return Created op output, which has the same shape as `input`.
    */
  def stringToHashBucketFast(input: Output, numBuckets: Int, name: String = "StringToHashBucketFast"): Output = {
    Op.Builder(opType = "StringToHashBucketFast", name = name)
        .addInput(input)
        .setAttribute("num_buckets", numBuckets)
        .build().outputs(0)
  }

  /** $OpDocTextStringToHashBucketStrong
    *
    * @group TextOps
    * @param  input      `STRING` tensor containing the strings to assign to each bucket.
    * @param  numBuckets Number of buckets.
    * @param  key1       First part of the key for the keyed hash function.
    * @param  key2       Second part of the key for the keyed hash function.
    * @param  name       Name for the created op.
    * @return Created op output, which has the same shape as `input`.
    */
  def stringToHashBucketStrong(
      input: Output, numBuckets: Int, key1: Long, key2: Long, name: String = "StringToHashBucketStrong"): Output = {
    Op.Builder(opType = "StringToHashBucketStrong", name = name)
        .addInput(input)
        .setAttribute("num_buckets", numBuckets)
        .setAttribute("key", Seq(key1, key2))
        .build().outputs(0)
  }
}

private[api] object Text extends Text {
  private[ops] trait Implicits {
    implicit def outputToTextOps(value: Output): TextOps = TextOps(value)
    implicit def outputConvertibleToTextOps[T](value: T)(implicit f: (T) => Output): TextOps = TextOps(f(value))
  }

  case class TextOps private[ops](output: Output) {

    /** $OpDocTextStringToHashBucket
      *
      * @group TextOps
      * @param  numBuckets Number of buckets.
      * @param  name       Name for the created op.
      * @return Created op output, which has the same shape as `input`.
      */
    @deprecated("It is recommended to use `stringToHashBucketFast` or `stringToHashBucketStrong`.", "0.1.0")
    def stringToHashBucket(numBuckets: Int, name: String = "StringToHashBucket"): Output = {
      Text.stringToHashBucket(output, numBuckets, name)
    }

    /** $OpDocTextStringToHashBucketFast
      *
      * @group TextOps
      * @param  numBuckets Number of buckets.
      * @param  name       Name for the created op.
      * @return Created op output, which has the same shape as `input`.
      */
    def stringToHashBucketFast(numBuckets: Int, name: String = "StringToHashBucketFast"): Output = {
      Text.stringToHashBucketFast(output, numBuckets, name)
    }

    /** $OpDocTextStringToHashBucketStrong
      *
      * @group TextOps
      * @param  numBuckets Number of buckets.
      * @param  key1       First part of the key for the keyed hash function.
      * @param  key2       Second part of the key for the keyed hash function.
      * @param  name       Name for the created op.
      * @return Created op output, which has the same shape as `input`.
      */
    def stringToHashBucketStrong(
        numBuckets: Int, key1: Long, key2: Long, name: String = "StringToHashBucketStrong"): Output = {
      Text.stringToHashBucketStrong(output, numBuckets, key1, key2, name)
    }
  }

  /** @define OpDocTextStringJoin
    *   The `stringJoin` op joins the strings in the given list of string tensors into one tensor, using the provided
    *   separator (which defaults to an empty string).
    *
    * @define OpDocTextStringToHashBucket
    *   The `stringToHashBucket` op converts each string in the input tensor to its hash mod the number of buckets.
    *
    *   The hash function is deterministic on the content of the string within the process. Note that the hash function
    *   may change from time to time.
    *
    * @define OpDocTextStringToHashBucketFast
    *   The `stringToHashBucketFast` op converts each string in the input tensor to its hash mod the number of buckets.
    *
    *   The hash function is deterministic on the content of the string within the process and will never change.
    *   However, it is not suitable for cryptography. This method may be used when CPU time is scarce and inputs are
    *   trusted or are unimportant. There is a risk of adversaries constructing inputs that all hash to the same bucket.
    *   To prevent this problem, use `stringToHashBucketStrong`.
    *
    * @define OpDocTextStringToHashBucketStrong
    *   The `stringToHashBucketStrong` op converts each string in the input tensor to its hash mod the number of
    *   buckets.
    *
    *   The hash function is deterministic on the content of the string within the process. The hash function is a keyed
    *   hash function, where `key1` and `key2` define the key of the hash function. A strong hash is important when
    *   inputs may be malicious (e.g., URLs with additional components). Adversaries could try to make their inputs hash
    *   to the same bucket for a denial-of-service attack or to skew the results. A strong hash prevents this by making
    *   it difficult, if not infeasible, to compute inputs that hash to the same bucket. This comes at a cost of roughly
    *   4x higher compute time than `stringToHashBucketFast`.
    */
  private[ops] trait Documentation
}
