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

package org.platanios.tensorflow.api.implicits.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Output, SparseOutput, Text}
import org.platanios.tensorflow.api.tensors.Tensor

trait TextImplicits {
  implicit def outputConvertibleToTextOps[OC](
      value: OC
  )(implicit f: OC => Output[String]): TextOps = {
    new TextOps(f(value))
  }

  implicit class TextOps(val output: Output[String]) {
    /** $OpDocTextRegexReplace
      *
      * @group TextOps
      * @param  pattern       Tensor containing the regular expression to match the input.
      * @param  rewrite       Tensor containing the rewrite to be applied to the matched expression.
      * @param  replaceGlobal If `true`, the replacement is global, otherwise the replacement is done only on the first
      *                       match.
      * @param  name          Name for the created op.
      * @return Created op output.
      */
    def regexReplace(
        pattern: Output[String],
        rewrite: Output[String],
        replaceGlobal: Boolean = true,
        name: String = "RegexReplace"
    ): Output[String] = {
      Text.regexReplace(output, pattern, rewrite, replaceGlobal, name)
    }

    /** $OpDocTextStringSplit
      *
      * @group TextOps
      * @param  delimiter Delimiter used for splitting. If `delimiter` is an empty string, each element of the `source`
      *                   is split into individual strings, each containing one byte. (This includes splitting
      *                   multi-byte sequences of UTF-8 characters). If `delimiter` contains multiple bytes, it is
      *                   treated as a set of delimiters with each considered a potential split point.
      * @param  skipEmpty Boolean value indicating whether or not to skip empty tokens.
      * @param  name      Name for the created op.
      * @return Created op output.
      */
    def stringSplit(
        delimiter: Output[String] = Tensor.fill[String](Shape())(" ").toOutput,
        skipEmpty: Boolean = true,
        name: String = "StringSplit"
    ): SparseOutput[String] = {
      Text.stringSplit(output, delimiter, skipEmpty, name)
    }

    /** $OpDocTextStringEncodeBase64
      *
      * @group TextOps
      * @param  pad  Boolean value indicating whether or not padding is applied at the string ends.
      * @param  name Name for the created op.
      * @return Created op output.
      */
    def encodeBase64(
        pad: Boolean = false,
        name: String = "EncodeBase64"
    ): Output[String] = {
      Text.encodeBase64(output, pad, name)
    }

    /** $OpDocTextStringDecodeBase64
      *
      * @group TextOps
      * @param  name Name for the created op.
      * @return Created op output.
      */
    def decodeBase64(
        name: String = "DecodeBase64"
    ): Output[String] = {
      Text.decodeBase64(output, name)
    }

    /** $OpDocTextStringToHashBucket
      *
      * @group TextOps
      * @param  numBuckets Number of buckets.
      * @param  name       Name for the created op.
      * @return Created op output, which has the same shape as `input`.
      */
    @deprecated("It is recommended to use `stringToHashBucketFast` or `stringToHashBucketStrong`.", "0.1.0")
    def stringToHashBucket(
        numBuckets: Int,
        name: String = "StringToHashBucket"
    ): Output[Long] = {
      Text.stringToHashBucket(output, numBuckets, name)
    }

    /** $OpDocTextStringToHashBucketFast
      *
      * @group TextOps
      * @param  numBuckets Number of buckets.
      * @param  name       Name for the created op.
      * @return Created op output, which has the same shape as `input`.
      */
    def stringToHashBucketFast(
        numBuckets: Int,
        name: String = "StringToHashBucketFast"
    ): Output[Long] = {
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
        numBuckets: Int,
        key1: Long,
        key2: Long,
        name: String = "StringToHashBucketStrong"
    ): Output[Long] = {
      Text.stringToHashBucketStrong(output, numBuckets, key1, key2, name)
    }
  }
}
