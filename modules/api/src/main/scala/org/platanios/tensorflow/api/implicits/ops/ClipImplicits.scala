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

import org.platanios.tensorflow.api.core.types.{IsIntOrLong, IsNotQuantized, TF}
import org.platanios.tensorflow.api.ops.{Clip, Output}
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault

trait ClipImplicits {
  implicit def outputConvertibleToClipOps[T, OC](
      value: OC
  )(implicit f: OC => Output[T]): ClipOps[T] = {
    new ClipOps(f(value))
  }

  implicit class ClipOps[T](val output: Output[T]) {
    protected implicit val evTTF: TF[T] = {
      TF.fromDataType(output.dataType)
    }

    /** $OpDocClipClipByValue
      *
      * @group ClipOps
      * @param  clipValueMin 0-D (scalar) tensor, or a tensor with the same shape as this tensor, specifying the minimum
      *                      value to clip by.
      * @param  clipValueMax 0-D (scalar) tensor, or a tensor with the same shape as this tensor, specifying the maximum
      *                      value to clip by.
      * @param  name         Name prefix for created ops.
      * @return Created op output.
      */
    def clipByValue(
        clipValueMin: Output[T],
        clipValueMax: Output[T],
        name: String = "ClipByValue"
    )(implicit ev: IsNotQuantized[T]): Output[T] = {
      Clip.clipByValue(output, clipValueMin, clipValueMax, name)
    }

    /** $OpDocClipClipByNorm
      *
      * @group ClipOps
      * @param  clipNorm 0-D (scalar) tensor > 0, specifying the maximum clipping value.
      * @param  axes     1-D (vector) tensor containing the dimensions to use for computing the l2-norm. If
      *                  `null` (the default), all dimensions are used.
      * @param  name     Name prefix for created ops.
      * @return Created op output.
      */
    def clipByNorm[I: IntDefault : TF : IsIntOrLong](
        clipNorm: Output[T],
        axes: Output[I] = null,
        name: String = "ClipByNorm"
    )(implicit ev: IsNotQuantized[T]): Output[T] = {
      Clip.clipByNorm(output, clipNorm, axes, name)
    }

    /** $OpDocClipClipByAverageNorm
      *
      * @group ClipOps
      * @param  clipNorm 0-D (scalar) tensor > 0, specifying the maximum clipping value.
      * @param  name     Name prefix for created ops.
      * @return Created op output.
      */
    def clipByAverageNorm(
        input: Output[T],
        clipNorm: Output[T],
        name: String = "ClipByAverageNorm"
    )(implicit ev: IsNotQuantized[T]): Output[T] = {
      Clip.clipByAverageNorm(output, clipNorm, name)
    }
  }
}
