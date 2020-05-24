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
import org.platanios.tensorflow.api.ops.{Output, Statistics}

trait StatisticsImplicits {
  implicit def outputConvertibleToStatisticsOps[T, OC](
      value: OC
  )(implicit f: OC => Output[T]): StatisticsOps[T] = {
    new StatisticsOps(f(value))
  }

  implicit class StatisticsOps[T](val output: Output[T]) {
    protected implicit val evTTF: TF[T] = {
      TF.fromDataType(output.dataType)
    }

    /** $OpDocStatisticsSufficientStatistics
      *
      * @group StatisticsOps
      * @param  axes     Tensor containing the axes along which to compute the mean and variance.
      * @param  shift    Optional tensor containing the value by which to shift the data for numerical stability.
      *                  Defaults to `null`, meaning that no shift needs to be performed. A shift close to the true
      *                  mean provides the most numerically stable results.
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Tuple containing the following created op outputs:
      *         - Count: The number of elements to average over.
      *         - Mean Sufficient Statistic: The (possibly shifted) sum of the elements in the tensor.
      *         - Variance Sufficient Statistic: The (possibly shifted) sum of squares of the elements in the tensor.
      *         - Shift: The shift by which the mean must be corrected, or `null` if no shift was used.
      */
    def sufficientStatistics[I: TF : IsIntOrLong](
        axes: Output[I],
        shift: Output[T] = null,
        keepDims: Boolean = false
    )(implicit ev: IsNotQuantized[T]): (Output[T], Output[T], Output[T], Output[T]) = {
      Statistics.sufficientStatistics(output, axes, shift, keepDims)
    }

    /** $OpDocStatisticsMoments
      *
      * @group StatisticsOps
      * @param  axes     Axes along which to compute the mean and variance.
      * @param  weights  Optional tensor of positive weights that can be broadcast with `input`, to weigh the samples.
      *                  Defaults to `null`, meaning that equal weighting is used (i.e., all samples have weight equal
      *                  to `1`).
      * @param  keepDims If `true`, retain the reduced axes.
      * @return Tuple containing the created op outputs: (i) the mean tensor, and (ii) the variance tensor.
      */
    def moments(
        axes: Seq[Int],
        weights: Output[T] = null,
        keepDims: Boolean = false
    )(implicit ev: IsNotQuantized[T]): (Output[T], Output[T]) = {
      Statistics.moments(output, axes, weights, keepDims)
    }
  }
}
