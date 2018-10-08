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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.core.types.{TF, IsInt32OrInt64, IsNotQuantized}

/** Contains functions for constructing ops related to statistics.
  *
  * @author Emmanouil Antonios Platanios, Sören Brunk
  */
trait Statistics {
  /** $OpDocStatisticsSufficientStatistics
    *
    * @group StatisticsOps
    * @param  input    Input tensor.
    * @param  axes     Tensor containing the axes along which to compute the mean and variance.
    * @param  shift    Optional tensor containing the value by which to shift the data for numerical stability.
    *                  Defaults to `null`, meaning that no shift needs to be performed. A shift close to the true mean
    *                  provides the most numerically stable results.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Tuple containing the following created op outputs:
    *         - Count: The number of elements to average over.
    *         - Mean Sufficient Statistic: The (possibly shifted) sum of the elements in the tensor.
    *         - Variance Sufficient Statistic: The (possibly shifted) sum of squares of the elements in the tensor.
    *         - Shift: The shift by which the mean must be corrected, or `null` if no shift was used.
    */
  def sufficientStatistics[T: TF : IsNotQuantized, I: TF : IsInt32OrInt64](
      input: Output[T],
      axes: Output[I],
      shift: Output[T] = null,
      keepDims: Boolean = false,
      name: String = "SufficientStatistics"
  ): (Output[T], Output[T], Output[T], Output[T]) = {
    Op.nameScope(name) {
      val dynamicAxes = axes
      val inputShape = Basic.shape(input).castTo[T](TF.fromDataType(input.dataType))
      val counts = Math.prod(Basic.gather(inputShape, dynamicAxes, axis = 0))
      val mSS = if (shift == null) input else input - shift
      val vSS = if (shift == null) Math.square(input) else Math.squaredDifference(input, shift)
      val meanSS = Math.sum(mSS, axes = dynamicAxes, keepDims = keepDims, name = "MeanSS")
      val varSS = Math.sum(vSS, axes = dynamicAxes, keepDims = keepDims, name = "VarSS")
      (counts, meanSS, varSS, shift)
    }
  }

  /** $OpDocStatisticsMomentsFromSufficientStatistics
    *
    * @group StatisticsOps
    * @param  counts Total number of elements over which the provided sufficient statistics were computed.
    * @param  meanSS Mean sufficient statistics: the (possibly shifted) sum of the elements.
    * @param  varSS  Variance sufficient statistics: the (possibly shifted) sum of squares of the elements.
    * @param  shift  The shift by which the mean must be corrected, or `null` if no shift was used.
    * @param  name   Name for the created op.
    * @return Tuple containing the created op outputs: (i) the mean tensor, and (ii) the variance tensor.
    */
  def momentsFromSufficientStatistics[T: TF : IsNotQuantized](
      counts: Output[T],
      meanSS: Output[T],
      varSS: Output[T],
      shift: Output[T] = null,
      name: String = "MomentsFromSufficientStatistics"
  ): (Output[T], Output[T]) = {
    Op.nameScope(name) {
      val divisor = Math.reciprocal(counts, name = "Divisor")
      val (mean, shiftedMean) = {
        if (shift == null) {
          val mean = Math.multiply(meanSS, divisor, name = "Mean")
          (mean, mean)
        } else {
          val shiftedMean = Math.multiply(meanSS, divisor, name = "ShiftedMean")
          val mean = Math.add(shiftedMean, shift, name = "Mean")
          (mean, shiftedMean)
        }
      }
      val variance = Math.subtract(
        Math.multiply(varSS, divisor),
        Math.square(shiftedMean),
        name = "Variance")
      (mean, variance)
    }
  }

  /** $OpDocStatisticsMoments
    *
    * @group StatisticsOps
    * @param  input    Input tensor.
    * @param  axes     Axes along which to compute the mean and variance.
    * @param  weights  Optional tensor of positive weights that can be broadcast with `input`, to weigh the samples.
    *                  Defaults to `null`, meaning that equal weighting is used (i.e., all samples have weight equal to
    *                  `1`).
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Tuple containing the created op outputs: (i) the mean tensor, and (ii) the variance tensor.
    */
  def moments[T: TF : IsNotQuantized](
      input: Output[T],
      axes: Seq[Int],
      weights: Output[T] = null,
      keepDims: Boolean = false,
      name: String = "Moments"
  ): (Output[T], Output[T]) = {
    if (weights == null) {
      Op.nameScope(name) {
        val dynamicAxes = axes
        // Compute true mean while keeping the dimensions for proper broadcasting.
        var mean = Math.mean(input, axes = dynamicAxes, keepDims = true, name = "Mean")
        // Compute the sample variance (i.e., not an unbiased variance estimate).
        var variance = Math.mean(
          Math.squaredDifference(input, Basic.stopGradient(input)),
          axes = dynamicAxes, keepDims = true, name = "Variance")
        if (!keepDims) {
          mean = Basic.squeeze(mean, axes)
          variance = Basic.squeeze(variance, axes)
        }
        (mean, variance)
      }
    } else {
      // Unlike the case with no weights, this just uses a simple two-pass method.
      Op.nameScope(name) {
        val dynamicAxes = axes
        // Note that we use keepDims = true for our reductions regardless of the provided function argument. This is so
        // that the results remain broadcast-compatible with the inputs.
        val weightedInputSum = Math.sum(
          weights * input,
          axes = dynamicAxes,
          keepDims = true,
          name = "WeightedInputsSum")
        // The shape of the weights isn't necessarily the same as the input shape; it is just broadcast-compatible with
        // it. So, this expression performs broadcasting to give a per-item weight, with the same shape as
        // (weights * input). This avoids having to reason through all the broadcast logic to compute a correct sum of
        // weights.
        val broadcastedWeights = weights + Basic.zerosLike(input)
        val weightsSum = Math.sum(broadcastedWeights, axes = dynamicAxes, keepDims = true, name = "WeightsSum")
        val divisor = Math.reciprocal(weightsSum, name = "Divisor")
        var mean = Math.multiply(weightedInputSum, divisor, name = "Mean")
        var variance = Math.multiply(Math.mean(
          weights * Math.squaredDifference(mean, Basic.stopGradient(input)),
          axes = dynamicAxes, keepDims = true), divisor, name = "Variance")
        if (!keepDims) {
          mean = Basic.squeeze(mean, axes)
          variance = Basic.squeeze(variance, axes)
        }
        (mean, variance)
      }
    }
  }
}

object Statistics extends Statistics {
  private[ops] trait Implicits {
    implicit def outputConvertibleToStatisticsOps[OC, T: TF](
        value: OC
    )(implicit f: OC => Output[T]): StatisticsOps[T] = {
      new StatisticsOps(f(value))
    }

    implicit class StatisticsOps[T: TF](val output: Output[T]) {
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
      def sufficientStatistics[I: TF : IsInt32OrInt64](
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

  /** @define OpDocStatisticsSufficientStatistics
    *   The `sufficientStatistics` op calculates the sufficient statistics for the mean and variance of `input`.
    *
    *   These sufficient statistics are computed using a one pass algorithm on an input that's optionally shifted.
    *   Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data)
    *
    * @define OpDocStatisticsMomentsFromSufficientStatistics
    *   The `momentsFromSufficientStatistics` op calculates mean and variance based on some sufficient statistics.
    *
    *   This function can be directly applied to the values that the [[sufficientStatistics]] function returns.
    *
    * @define OpDocStatisticsMoments
    *   The `moments` op calculates the mean and variance of `input`, across the `axes` dimensions.
    *
    *   The mean and variance are calculated by aggregating the contents of `input` across `axes`. If `input` is 1-D and
    *   `axes = [0]` this is just the mean and variance of a vector.
    *
    *   When using these moments for batch normalization:
    *     - for so-called "global normalization", used with convolutional filters with shape
    *       `[batch, height, width, depth]`, pass `axes = [0, 1, 2]`.
    *     - for simple batch normalization pass `axes = [0]` (batch only).
    */
  private[ops] trait Documentation
}
