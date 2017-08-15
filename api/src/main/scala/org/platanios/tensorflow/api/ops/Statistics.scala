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

import org.platanios.tensorflow.api.ops.Output.Implicits._
import org.platanios.tensorflow.api.types.{FLOAT16, FLOAT32}

/** Contains functions for constructing ops related to statistics.
  *
  * @author Emmanouil Antonios Platanios, Sören Brunk
  */
private[ops] trait Statistics {
  /** Creates an op that calculates the sufficient statistics for the mean and variance of `input`.
    *
    * These sufficient statistics are computed using a one pass algorithm on an input that's optionally shifted.
    * Source: [https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data)
    *
    * @param  input    Input tensor.
    * @param  axes     Axes along which to compute the mean and variance.
    * @param  shift    Optional tensor containing the value by which to shift the data for numerical stability. Defaults
    *                  to `null`, meaning that no shift needs to be performed. A shift close to the true mean provides
    *                  the most numerically stable results.
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Tuple containing the following created op outputs:
    *         - Count: The number of elements to average over.
    *         - Mean Sufficient Statistic: The (possibly shifted) sum of the elements in the tensor.
    *         - Variance Sufficient Statistic: The (possibly shifted) sum of squares of the elements in the tensor.
    *         - Shift: The shift by which the mean must be corrected, or `null` if no shift was used.
    */
  def sufficientStatistics(
      input: Output, axes: Array[Int], shift: Output = null, keepDims: Boolean = false,
      name: String = "SufficientStatistics"): (Output, Output, Output, Output) = {
    Op.createWithNameScope(name, Set(input.op)) {
      val dynamicAxes: Output = axes
      val inputShape = input.shape
      val counts = {
        if (axes.map(inputShape(_)).forall(_ > -1))
          Basic.constant(axes.map(inputShape(_)).product, input.dataType)
        else
          Math.prod(Basic.gather(Math.cast(Basic.shape(input), input.dataType), dynamicAxes))
      }
      val mSS = if (shift == null) input else input - shift
      val vSS = if (shift == null) Math.square(input) else Math.squaredDifference(input, shift)
      val meanSS = Math.sum(mSS, axes = dynamicAxes, keepDims = keepDims, name = "MeanSS")
      val varSS = Math.sum(vSS, axes = dynamicAxes, keepDims = keepDims, name = "VarSS")
      (counts, meanSS, varSS, shift)
    }
  }

  /** Creates an op that calculates mean and variance based on some sufficient statistics.
    *
    * This function can be directly applied to the values that the [[sufficientStatistics]] function returns.
    *
    * @param  counts Total number of elements over which the provided sufficient statistics were computed.
    * @param  meanSS Mean sufficient statistics: the (possibly shifted) sum of the elements.
    * @param  varSS  Variance sufficient statistics: the (possibly shifted) sum of squares of the elements.
    * @param  shift  The shift by which the mean must be corrected, or `null` if no shift was used.
    * @param  name   Name for the created op.
    * @return Tuple containing the created op outputs: (i) the mean tensor, and (ii) the variance tensor.
    */
  def momentsFromSufficientStatistics(
      counts: Output, meanSS: Output, varSS: Output, shift: Output = null,
      name: String = "MomentsFromSufficientStatistics"): (Output, Output) = {
    Op.createWithNameScope(name, Set(counts.op, meanSS.op, varSS.op)) {
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
      val variance = Math.subtract(Math.multiply(varSS, divisor), Math.square(shiftedMean), name = "Variance")
      (mean, variance)
    }
  }

  /** Creates an op that calculates the mean and variance of `input`, across the `axes` dimensions.
    *
    * The mean and variance are calculated by aggregating the contents of `input` across `axes`. If `input` is 1-D and
    * `axes = [0]` this is just the mean and variance of a vector.
    *
    * When using these moments for batch normalization:
    *   - for so-called "global normalization", used with convolutional filters with shape
    *     `[batch, height, width, depth]`, pass `axes = [0, 1, 2]`.
    *   - for simple batch normalization pass `axes = [0]` (batch only).
    *
    * @param  input    Input tensor.
    * @param  axes     Axes along which to compute the mean and variance.
    * @param  weights  Optional tensor of positive weights that can be broadcast with `input`, to weigh the samples.
    *                  Defaults to `null`, meaning that equal weighting is used (i.e., all samples have weight equal to
    *                  `1`).
    * @param  keepDims If `true`, retain the reduced axes.
    * @param  name     Name for the created op.
    * @return Tuple containing the created op outputs: (i) the mean tensor, and (ii) the variance tensor.
    */
  def moments(
      input: Output, axes: Array[Int], weights: Output = null, keepDims: Boolean = false,
      name: String = "Moments"): (Output, Output) = {
    if (weights == null) {
      Op.createWithNameScope(name, Set(input.op)) {
        val dynamicAxes: Output = axes
        // The dynamic range of FLOAT16 is too limited to support the collection of sufficient statistics. As a
        // workaround we simply perform the operations on 32-bit floats before converting the mean and variance back to
        // FLOAT16.
        val preciseInput = if (input.dataType == FLOAT16) Math.cast(input, FLOAT32) else input
        // Compute true mean while keeping the dimensions for proper broadcasting.
        var mean = Math.mean(preciseInput, axes = dynamicAxes, keepDims = true, name = "Mean")
        // Compute the sample variance (i.e., not an unbiased variance estimate).
        var variance = Math.mean(
          Math.squaredDifference(preciseInput, Basic.stopGradient(input)),
          axes = dynamicAxes, keepDims = true, name = "Variance")
        if (!keepDims) {
          mean = Basic.squeeze(mean, axes)
          variance = Basic.squeeze(variance, axes)
        }
        // Cast back to FLOAT16 if necessary.
        if (input.dataType == FLOAT16)
          (Math.cast(mean, FLOAT16), Math.cast(variance, FLOAT16))
        else
          (mean, variance)
      }
    } else {
      // Unlike the case with no weights, this just uses a simple two-pass method.
      Op.createWithNameScope(name, Set(input.op, weights.op)) {
        val dynamicAxes: Output = axes
        // The dynamic range of FLOAT16 is too limited to support the collection of sufficient statistics. As a
        // workaround we simply perform the operations on 32-bit floats before converting the mean and variance back to
        // FLOAT16.
        val preciseInput = if (input.dataType == FLOAT16) Math.cast(input, FLOAT32) else input
        val preciseWeights = Math.cast(weights, preciseInput.dataType)
        // Note that we use keepDims = true for our reductions regardless of the provided function argument. This is so
        // that the results remain broadcast-compatible with the inputs.
        val weightedInputSum = Math.sum(
          preciseWeights * preciseInput, axes = dynamicAxes, keepDims = true, name = "WeightedInputsSum")
        // The shape of the weights isn't necessarily the same as the input shape; it is just broadcast-compatible with
        // it. So, this expression performs broadcasting to give a per-item weight, with the same shape as
        // (weights * input). This avoids having to reason through all the broadcast logic to compute a correct sum of
        // weights.
        val broadcastedWeights = preciseWeights + Basic.zerosLike(preciseInput)
        val weightsSum = Math.sum(broadcastedWeights, axes = dynamicAxes, keepDims = true, name = "WeightsSum")
        val divisor = Math.reciprocal(weightsSum, name = "Divisor")
        var mean = Math.multiply(weightedInputSum, divisor, name = "Mean")
        var variance = Math.multiply(Math.mean(
          preciseWeights * Math.squaredDifference(mean, Basic.stopGradient(preciseInput)),
          axes = dynamicAxes, keepDims = true), divisor, name = "Variance")
        if (!keepDims) {
          mean = Basic.squeeze(mean, axes)
          variance = Basic.squeeze(variance, axes)
        }
        // Cast back to FLOAT16 if necessary.
        if (input.dataType == FLOAT16)
          (Math.cast(mean, FLOAT16), Math.cast(variance, FLOAT16))
        else
          (mean, variance)
      }
    }
  }
}

private[ops] object Statistics extends Statistics
