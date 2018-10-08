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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.utilities.DefaultsTo.FloatDefault

/** Contains functions for constructing ops related to random numbers and tensors.
  *
  * @author Emmanouil Antonios Platanios, Sören Brunk
  */
trait Random {
  /** $OpDocRandomRandomShuffle
    *
    * @group RandomOps
    * @param  value Tensor to be shuffled.
    * @param  seed  Optional random seed, used to generate a random seed pair for the random number generator, when
    *               combined with the graph-level seed.
    * @param  name  Name for the created op.
    * @tparam T Tensor data type.
    * @return Created op output.
    */
  def randomShuffle[T: TF](
      value: Output[T],
      seed: Option[Int] = None,
      name: String = "RandomShuffle"
  ): Output[T] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    Op.Builder[Output[T], Output[T]](
      opType = "RandomShuffle",
      name = name,
      input = value
    ).setAttribute("seed", graphSeed.getOrElse(0))
        .setAttribute("seed2", opSeed.getOrElse(0))
        .build().output
  }

  /** $OpDocRandomRandomUniform
    *
    * @group RandomOps
    * @param  shape    Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  minValue Scalar tensor containing the inclusive lower bound on the random of random values to generate.
    *                  Defaults to `0`.
    * @param  maxValue Scalar tensor containing the exclusive upper bound on the random of random values to generate.
    *                  Defaults to `1`.
    * @param  seed     Optional random seed, used to generate a random seed pair for the random number generator, when
    *                  combined with the graph-level seed.
    * @param  name     Name for the created op.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def randomUniform[T: FloatDefault : TF : IsInt32OrInt64OrFloat16OrFloat32OrFloat64, I: TF : IsInt32OrInt64](
      shape: Output[I],
      minValue: Output[T] = null,
      maxValue: Output[T] = null,
      seed: Option[Int] = None,
      name: String = "RandomUniform"
  ): Output[T] = {
    val dataType = TF[T].dataType
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val minValueWithDefault = if (minValue == null) Basic.zeros[T](Shape()) else minValue
    val maxValueWithDefault = if (maxValue == null) Basic.ones[T](Shape()) else maxValue
    if (dataType.isInteger) {
      Op.Builder[(Output[I], Output[T], Output[T]), Output[T]](
        opType = "RandomUniformInt",
        name = name,
        input = (shape, minValueWithDefault, maxValueWithDefault)
      ).setAttribute("seed", graphSeed.getOrElse(0))
          .setAttribute("seed2", opSeed.getOrElse(0))
          .build().output
    } else {
      val random = Op.Builder[Output[I], Output[T]](
        opType = "RandomUniform",
        name = name,
        input = shape
      ).setAttribute("dtype", dataType)
          .setAttribute("seed", graphSeed.getOrElse(0))
          .setAttribute("seed2", opSeed.getOrElse(0))
          .build().output
      Math.add(random * (maxValueWithDefault - minValueWithDefault), minValueWithDefault)
    }
  }

  /** $OpDocRandomRandomNormal
    *
    * @group RandomOps
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @param  name              Name for the created op.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def randomNormal[T: FloatDefault : TF : IsFloat16OrFloat32OrFloat64, I: TF : IsInt32OrInt64](
      shape: Output[I],
      mean: Output[T] = null,
      standardDeviation: Output[T] = null,
      seed: Option[Int] = None,
      name: String = "RandomNormal"
  ): Output[T] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val meanWithDefault = if (mean == null) Basic.zeros[T](Shape()) else mean
    val standardDeviationWithDefault = if (standardDeviation == null) Basic.ones[T](Shape()) else standardDeviation
    val random = Op.Builder[Output[I], Output[T]](
      opType = "RandomStandardNormal",
      name = name,
      input = shape
    ).setAttribute("dtype", TF[T].dataType)
        .setAttribute("seed", graphSeed.getOrElse(0))
        .setAttribute("seed2", opSeed.getOrElse(0))
        .build().output
    Math.add(random * standardDeviationWithDefault, meanWithDefault)
  }

  /** $OpDocRandomRandomTruncatedNormal
    *
    * @group RandomOps
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @param  name              Name for the created op.
    * @tparam T Tensor data type.
    * @tparam I Tensor shape type.
    * @return Created op output.
    */
  def randomTruncatedNormal[T: FloatDefault : TF : IsFloat16OrFloat32OrFloat64, I: TF : IsInt32OrInt64](
      shape: Output[I],
      mean: Output[T] = null,
      standardDeviation: Output[T] = null,
      seed: Option[Int] = None,
      name: String = "RandomTruncatedNormal"
  ): Output[T] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val meanWithDefault = if (mean == null) Basic.zeros[T](Shape()) else mean
    val standardDeviationWithDefault = if (standardDeviation == null) Basic.ones[T](Shape()) else standardDeviation
    val random = Op.Builder[Output[I], Output[T]](
      opType = "TruncatedNormal",
      name = name,
      input = shape
    ).setAttribute("dtype", TF[T].dataType)
        .setAttribute("seed", graphSeed.getOrElse(0))
        .setAttribute("seed2", opSeed.getOrElse(0))
        .build().output
    Math.add(random * standardDeviationWithDefault, meanWithDefault)
  }
}

object Random extends Random {
  /** @define OpDocRandomRandomShuffle
    *   The `randomShuffle` op randomly shuffles a tensor along its first axis.
    *
    *   The tensor is shuffled along axis `0`, such that each `value(j)` is mapped to one and only one `output(i)`. For
    *   example, a mapping that might occur for a 3x2 tensor is:
    *   {{{
    *     [[1, 2],       [[5, 6],
    *      [3, 4],  ==>   [1, 2],
    *      [5, 6]]        [3, 4]]
    *   }}}
    *
    * @define OpDocRandomRandomUniform
    *   The `randomUniform` op outputs random values drawn from a uniform distribution.
    *
    *   The generated values follow a uniform distribution in the range `[minValue, maxValue)`. The lower bound
    *   `minValue` is included in the range, while the upper bound `maxValue` is not.
    *
    *   In the integer case, the random integers are slightly biased unless `maxValue - minValue` is an exact power of
    *   two. The bias is small for values of `maxValue - minValue` significantly smaller than the range of the output
    *   (either `2^32` or `2^64`, depending on the data type).
    *
    * @define OpDocRandomRandomNormal
    *   The `randomNormal` op outputs random values drawn from a Normal distribution.
    *
    *   The generated values follow a Normal distribution with mean `mean` and standard deviation `standardDeviation`.
    *
    * @define OpDocRandomRandomTruncatedNormal
    *   The `randomTruncatedNormal` op outputs random values drawn from a truncated Normal distribution.
    *
    *   The generated values follow a Normal distribution with mean `mean` and standard deviation `standardDeviation`,
    *   except that values whose magnitude is more than two standard deviations from the mean are dropped and resampled.
    */
  private[ops] trait Documentation
}
