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
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types._

/** Contains functions for constructing ops related to random numbers and tensors.
  *
  * @author Emmanouil Antonios Platanios, Sören Brunk
  */
private[api] trait Random {
  /** $OpDocRandomRandomShuffle
    *
    * @group RandomOps
    * @param  value Tensor to be shuffled.
    * @param  seed  Optional random seed, used to generate a random seed pair for the random number generator, when
    *               combined with the graph-level seed.
    * @param  name  Name for the created op.
    * @return Created op output.
    */
  def randomShuffle(value: Output, seed: Option[Int] = None, name: String = "RandomShuffle"): Output = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    Op.Builder(opType = "RandomShuffle", name = name)
        .addInput(value)
        .setAttribute("seed", graphSeed.getOrElse(0))
        .setAttribute("seed2", opSeed.getOrElse(0))
        .build().outputs(0)
  }

  /** $OpDocRandomRandomUniform
    *
    * @group RandomOps
    * @param  dataType Data type for the output tensor. Must be one of: [[FLOAT16]], [[FLOAT32]], [[FLOAT64]],
    *                  [[INT32]], or [[INT64]].
    * @param  shape    Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  minValue Scalar tensor containing the inclusive lower bound on the random of random values to generate.
    *                  Defaults to `0`.
    * @param  maxValue Scalar tensor containing the exclusive upper bound on the random of random values to generate.
    *                  Defaults to `1`.
    * @param  seed     Optional random seed, used to generate a random seed pair for the random number generator, when
    *                  combined with the graph-level seed.
    * @param  name     Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `dataType` has an unsupported value.
    */
  @throws[IllegalArgumentException]
  def randomUniform(
      dataType: DataType = FLOAT32,
      shape: Output = Shape.scalar(),
      minValue: Output = 0.0,
      maxValue: Output = 1.0,
      seed: Option[Int] = None,
      name: String = "RandomUniform"
  ): Output = {
    if (!Set[DataType](FLOAT16, FLOAT32, FLOAT64, INT32, INT64).contains(dataType))
      throw new IllegalArgumentException(
        s"'dataType' ($dataType) must be one of: FLOAT16, FLOAT32, FLOAT64, INT32, or INT64.")
    Op.createWithNameScope(name, Set(shape.op, minValue.op, maxValue.op)) {
      val castedMinValue = Cast.cast(minValue, dataType)
      val castedMaxValue = Cast.cast(maxValue, dataType)
      val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
      if (dataType.isInteger) {
        Op.Builder(opType = "RandomUniformInt", name = name)
            .addInput(shape)
            .addInput(castedMinValue)
            .addInput(castedMaxValue)
            .setAttribute("seed", graphSeed.getOrElse(0))
            .setAttribute("seed2", opSeed.getOrElse(0))
            .build().outputs(0)
      } else {
        val random = Op.Builder(opType = "RandomUniform", name = name)
            .addInput(shape)
            .setAttribute("dtype", dataType)
            .setAttribute("seed", graphSeed.getOrElse(0))
            .setAttribute("seed2", opSeed.getOrElse(0))
            .build().outputs(0)
        Math.add(random * (castedMaxValue - castedMinValue), castedMinValue)
      }
    }
  }

  /** $OpDocRandomRandomNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor. Must be one of: [[FLOAT16]], [[FLOAT32]], or
    *                           [[FLOAT64]].
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @param  name              Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `dataType` has an unsupported value.
    */
  @throws[IllegalArgumentException]
  def randomNormal(
      dataType: DataType = FLOAT32,
      shape: Output = Shape.scalar(),
      mean: Output = 0.0,
      standardDeviation: Output = 1.0,
      seed: Option[Int] = None,
      name: String = "RandomNormal"
  ): Output = {
    if (dataType != FLOAT16 && dataType != FLOAT32 && dataType != FLOAT64)
      throw new IllegalArgumentException(s"'dataType' ($dataType) must be one of: FLOAT16, FLOAT32, or FLOAT64.")
    Op.createWithNameScope(name, Set(shape.op, mean.op, standardDeviation.op)) {
      val castedMean = Cast.cast(mean, dataType)
      val castedStandardDeviation = Cast.cast(standardDeviation, dataType)
      val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
      val random = Op.Builder(opType = "RandomStandardNormal", name = name)
          .addInput(shape)
          .setAttribute("dtype", dataType)
          .setAttribute("seed", graphSeed.getOrElse(0))
          .setAttribute("seed2", opSeed.getOrElse(0))
          .build().outputs(0)
      Math.add(random * castedStandardDeviation, castedMean)
    }
  }

  /** $OpDocRandomRandomTruncatedNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor. Must be one of: [[FLOAT16]], [[FLOAT32]], or
    *                           [[FLOAT64]].
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @param  name              Name for the created op.
    * @return Created op output.
    * @throws IllegalArgumentException If `dataType` has an unsupported value.
    */
  @throws[IllegalArgumentException]
  def randomTruncatedNormal(
      dataType: DataType = FLOAT32,
      shape: Output = Shape.scalar(),
      mean: Output = 0.0,
      standardDeviation: Output = 1.0,
      seed: Option[Int] = None,
      name: String = "RandomTruncatedNormal"
  ): Output = {
    if (dataType != FLOAT16 && dataType != FLOAT32 && dataType != FLOAT64)
      throw new IllegalArgumentException(s"'dataType' ($dataType) must be one of: FLOAT16, FLOAT32, or FLOAT64.")
    Op.createWithNameScope(name, Set(shape.op, mean.op, standardDeviation.op)) {
      val castedMean = Cast.cast(mean, dataType)
      val castedStandardDeviation = Cast.cast(standardDeviation, dataType)
      val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
      val random = Op.Builder(opType = "TruncatedNormal", name = name)
          .addInput(shape)
          .setAttribute("dtype", dataType)
          .setAttribute("seed", graphSeed.getOrElse(0))
          .setAttribute("seed2", opSeed.getOrElse(0))
          .build().outputs(0)
      Math.add(random * castedStandardDeviation, castedMean)
    }
  }
}

private[api] object Random extends Random {
  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("RandomShuffle")
    GradientsRegistry.registerNonDifferentiable("RandomUniform")
    GradientsRegistry.registerNonDifferentiable("RandomUniformInt")
    GradientsRegistry.registerNonDifferentiable("RandomStandardNormal")
  }

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
