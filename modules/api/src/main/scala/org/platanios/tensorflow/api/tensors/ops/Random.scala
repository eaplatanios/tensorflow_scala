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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Random => NativeTensorOpsRandom}

/** Contains functions for executing ops related to random numbers and tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Random {
  /** $OpDocRandomRandomShuffle
    *
    * @group RandomOps
    * @param  value Tensor to be shuffled.
    * @param  seed  Optional random seed, used to generate a random seed pair for the random number generator, when
    *               combined with the graph-level seed.
    * @return Result as a new tensor.
    */
  def randomShuffle[T](
      value: Tensor[T],
      seed: Option[Int] = None
  ): Tensor[T] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomShuffle(
      executionContext.value.nativeHandle, value.nativeHandle,
      graphSeed.getOrElse(0).toLong, opSeed.getOrElse(0).toLong))
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
    * @tparam T Data type for the output tensor.
    * @tparam I Shape type.
    * @return Result as a new tensor.
    */
  def randomUniform[T: IsInt32OrInt64OrFloat16OrFloat32OrFloat64 : SupportedType, I: IsInt32OrInt64](
      shape: Tensor[I],
      minValue: Tensor[T] = null,
      maxValue: Tensor[T] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    val dataType = implicitly[SupportedType[T]].dataType
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val minValueWithDefault = if (minValue == null) Tensor.zeros[T](Shape()) else minValue
    val maxValueWithDefault = if (maxValue == null) Tensor.ones[T](Shape()) else maxValue
    if (dataType.isInteger) {
      Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomUniformInt(
        executionContext.value.nativeHandle, shape.nativeHandle, minValueWithDefault.nativeHandle,
        maxValueWithDefault.nativeHandle, graphSeed.getOrElse(0).toLong, opSeed.getOrElse(0).toLong))
    } else {
      val random = Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomUniform(
        executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
        opSeed.getOrElse(0).toLong))
      Math.add(random * (maxValueWithDefault - minValueWithDefault), minValue)
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
    * @tparam T Data type for the output tensor.
    * @tparam I Shape type.
    * @return Result as a new tensor.
    */
  def randomNormal[T: IsFloat16OrFloat32OrFloat64 : SupportedType, I: IsInt32OrInt64](
      shape: Tensor[I],
      mean: Tensor[T] = null,
      standardDeviation: Tensor[T] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    val dataType = implicitly[SupportedType[T]].dataType
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val meanWithDefault = if (mean == null) Tensor.zeros[T](Shape()) else mean
    val standardDeviationWithDefault = if (standardDeviation == null) Tensor.ones[T](Shape()) else standardDeviation
    val random = Tensor.fromNativeHandle[T](NativeTensorOpsRandom.randomStandardNormal(
      executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
      opSeed.getOrElse(0).toLong))
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
    * @tparam T Data type for the output tensor.
    * @tparam I Shape type.
    * @return Result as a new tensor.
    */
  def randomTruncatedNormal[T: IsFloat16OrFloat32OrFloat64 : SupportedType, I: IsInt32OrInt64](
      shape: Tensor[I],
      mean: Tensor[T] = null,
      standardDeviation: Tensor[T] = null,
      seed: Option[Int] = None
  ): Tensor[T] = {
    val dataType = implicitly[SupportedType[T]].dataType
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val meanWithDefault = if (mean == null) Tensor.zeros[T](Shape()) else mean
    val standardDeviationWithDefault = if (standardDeviation == null) Tensor.ones[T](Shape()) else standardDeviation
    val random = Tensor.fromNativeHandle[T](NativeTensorOpsRandom.truncatedNormal(
      executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
      opSeed.getOrElse(0).toLong))
    Math.add(random * standardDeviationWithDefault, meanWithDefault)
  }
}

object Random extends Random
