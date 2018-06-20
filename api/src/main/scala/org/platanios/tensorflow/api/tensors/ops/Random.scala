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
import org.platanios.tensorflow.api.tensors.{executionContext, Tensor}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Random => NativeTensorOpsRandom}

/** Contains functions for executing ops related to random numbers and tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Random {
  /** $OpDocRandomRandomUniform
    *
    * @group RandomOps
    * @param  dataType Data type for the output tensor.
    * @param  shape    Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  minValue Scalar tensor containing the inclusive lower bound on the random of random values to generate.
    *                  Defaults to `0`.
    * @param  maxValue Scalar tensor containing the exclusive upper bound on the random of random values to generate.
    *                  Defaults to `1`.
    * @param  seed     Optional random seed, used to generate a random seed pair for the random number generator, when
    *                  combined with the graph-level seed.
    * @return Result as a new tensor.
    */
  def randomUniform[D <: Int32OrInt64OrFloat16OrFloat32OrFloat64](
      dataType: D,
      shape: Tensor[INT64] = Shape()
  )(
      minValue: Tensor[D] = Tensor.zeros(dataType, Shape()),
      maxValue: Tensor[D] = Tensor.ones(dataType, Shape()),
      seed: Option[Int] = None
  ): Tensor[D] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    if (dataType.isInteger) {
      Tensor.fromNativeHandle[D](NativeTensorOpsRandom.randomUniformInt(
        executionContext.value.nativeHandle, shape.nativeHandle, minValue.nativeHandle,
        maxValue.nativeHandle, graphSeed.getOrElse(0).toLong, opSeed.getOrElse(0).toLong))
    } else {
      val random = Tensor.fromNativeHandle[D](NativeTensorOpsRandom.randomUniform(
        executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
        opSeed.getOrElse(0).toLong))
      Math.add(random * (maxValue - minValue), minValue)
    }
  }

  /** $OpDocRandomRandomNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor.
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @return Result as a new tensor.
    */
  def randomNormal[D <: Float16OrFloat32OrFloat64](
      dataType: D,
      shape: Tensor[INT64] = Shape()
  )(
      mean: Tensor[D] = Tensor.zeros(dataType, Shape()),
      standardDeviation: Tensor[D] = Tensor.ones(dataType, Shape()),
      seed: Option[Int] = None
  ): Tensor[D] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val random = Tensor.fromNativeHandle[D](NativeTensorOpsRandom.randomStandardNormal(
      executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
      opSeed.getOrElse(0).toLong))
    Math.add(random * standardDeviation, mean)
  }

  /** $OpDocRandomRandomTruncatedNormal
    *
    * @group RandomOps
    * @param  dataType          Data type for the output tensor.
    * @param  shape             Rank-1 tensor containing the shape of the output tensor. Defaults to a scalar tensor.
    * @param  mean              Scalar tensor containing the mean of the Normal distribution. Defaults to `0`.
    * @param  standardDeviation Scalar tensor containing the standard deviation of the Normal distribution. Defaults to
    *                           `1`.
    * @param  seed              Optional random seed, used to generate a random seed pair for the random number
    *                           generator, when combined with the graph-level seed.
    * @return Result as a new tensor.
    */
  def randomTruncatedNormal[D <: Float16OrFloat32OrFloat64](
      dataType: D,
      shape: Tensor[INT64] = Shape()
  )(
      mean: Tensor[D] = Tensor.zeros(dataType, Shape()),
      standardDeviation: Tensor[D] = Tensor.ones(dataType, Shape()),
      seed: Option[Int] = None
  ): Tensor[D] = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val random = Tensor.fromNativeHandle[D](NativeTensorOpsRandom.truncatedNormal(
      executionContext.value.nativeHandle, shape.nativeHandle, dataType.cValue, graphSeed.getOrElse(0).toLong,
      opSeed.getOrElse(0).toLong))
    Math.add(random * standardDeviation, mean)
  }
}

private[api] object Random extends Random
