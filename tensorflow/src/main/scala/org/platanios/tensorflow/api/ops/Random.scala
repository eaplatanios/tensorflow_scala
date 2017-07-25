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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{FLOAT32, FLOAT64, INT32, INT64, RealNumericDataType}
import org.platanios.tensorflow.api.utilities.RandomSeed

/**
  * Contains operations for generating random numbers.
  *
  * @author Sören Brunk
  */
trait Random {

  /** Outputs random values from a normal distribution.
    *
    * @param shape The shape of the output tensor.
    * @param mean The mean of the normal distribution.
    * @param stddev The standard deviation of the normal distribution.
    * @param dataType The type of the output.
    * @param seed Used to create a random seed for the distribution.
    * @param name A name for the operation (optional).
    * @return A tensor of the specified shape filled with random normal values.
    * @throws InvalidDataTypeException If `dataType` is neither [[FLOAT32]] nor [[FLOAT64]]
    */
  def randomNormal(dataType: RealNumericDataType = FLOAT32)(
    shape: Shape,
    mean: Tensor = Tensor.fill(dataType, Shape.scalar())(0.0),
    stddev: Tensor = Tensor.fill(dataType, Shape.scalar())(1.0),
    seed: Option[Int] = None,
    name: String = "RandomNormal"
  ): Output = {
    if (dataType != FLOAT32 && dataType != FLOAT64) // TODO float16 once implemented
      throw InvalidDataTypeException(s"'dataType' '$dataType', is not 'FLOAT32' or 'FLOAT64', as required.")
    val (seed1, seed2) = RandomSeed.getSeed(seed)
    Op.createWithNameScope(name) {
      val builder = Op.Builder(opType = "RandomStandardNormal", name = name)
        .addInput(shape)
        .setAttribute("dtype", dataType)
      seed1.foreach(builder.setAttribute("seed", _))
      seed2.foreach(builder.setAttribute("seed2", _))
      val randomStandardNormal = builder.build().outputs(0)
      (randomStandardNormal * stddev) + mean
    }
  }

  /** Outputs random values from a uniform distribution.
    *
    * The generated values follow a uniform distribution in the range `[minval, maxval)`.
    * The lower bound `minval` is included in the range, while the upper bound `maxval` is excluded.
    *
    * For floats, the default range is `[0, 1)`.  For ints, at least `maxval` must be specified explicitly.
    *
    * In the integer case, the random integers are slightly biased unless `maxval - minval` is an exact power of two.
    * The bias is small for values of `maxval - minval` significantly smaller than the range of the output (either
    * `2**32` or `2**64`).
    *
    * @param shape  The shape of the output tensor.
    * @param minval A 0-D Tensor of type `dataType`. The lower bound on the range of random values to generate.
    *               Defaults to 0.
    * @param maxval A 0-D Tensor of type `dataType`. The upper bound on the range of random values to generate.
    *               Defaults to 1 if `dataType` is floating point.
    * @param seed   Used to create a random seed for the distribution.
    *     @see @{tf.set_random_seed}
    *     for behavior.
    * @param name A name for the operation (optional).
    * @return A tensor of the specified shape filled with random uniform values.
    * @throws IllegalArgumentException If `dataType` is integral and `maxval` is not specified or if `minval` or
    *                                  `maxval` are non scalar.
    * @throws InvalidDataTypeException If `dataType` isn't [[FLOAT32]], [[FLOAT64]], [[INT32]] or [[INT64]].
    */
  @throws[IllegalArgumentException]
  @throws[InvalidDataTypeException]
  def randomUniform(dataType: RealNumericDataType = FLOAT32)(
    shape: Shape,
    minval: Tensor = Tensor.fill(dataType, Shape.scalar())(0),
    maxval: Option[Tensor] = None,
    seed: Option[Int] = None,
    name: String = "RandomUniform"): Output = {
    if (minval.rank != 0 || minval.dataType != dataType)
      throw new IllegalArgumentException(
        s"'minval' (rank = ${minval.rank}, dataType = ${minval.dataType}) must be a scalar $dataType tensor.")
    maxval.foreach { max =>
      if (max.rank != 0 || max.dataType != dataType)
        throw new IllegalArgumentException(
          s"'maxval' (rank = ${max.rank}, dataType = ${max.dataType}) must be a scalar $dataType tensor.")
    }
    val (seed1, seed2) = RandomSeed.getSeed(seed)
    dataType match {
      case FLOAT32 | FLOAT64 => // TODO float16 once implemented
        Op.createWithNameScope(name) {
          val builder = Op.Builder(opType = "RandomUniform", name = name)
            .addInput(shape)
            .setAttribute("dtype", dataType)
          seed1.foreach(builder.setAttribute("seed", _))
          seed2.foreach(builder.setAttribute("seed2", _))
          val rnd = builder.build().outputs(0)
          val max = maxval.getOrElse(Tensor.fill(dataType)(1.0))
          rnd * (max - minval)
        }
      case INT32 | INT64 =>
        maxval.fold(
          throw new IllegalArgumentException(s"Must specify maxval for integer dtype $dataType"))(
          max => {
            val builder = Op.Builder(opType = "RandomUniformInt", name = name)
              .addInput(shape)
              .addInput(minval)
              .addInput(max)
              .setAttribute("dtype", dataType)
            seed1.foreach(builder.setAttribute("seed", _))
            seed2.foreach(builder.setAttribute("seed2", _))
            builder.build().outputs(0)
          }
        )
      case _ => throw InvalidDataTypeException(
        s"'dataType' '$dataType', is not 'FLOAT32' or 'FLOAT64' or 'INT32' or 'INT64', as required.")
    }
  }
}

object Random extends Random {
  private[api] object Gradients {
    GradientsRegistry.registerNonDifferentiable("RandomStandardNormal")
    GradientsRegistry.registerNonDifferentiable("RandomUniform")
  }
}