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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.ShapeMismatchException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Op, Output, Random}
import org.platanios.tensorflow.api.ops.variables.Variable.PartitionInformation
import org.platanios.tensorflow.api.ops.variables.VarianceScalingInitializer.FanInScalingMode
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, IsFloat16OrFloat32OrFloat64, TF}

// TODO: [TYPES] Make initializers type safe.

/** Base trait for all variable initializers.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Initializer {
  def apply[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    Op.initializationScope {
      initialValue(shape, partitionInfo)
    }
  }

  /** Generates an initial value op.
    *
    * @param  dataType      Data type for the output tensor.
    * @param  shape         Shape for the output tensor.
    * @param  partitionInfo [[PartitionInformation]] object holding additional information about how the variable is
    *                       partitioned. May be `null` if the variable is not partitioned.
    * @return Created op output.
    * @throws ShapeMismatchException If the initializer cannot produce a value with the requested shape.
    */
  @throws[ShapeMismatchException]
  def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T]
}

private[variables] case class InitializerWithPartitionInformation(
    initializer: Initializer,
    partitionInfo: PartitionInformation
) extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    if (partitionInfo == null)
      initializer.initialValue(shape, this.partitionInfo)
    else
      initializer.initialValue(shape, partitionInfo)
  }
}

/** Initializer that sets all elements of the variable tensor to zeros. */
object ZerosInitializer extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    Op.nameScope("ZerosInitializer") {
      Basic.zeros[T](shape)
    }
  }
}

/** Initializer that sets all elements of the variable tensor to ones. */
object OnesInitializer extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    Op.nameScope("OnesInitializer") {
      Basic.ones[T](shape)
    }
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
case class ConstantInitializer[V: TF](value: Tensor[V]) extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    Basic.constant(value.castTo[T], shape, name = "ConstantInitializer")
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
case class DynamicConstantInitializer[V: TF](value: Output[V]) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    Op.colocateWith(Set.empty, ignoreExisting = true) {
      if (shape.isCompatibleWith(value.shape)) {
        Basic.identity(value.castTo[T], name = "ConstantInitializer")
      } else if (shape.rank > 0 && value.shape.rank == 0 || (value.shape.rank == 1 && value.shape(0) == 1)) {
        Op.nameScope("ConstantInitializer") {
          Basic.fill(shape)(value.castTo[T])
        }
      } else {
        throw ShapeMismatchException(
          s"The constant value shape '${value.shape}' is not compatible " +
              s"with the requested shape '$shape'.")
      }
    }
  }
}

/** Initializer that sets the value of the variable to a `value` drawn from a uniform distribution. */
case class RandomUniformInitializer(
    minValue: Tensor[Float] = 0.0f,
    maxValue: Tensor[Float] = 1.0f,
    seed: Option[Int] = None
) extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsFloat16OrFloat32OrFloat64[T] = new IsFloat16OrFloat32OrFloat64[T] {}

    Random.randomUniform(
      shape = shape,
      minValue = minValue.castTo[T],
      maxValue = maxValue.castTo[T],
      seed = seed,
      name = "RandomUniformInitializer")
  }
}

/** Initializer that sets the value of the variable to a `value` drawn from a Normal distribution. */
case class RandomNormalInitializer(
    mean: Tensor[Float] = 0.0f,
    standardDeviation: Tensor[Float] = 1.0f,
    seed: Option[Int] = None
) extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsFloat16OrFloat32OrFloat64[T] = new IsFloat16OrFloat32OrFloat64[T] {}

    Random.randomNormal(
      shape = shape,
      mean = mean.castTo[T],
      standardDeviation = standardDeviation.castTo[T],
      seed = seed,
      name = "RandomNormalInitializer")
  }
}

/** Initializer that sets the value of the variable to a `value` drawn from a truncated Normal distribution. */
case class RandomTruncatedNormalInitializer(
    mean: Tensor[Float] = 0.0f,
    standardDeviation: Tensor[Float] = 1.0f,
    seed: Option[Int] = None
) extends Initializer {
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsFloat16OrFloat32OrFloat64[T] = new IsFloat16OrFloat32OrFloat64[T] {}

    Random.randomTruncatedNormal(
      shape = shape,
      mean = mean.castTo[T],
      standardDeviation = standardDeviation.castTo[T],
      seed = seed,
      name = "RandomTruncatedNormalInitializer")
  }
}

/** Initializer capable of adapting its scale to the shape of weights tensors.
  *
  * With the Normal distribution option, samples are drawn from a truncated Normal distribution centered on zero, and
  * with standard deviation equal to `sqrt(initialScale / n)`, where `n` is:
  *
  *   - the number of input units in the weight tensor, if `mode == FanInScalingMode`,
  *   - the number of output units, if `mode == FanOutScalingMode`, or
  *   - the average of the numbers of input and output units, if `mode == FanAverageScalingMode`
  *
  * With uniform distribution option, samples are drawn from a uniform distribution within `[-limit, limit]`, where
  * `limit = sqrt(3 * initialScale / n)`.
  *
  * @param  initialScale Initial variance scale.
  * @param  scalingMode  Variance scaling mode.
  * @param  distribution Distribution to use when sampling.
  * @param  seed         Optional random seed, used to generate a random seed pair for the random number generator,
  *                      when combined with the graph-level seed.
  */
class VarianceScalingInitializer(
    val initialScale: Float = 1.0f,
    val scalingMode: VarianceScalingInitializer.ScalingMode = FanInScalingMode,
    val distribution: VarianceScalingInitializer.Distribution = VarianceScalingInitializer.NormalDistribution,
    val seed: Option[Int] = None
) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue[T: TF](
      shape: Shape,
      partitionInfo: PartitionInformation
  ): Output[T] = {
    val scale = scalingMode.scale(
      initialScale,
      if (partitionInfo != null) partitionInfo.fullShape else shape)
    distribution.initialValue(scale, shape, seed)
  }
}

object VarianceScalingInitializer {
  def apply(
      initialScale: Float = 1.0f,
      scalingMode: ScalingMode = FanInScalingMode,
      distribution: Distribution = VarianceScalingInitializer.NormalDistribution,
      seed: Option[Int] = None
  ): VarianceScalingInitializer = {
    new VarianceScalingInitializer(initialScale, scalingMode, distribution, seed)
  }

  sealed trait ScalingMode {
    def scale(initialScale: Float, shape: Shape): Float

    /** Computes the number of input and output units for the provided weights shape. */
    protected def computeFans(shape: Shape): (Long, Long) = {
      if (shape.rank == 0) {
        (0L, 0L)
      } else if (shape.rank == 1) {
        (shape(0), shape(0))
      } else if (shape.rank == 2) {
        (shape(0), shape(1))
      } else {
        // Assuming convolution kernels (2D, 3D, or more) with shape: [..., inputDepth, depth]
        val receptiveFieldSize = shape(0 :: -2).asArray.product
        (shape(-2) * receptiveFieldSize, shape(-1) * receptiveFieldSize)
      }
    }
  }

  case object FanInScalingMode extends ScalingMode {
    override def scale(initialScale: Float, shape: Shape): Float = {
      val (fanIn, _) = computeFans(shape)
      initialScale / Math.max(1L, fanIn).toFloat
    }
  }

  case object FanOutScalingMode extends ScalingMode {
    override def scale(initialScale: Float, shape: Shape): Float = {
      val (_, fanOut) = computeFans(shape)
      initialScale / Math.max(1L, fanOut).toFloat
    }
  }

  case object FanAverageScalingMode extends ScalingMode {
    override def scale(initialScale: Float, shape: Shape): Float = {
      val (fanIn, fanOut) = computeFans(shape)
      initialScale / Math.max(1.0f, (fanIn + fanOut).toFloat / 2.0f)
    }
  }

  sealed trait Distribution {
    def initialValue[T: TF](
        scale: Float,
        shape: Shape,
        seed: Option[Int] = None
    ): Output[T]
  }

  case object NormalDistribution extends Distribution {
    override def initialValue[T: TF](
        scale: Float,
        shape: Shape,
        seed: Option[Int] = None
    ): Output[T] = {

      // TODO: [TYPES] !!! Super hacky. Remove in the future.
      implicit val ev: IsFloat16OrFloat32OrFloat64[T] = new IsFloat16OrFloat32OrFloat64[T] {}

      Random.randomTruncatedNormal(
        shape = shape,
        mean = Basic.zeros[T](Shape()),
        standardDeviation = Basic.constant(Math.sqrt(scale)).castTo[T],
        seed = seed)
    }
  }

  case object UniformDistribution extends Distribution {
    override def initialValue[T: TF](
        scale: Float,
        shape: Shape,
        seed: Option[Int] = None
    ): Output[T] = {

      // TODO: [TYPES] !!! Super hacky. Remove in the future.
      implicit val ev: IsFloat16OrFloat32OrFloat64[T] = new IsFloat16OrFloat32OrFloat64[T] {}

      val limit = Math.sqrt(3.0f * scale)
      Random.randomUniform(
        shape = shape,
        minValue = Basic.constant(-limit).castTo[T],
        maxValue = Basic.constant(limit).castTo[T],
        seed = seed)
    }
  }
}

/** Glorot uniform initializer, also called the Xavier uniform initializer..
  *
  * This initializer draws samples from a uniform distribution within `[-limit, limit]`, where `limit` is equal to
  * `sqrt(6 / (fanIn + fanOut))`, where `fanIn` is the number of input units in the weight tensor and `fanOut` is the
  * number of output units in the weight tensor.
  *
  * Reference: [Understanding the difficulty of training deep feed-forward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
  *
  * @param  seed Optional random seed, used to generate a random seed pair for the random number generator, when
  *              combined with the graph-level seed.
  */
case class GlorotUniformInitializer(override val seed: Option[Int] = None)
    extends VarianceScalingInitializer(
      initialScale = 1.0f,
      scalingMode = VarianceScalingInitializer.FanAverageScalingMode,
      distribution = VarianceScalingInitializer.UniformDistribution,
      seed = seed)

/** Glorot Normal initializer, also called the Xavier Normal initializer..
  *
  * This initializer draws samples from a Normal distribution centered on zero and with standard deviation equal to
  * `sqrt(2 / (fanIn + fanOut))`, where `fanIn` is the number of input units in the weight tensor and `fanOut` is the
  * number of output units in the weight tensor.
  *
  * Reference: [Understanding the difficulty of training deep feed-forward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
  *
  * @param  seed Optional random seed, used to generate a random seed pair for the random number generator, when
  *              combined with the graph-level seed.
  */
case class GlorotNormalInitializer(override val seed: Option[Int] = None)
    extends VarianceScalingInitializer(
      initialScale = 1.0f,
      scalingMode = VarianceScalingInitializer.FanAverageScalingMode,
      distribution = VarianceScalingInitializer.NormalDistribution,
      seed = seed)
