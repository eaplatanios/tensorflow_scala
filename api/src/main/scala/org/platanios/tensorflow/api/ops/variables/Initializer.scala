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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.ShapeMismatchException
import org.platanios.tensorflow.api.ops.{Basic, Output, Random}
import org.platanios.tensorflow.api.ops.variables.Variable.PartitionInformation
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

// TODO: [VARIABLE_INITIALIZERS] RandomUniform/Normal, TruncatedNormal, UniformUnitScaling, Orthogonal.
// TODO: [VARIABLE_INITIALIZERS] VarianceScaling, Glorot/Xavier Uniform and Normal.

/** Base trait for all variable initializers.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Initializer {
  /** Shape of the values produced by this initializer. If `null`, then the initializer may produce values of any
    * shape. */
  val shape: Shape = null

  def apply(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    initialValue(dataType, shape, partitionInfo)
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
  def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output
}

private[variables] case class InitializerWithPartitionInformation(
    initializer: Initializer, partitionInfo: PartitionInformation) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    if (partitionInfo == null)
      initializer.initialValue(dataType, shape, this.partitionInfo)
    else
      initializer.initialValue(dataType, shape, partitionInfo)
  }
}

/** Initializer that sets all elements of the variable tensor to zeros. */
object ZerosInitializer extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    Basic.zeros(dataType, shape, name = "ZerosInitializer")
  }
}

/** Initializer that sets all elements of the variable tensor to ones. */
object OnesInitializer extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    Basic.ones(dataType, shape, name = "OnesInitializer")
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
case class ConstantInitializer(value: Tensor) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    Basic.constant(value, dataType, shape, name = "ConstantInitializer")
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
case class DynamicConstantInitializer(value: Output) extends Initializer {
  override val shape: Shape = value.shape

  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    if (this.shape == null) {
      Basic.fill(dataType, shape)(value, name = "ConstantInitializer")
    } else if (shape.isCompatibleWith(this.shape)) {
      Basic.identity(value, name = "ConstantInitializer")
    } else if (shape.rank > 0 && this.shape.rank == 0 || (this.shape.rank == 1 && this.shape(0) == 1)) {
      Basic.fill(dataType, shape)(value, name = "ConstantInitializer")
    } else {
      throw ShapeMismatchException(
        s"The constant value shape '${this.shape}' is not compatible with the requested shape '$shape'.")
    }
  }
}

/** Initializer that sets the value of the variable to a `value` drawn from a uniform distribution. */
case class RandomUniformInitializer(
    minValue: Tensor = 0.0, maxValue: Tensor = 1.0, seed: Option[Int] = None) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    Random.randomUniform(
      dataType, shape, minValue = minValue, maxValue = maxValue, seed = seed, name = "RandomUniformInitializer")
  }
}

/** Initializer that sets the value of the variable to a `value` drawn from a Normal distribution. */
case class RandomNormalInitializer(
    mean: Tensor = 0.0, standardDeviation: Tensor = 1.0, seed: Option[Int] = None) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(dataType: DataType, shape: Shape, partitionInfo: PartitionInformation): Output = {
    Random.randomNormal(
      dataType, shape, mean = mean, standardDeviation = standardDeviation, seed = seed,
      name = "RandomNormalInitializer")
  }
}
