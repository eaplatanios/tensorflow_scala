package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.Exception.ShapeMismatchException
import org.platanios.tensorflow.api.Shape
import org.platanios.tensorflow.api.ops.{Basic, Op}
import org.platanios.tensorflow.api.ops.variables.Variable.PartitionInformation
import org.platanios.tensorflow.api.tf.{DataType, Tensor}

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

  def apply(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    initialValue(shape, dataType, partitionInfo)
  }

  /** Generates an initial value op.
    *
    * @param  shape         Shape for the output tensor.
    * @param  dataType      Data type for the output tensor.
    * @param  partitionInfo [[PartitionInformation]] object holding additional information about how the variable is
    *                       partitioned. May be `null` if the variable is not partitioned.
    * @return Created op output.
    * @throws ShapeMismatchException If the initializer cannot produce a value with the requested shape.
    */
  @throws[ShapeMismatchException]
  def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output
}

private[api] case class InitializerWithPartitionInformation(
    initializer: Initializer, partitionInfo: PartitionInformation) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    if (partitionInfo == null)
      initializer.initialValue(shape, dataType, this.partitionInfo)
    else
      initializer.initialValue(shape, dataType, partitionInfo)
  }
}

/** Initializer that sets all elements of the variable tensor to zeros. */
private[api] object ZerosInitializer extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    Basic.zeros(shape, dataType, name = "ZerosInitializer")
  }
}

/** Initializer that sets all elements of the variable tensor to ones. */
private[api] object OnesInitializer extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    Basic.ones(shape, dataType, name = "OnesInitializer")
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
private[api] case class ConstantInitializer(value: Tensor) extends Initializer {
  @throws[ShapeMismatchException]
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    Basic.constant(value, dataType, shape, name = "ConstantInitializer")
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
private[api] case class DynamicConstantInitializer(value: Op.Output) extends Initializer {
  override val shape: Shape = value.shape

  @throws[ShapeMismatchException]
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    if (this.shape == null) {
      Basic.fill(shape, value, dataType, name = "ConstantInitializer")
    } else if (shape.isCompatibleWith(this.shape)) {
      Basic.identity(value, name = "ConstantInitializer")
    } else if (shape.rank > 0 && this.shape.rank == 0 || (this.shape.rank == 1 && this.shape(0) == 1)) {
      Basic.fill(shape, value, dataType, name = "ConstantInitializer")
    } else {
      throw ShapeMismatchException(
        s"The constant value shape '${this.shape}' is not compatible with the requested shape '$shape'.")
    }
  }
}
