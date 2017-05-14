package org.platanios.tensorflow.api.ops.variables

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
    */
  def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output
}

private[api] case class InitializerWithPartitionInformation(
    initializer: Initializer, partitionInfo: PartitionInformation) extends Initializer {
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    if (partitionInfo == null)
      initializer.initialValue(shape, dataType, this.partitionInfo)
    else
      initializer.initialValue(shape, dataType, partitionInfo)
  }
}

/** Initializer that sets all elements of the variable tensor to zeros. */
private[api] object ZerosInitializer extends Initializer {
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    Basic.zeros(shape, dataType)
  }
}

/** Initializer that sets all elements of the variable tensor to ones. */
private[api] object OnesInitializer extends Initializer {
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    Basic.ones(shape, dataType)
  }
}

/** Initializer that sets the value of the variable to the provided `value`. */
private[api] case class ConstantInitializer(value: Tensor) extends Initializer {
  override def initialValue(shape: Shape, dataType: DataType, partitionInfo: PartitionInformation): Op.Output = {
    Basic.constant(value, dataType, shape)
  }
}
