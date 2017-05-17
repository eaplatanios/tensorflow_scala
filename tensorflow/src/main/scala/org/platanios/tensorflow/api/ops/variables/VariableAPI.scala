package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}
import org.platanios.tensorflow.api.ops.variables
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, FLOAT32}

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait VariableAPI {
  type Variable = variables.Variable
  type PartitionedVariable = variables.PartitionedVariable
  type VariableGetter = variables.Variable.VariableGetter
  type VariableInitializer = variables.Initializer
  type VariableRegularizer = variables.Regularizer
  type VariablePartitioner = variables.Partitioner
  type VariableStore = variables.VariableStore
  type VariableScope = variables.VariableScope

  val Variable      = variables.Variable
  val VariableStore = variables.VariableStore
  val VariableScope = variables.VariableScope

  val zerosInitializer = variables.ZerosInitializer
  val onesInitializer  = variables.OnesInitializer

  def constantInitializer(value: Tensor) = variables.ConstantInitializer(value)
  def constantInitializer(value: Op.Output) = variables.DynamicConstantInitializer(value)

  type Saver = variables.Saver
  val Saver = variables.Saver

  def variable(
      name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
      regularizer: VariableRegularizer = null, trainable: Boolean = true, reuse: java.lang.Boolean = null,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): Variable = {
    Variable.getVariable(
      name, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice)
  }

  def partitionedVariable(
      name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
      regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, trainable: Boolean = true,
      reuse: java.lang.Boolean = null, collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): PartitionedVariable = {
    Variable.getPartitionedVariable(
      name, shape, dataType, initializer, regularizer, partitioner, trainable, reuse, collections, cachingDevice)
  }

  def localVariable(
      name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
      regularizer: VariableRegularizer = null, reuse: java.lang.Boolean = null,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): Variable = {
    Variable.getLocalVariable(name, shape, dataType, initializer, regularizer, reuse, collections, cachingDevice)
  }

  def localPartitionedVariable(
      name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
      regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, reuse: java.lang.Boolean = null,
      collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): PartitionedVariable = {
    Variable.getLocalPartitionedVariable(
      name, shape, dataType, initializer, regularizer, partitioner, reuse, collections, cachingDevice)
  }
}
