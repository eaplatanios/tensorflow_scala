package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends Implicits {
  private[api] val defaultGraph: core.Graph = core.Graph()

  object tf {
    type Graph = core.Graph
    val Graph = core.Graph

    val defaultGraph: core.Graph = api.defaultGraph

    type Indexer = core.Indexer
    type Index = core.Index
    type Slice = core.Slice

    val Indexer  = core.Indexer
    val Index    = core.Index
    val Slice    = core.Slice
    val NewAxis  = core.NewAxis
    val Ellipsis = core.Ellipsis

    type Session = core.Session
    val Session = core.Session

    type Shape = core.Shape
    val Shape = core.Shape

    type DeviceSpecification = core.DeviceSpecification
    val DeviceSpecification = core.DeviceSpecification

    private[api] type ShapeMismatchException = core.exception.ShapeMismatchException
    private[api] type GraphMismatchException = core.exception.GraphMismatchException
    private[api] type IllegalNameException = core.exception.IllegalNameException
    private[api] type InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
    private[api] type InvalidGraphElementException = core.exception.InvalidGraphElementException
    private[api] type InvalidShapeException = core.exception.InvalidShapeException
    private[api] type InvalidIndexerException = core.exception.InvalidIndexerException
    private[api] type InvalidDataTypeException = core.exception.InvalidDataTypeException
    private[api] type OpBuilderUsedException = core.exception.OpBuilderUsedException

    private[api] val ShapeMismatchException              = core.exception.ShapeMismatchException
    private[api] val GraphMismatchException              = core.exception.GraphMismatchException
    private[api] val IllegalNameException                = core.exception.IllegalNameException
    private[api] val InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
    private[api] val InvalidGraphElementException        = core.exception.InvalidGraphElementException
    private[api] val InvalidShapeException               = core.exception.InvalidShapeException
    private[api] val InvalidIndexerException             = core.exception.InvalidIndexerException
    private[api] val InvalidDataTypeException            = core.exception.InvalidDataTypeException
    private[api] val OpBuilderUsedException              = core.exception.OpBuilderUsedException

    type SupportedType[T] = types.SupportedType[T]
    type FixedSizeSupportedType[T] = types.FixedSizeSupportedType[T]
    type NumericSupportedType[T] = types.NumericSupportedType[T]
    type SignedNumericSupportedType[T] = types.SignedNumericSupportedType[T]
    type RealNumericSupportedType[T] = types.RealNumericSupportedType[T]
    type ComplexNumericSupportedType[T] = types.ComplexNumericSupportedType[T]

    type DataType = types.DataType
    type FixedSizeDataType = types.FixedSizeDataType
    type NumericDataType = types.NumericDataType
    type SignedNumericDataType = types.SignedNumericDataType
    type RealNumericDataType = types.RealNumericDataType
    type ComplexNumericDataType = types.ComplexNumericDataType

    val DataType = types.DataType

    val STRING   = types.STRING
    val BOOLEAN  = types.BOOLEAN
    // val FLOAT16 = types.TFFloat16
    val FLOAT32  = types.FLOAT32
    val FLOAT64  = types.FLOAT64
    // val BFLOAT16 = types.TFBFloat16
    // val COMPLEX64 = types.TFComplex64
    // val COMPLEX128 = types.TFComplex128
    val INT8     = types.INT8
    val INT16    = types.INT16
    val INT32    = types.INT32
    val INT64    = types.INT64
    val UINT8    = types.UINT8
    val UINT16   = types.UINT16
    val QINT8    = types.QINT8
    val QINT16   = types.QINT16
    val QINT32   = types.QINT32
    val QUINT8   = types.QUINT8
    val QUINT16  = types.QUINT16
    val RESOURCE = types.RESOURCE

    type Tensor = tensors.Tensor
    type FixedSizeTensor = tensors.FixedSizeTensor
    type NumericTensor = tensors.NumericTensor

    val Tensor = tensors.Tensor

    type Order = tensors.Order
    val RowMajorOrder = tensors.RowMajorOrder

    type Op = ops.Op
    val Op = ops.Op

    type OpCreationContext = ops.OpCreationContext
    type OpSpecification = ops.OpSpecification

    //region Op Construction Aliases

    def currentGraph: Graph = ops.Op.currentGraph
    def currentNameScope: String = ops.Op.currentNameScope
    def currentVariableScope: VariableScope = ops.Op.currentVariableScope
    def currentDevice: OpSpecification => String = ops.Op.currentDevice
    def currentColocationOps: Set[Op] = ops.Op.currentColocationOps
    def currentControlDependencies: Set[Op] = ops.Op.currentControlDependencies
    def currentAttributes: Map[String, Any] = ops.Op.currentAttributes
    def currentContainer: String = ops.Op.currentContainer

    // TODO: Maybe remove "current" from the above names.

    private[api] def currentVariableStore: VariableStore = ops.Op.currentVariableStore

    def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): Op = {
      ops.Op.currentGraph.globalVariablesInitializer(name)
    }

    def localVariablesInitializer(name: String = "LocalVariablesInitializer"): Op = {
      ops.Op.currentGraph.localVariablesInitializer(name)
    }

    def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): Op = {
      ops.Op.currentGraph.modelVariablesInitializer(name)
    }

    def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): Op = {
      ops.Op.currentGraph.trainableVariablesInitializer(name)
    }

    def createWith[R](
        graph: Graph = null, nameScope: String = null, device: ops.OpSpecification => String = _ => "",
        colocationOps: Set[Op] = null, controlDependencies: Set[Op] = null, attributes: Map[String, Any] = null,
        container: String = null)(block: => R): R = {
      ops.Op.createWith(graph, nameScope, device, colocationOps, controlDependencies, attributes, container)(block)
    }

    def createWithNameScope[R](nameScope: String, values: Set[Op] = Set.empty[Op])(block: => R): R = {
      ops.Op.createWithNameScope(nameScope, values)(block)
    }

    def createWithVariableScope[R](
        name: String, reuse: java.lang.Boolean = null, dataType: DataType = null,
        initializer: VariableInitializer = null, regularizer: VariableRegularizer = null,
        partitioner: VariablePartitioner = null, cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false)
        (block: => R): R = {
      ops.variables.VariableScope.createWithVariableScope(
        name, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter, isDefaultName,
        isPure)(block)
    }

    def createWithUpdatedVariableScope[R](
        variableScope: VariableScope, reuse: java.lang.Boolean = null, dataType: DataType = null,
        initializer: VariableInitializer = null, regularizer: VariableRegularizer = null,
        partitioner: VariablePartitioner = null, cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null, isPure: Boolean = false)(block: => R): R = {
      ops.variables.VariableScope.createWithUpdatedVariableScope(
        variableScope, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter,
        isPure)(block)
    }

    def colocateWith[R](colocationOps: Set[Op], ignoreExisting: Boolean = false)(block: => R): R = {
      ops.Op.colocateWith(colocationOps, ignoreExisting)(block)
    }

    //region Basic Ops

    def constant(
        tensor: Tensor, dataType: DataType = null, shape: Shape = null, name: String = "Constant"): Op.Output = {
      ops.Basic.constant(tensor, dataType, shape, name)
    }

    def placeholder(dataType: DataType, shape: Shape = null, name: String = "Placeholder"): Op.Output = {
      ops.Basic.placeholder(dataType, shape, name)
    }

    //endregion Basic Ops

    //region Variables

    type Variable = ops.variables.Variable
    type PartitionedVariable = ops.variables.PartitionedVariable
    type VariableGetter = ops.variables.Variable.VariableGetter
    type VariableInitializer = ops.variables.Initializer
    type VariableRegularizer = ops.variables.Regularizer
    type VariablePartitioner = ops.variables.Partitioner
    type VariableStore = ops.variables.VariableStore
    type VariableScope = ops.variables.VariableScope

    val Variable      = ops.variables.Variable
    val VariableStore = ops.variables.VariableStore
    val VariableScope = ops.variables.VariableScope

    val zerosInitializer = ops.variables.ZerosInitializer
    val onesInitializer  = ops.variables.OnesInitializer

    def constantInitializer(value: Tensor) = ops.variables.ConstantInitializer(value)
    def constantInitializer(value: Op.Output) = ops.variables.DynamicConstantInitializer(value)

    def variable(
        name: String, shape: Shape = null, dataType: tf.DataType = tf.FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, trainable: Boolean = true, reuse: java.lang.Boolean = null,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): Variable = {
      Variable.getVariable(
        name, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    }

    def partitionedVariable(
        name: String, shape: Shape = null, dataType: tf.DataType = tf.FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, trainable: Boolean = true,
        reuse: java.lang.Boolean = null, collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): PartitionedVariable = {
      Variable.getPartitionedVariable(
        name, shape, dataType, initializer, regularizer, partitioner, trainable, reuse, collections, cachingDevice)
    }

    def localVariable(
        name: String, shape: Shape = null, dataType: tf.DataType = tf.FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, reuse: java.lang.Boolean = null,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): Variable = {
      Variable.getLocalVariable(name, shape, dataType, initializer, regularizer, reuse, collections, cachingDevice)
    }

    def localPartitionedVariable(
        name: String, shape: Shape = null, dataType: tf.DataType = tf.FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, reuse: java.lang.Boolean = null,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): PartitionedVariable = {
      Variable.getLocalPartitionedVariable(
        name, shape, dataType, initializer, regularizer, partitioner, reuse, collections, cachingDevice)
    }

    //endregion Variables

    val Gradients         = ops.Gradients
    val GradientsRegistry = ops.Gradients.Registry

    ops.Basic.Gradients
    ops.Math.Gradients
    ops.variables.Variable.Gradients

    object train {
      type Optimizer = ops.optimizers.Optimizer
      val Optimizer = ops.optimizers.Optimizer

      type GradientDescent = ops.optimizers.GradientDescent
      val GradientDescent = ops.optimizers.GradientDescent

      type AdaGrad = ops.optimizers.AdaGrad
      val AdaGrad = ops.optimizers.AdaGrad
    }

    //endregion Op Construction Aliases
  }

  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = tensors.RowMajorOrder

  //region Op Creation

  private[api] val COLOCATION_OPS_ATTRIBUTE_NAME   = "_class"
  private[api] val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  private[api] val VALID_OP_NAME_REGEX   : Regex   = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[api] val VALID_NAME_SCOPE_REGEX: Regex   = "^[A-Za-z0-9_.\\-/]*$".r

  private[api] val META_GRAPH_UNBOUND_INPUT_PREFIX: String = "$unbound_inputs_"

  import org.platanios.tensorflow.api.ops.OpCreationContext

  implicit val opCreationContext: DynamicVariable[OpCreationContext] = {
    new DynamicVariable[OpCreationContext](OpCreationContext(graph = defaultGraph))
  }

  implicit def dynamicVariableToOpCreationContext(context: DynamicVariable[OpCreationContext]): OpCreationContext = {
    context.value
  }

  //endregion Op Creation

  //region Utilities

  trait Closeable {
    def close(): Unit
  }

  def using[T <: Closeable, R](resource: T)(block: T => R): R = {
    try {
      block(resource)
    } finally {
      if (resource != null)
        resource.close()
    }
  }

  private[api] val Disposer = utilities.Disposer

  type ProtoSerializable = utilities.Proto.Serializable

  //endregion Utilities
}
