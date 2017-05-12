package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends Implicits {
  object tf {
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

    type Graph = api.Graph
    val Graph = api.Graph

    val defaultGraph: Graph = Graph()

    type Session = api.Session
    val Session = api.Session

    type Op = ops.Op
    val Op = ops.Op

    //region Op Construction Aliases

    def createWith[R](
        graph: Graph = null, nameScope: String = null, device: ops.OpSpecification => String = _ => "",
        colocationOps: Set[Op] = null, controlDependencies: Set[Op] = null, attributes: Map[String, Any] = null,
        container: String = null)(block: => R): R = {
      ops.Op.createWith(graph, nameScope, device, colocationOps, controlDependencies, attributes, container)(block)
    }

    def createWithNameScope[R](nameScope: String, values: Set[Op] = Set.empty[Op])(block: => R): R = {
      ops.Op.createWithNameScope(nameScope, values)(block)
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

    type Variable = ops.Variable
    val Variable = ops.Variable

    val zerosInitializer = ops.Variable.ZerosInitializer
    val onesInitializer = ops.Variable.OnesInitializer

    def constantInitializer(value: Tensor) = ops.Variable.ConstantInitializer(value)

    //endregion Variables

    val Gradients         = ops.Gradients
    val GradientsRegistry = ops.Gradients.Registry

    object train {
      type Optimizer = ops.optimizers.Optimizer
      val Optimizer = ops.optimizers.Optimizer

      type GradientDescent = ops.optimizers.GradientDescent
      val GradientDescent = ops.optimizers.GradientDescent
    }

    //endregion Op Construction Aliases
  }

  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = tensors.RowMajorOrder

  //region Op Creation

  private[api] val COLOCATION_OPS_ATTRIBUTE_NAME   = "_class"
  private[api] val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  private[api] val VALID_OP_NAME_REGEX   : Regex   = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[api] val VALID_NAME_SCOPE_REGEX: Regex   = "^[A-Za-z0-9_.\\-/]*$".r

  import org.platanios.tensorflow.api.ops.OpCreationContext

  implicit val opCreationContext: DynamicVariable[OpCreationContext] = {
    new DynamicVariable[OpCreationContext](OpCreationContext(graph = tf.defaultGraph))
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

  //endregion Utilities
}
