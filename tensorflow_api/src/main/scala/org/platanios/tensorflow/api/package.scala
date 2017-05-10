package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends Implicits {
  //region Data Types

  type SupportedType[T] = types.SupportedType[T]
  type FixedSizeSupportedType[T] = types.FixedSizeSupportedType[T]
  type NumericSupportedType[T] = types.NumericSupportedType[T]
  type RealNumericSupportedType[T] = types.RealNumericSupportedType[T]
  type ComplexNumericSupportedType[T] = types.ComplexNumericSupportedType[T]

  type DataType = types.DataType
  type FixedSizeDataType = types.FixedSizeDataType
  type NumericDataType = types.NumericDataType
  type RealNumericDataType = types.RealNumericDataType
  type ComplexNumericDataType = types.ComplexNumericDataType

  val DataType = types.DataType

  val TFString = types.TFString
  val TFBoolean = types.TFBoolean
  // val TFFloat16 = types.TFFloat16
  val TFFloat32 = types.TFFloat32
  val TFFloat64 = types.TFFloat64
  // val TFBFloat16 = types.TFBFloat16
  // val TFComplex64 = types.TFComplex64
  // val TFComplex128 = types.TFComplex128
  val TFInt8 = types.TFInt8
  val TFInt16 = types.TFInt16
  val TFInt32 = types.TFInt32
  val TFInt64 = types.TFInt64
  val TFUInt8 = types.TFUInt8
  val TFUInt16 = types.TFUInt16
  val TFQInt8 = types.TFQInt8
  val TFQInt16 = types.TFQInt16
  val TFQInt32 = types.TFQInt32
  val TFQUInt8 = types.TFQUInt8
  val TFQUInt16 = types.TFQUInt16
  val TFResource = types.TFResource

  //endregion Data Types

  //region Tensors

  type Tensor = tensors.Tensor
  type FixedSizeTensor = tensors.FixedSizeTensor
  type NumericTensor = tensors.NumericTensor

  val Tensor = tensors.Tensor

  type Order = tensors.Order
  val RowMajorOrder = tensors.RowMajorOrder
  val ColumnMajorOrder = tensors.ColumnMajorOrder

  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = RowMajorOrder

  //endregion Tensors

  //region Op Creation

  type Op = ops.Op
  val Op = ops.Op

  type Variable = ops.Variable
  val Variable = ops.Variable

  val Gradients = ops.Gradients
  val GradientsRegistry = ops.Gradients.Registry

  private[api] val COLOCATION_OPS_ATTRIBUTE_NAME   = "_class"
  private[api] val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  private[api] val VALID_OP_NAME_REGEX   : Regex   = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[api] val VALID_NAME_SCOPE_REGEX: Regex   = "^[A-Za-z0-9_.\\-/]*$".r

  import org.platanios.tensorflow.api.ops.OpCreationContext

  private[api] val defaultGraph: Graph = Graph()

  private[api] implicit val opCreationContext: DynamicVariable[OpCreationContext] =
    new DynamicVariable[OpCreationContext](OpCreationContext(graph = defaultGraph))
  private[api] implicit def dynamicVariableToOpCreationContext(
      context: DynamicVariable[OpCreationContext]): OpCreationContext = context.value

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
