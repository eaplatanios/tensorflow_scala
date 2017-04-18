package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends Implicits {
  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = Tensor.RowMajorOrder

  //region Data Types

  type SupportedScalaType = types.SupportedScalaType
  type DataType = types.DataType

  val DataType = types.DataType

  type Bool = types.Bool
  // type Float16 = types.Float16
  type Float32 = types.Float32
  type Float64 = types.Float64
  // type BFloat16 = types.BFloat16
  // type Complex64 = types.Complex64
  // type Complex128 = types.Complex128
  type Int8 = types.Int8
  type Int16 = types.Int16
  type Int32 = types.Int32
  type Int64 = types.Int64
  type UInt8 = types.UInt8
  type UInt16 = types.UInt16
  // type String = types.String
  // type Resource = types.Resource

  val Bool    = types.Bool
  // val Float16 = types.Float16
  val Float32 = types.Float32
  val Float64 = types.Float64
  // val BFloat16 = types.BFloat16
  // val Complex64 = types.Complex64
  // val Complex128 = types.Complex128
  val Int8    = types.Int8
  val Int16   = types.Int16
  val Int32   = types.Int32
  val Int64   = types.Int64
  val UInt8   = types.UInt8
  val UInt16  = types.UInt16
  // val String = types.String
  // val Resource = types.Resource

  //endregion Data Types

  //region Indexer

  val --- : Indexer = Indexer.---
  val ::  : Slice   = Slice.::

  //endregion Indexer

  //region Op Creation

  type Op = ops.Op
  val Op = ops.Op

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
}
