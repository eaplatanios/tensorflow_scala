package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends Implicits {
  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = Tensor.RowMajorOrder

  type SupportedType[T] = types.SupportedType[T]

  type DataType = types.DataType
  val DataType = types.DataType

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
