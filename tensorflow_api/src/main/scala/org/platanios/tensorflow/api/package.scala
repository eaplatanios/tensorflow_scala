package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api {
  val DataType = org.platanios.tensorflow.jni.DataType
  type DataType = org.platanios.tensorflow.jni.DataType
  type Op = ops.Op
  val Op = ops.Op

  private[api] val COLOCATION_OPS_ATTRIBUTE_NAME = "_class"
  private[api] val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  private[api] val VALID_OP_NAME_REGEX: Regex = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[api] val VALID_NAME_SCOPE_REGEX: Regex = "^[A-Za-z0-9_.\\-/]*$".r

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

  //region Op Creation Implicits

  import org.platanios.tensorflow.api.ops.OpCreationContext

  val defaultGraph: Graph = Graph()
  implicit val opCreationContext: DynamicVariable[OpCreationContext] =
    new DynamicVariable[OpCreationContext](OpCreationContext(graph = defaultGraph))

  implicit def dynamicVariableToOpCreationContext(context: DynamicVariable[OpCreationContext]): OpCreationContext =
    context.value

  //endregion

  //region Slice Implicits

  def :: : Slice = Slice.::
  implicit def intToSlice(int: Int): Slice = Slice.intToSlice(int)
  implicit def longToSlice(long: Long): Slice = Slice.longToSlice(long)
  implicit def intToSliceWithOneNumber(int: Int): SliceWithOneNumber = Slice.intToSliceWithOneNumber(int)
  implicit def longToSliceWithOneNumber(long: Long): SliceWithOneNumber = Slice.longToSliceWithOneNumber(long)
  implicit def sliceConstructionToSlice(sliceConstruction: SliceConstruction): Slice = {
    Slice.sliceConstructionToSlice(sliceConstruction)
  }

  //endregion
}
