package org.platanios.tensorflow

import org.platanios.tensorflow.api.ops.OpCreationContext

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
package object api {
  val DataType = org.platanios.tensorflow.jni.DataType
  type DataType[T] = org.platanios.tensorflow.jni.DataType[T]

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

  // Graph Creation Helpers

  implicit val opCreationContext: DynamicVariable[OpCreationContext] =
    new DynamicVariable[OpCreationContext](OpCreationContext())

  implicit def dynamicVariableToOpCreationContext(context: DynamicVariable[OpCreationContext]): OpCreationContext =
    context.value

  def createUsing[R](graph: Graph)(block: => R)(implicit context: DynamicVariable[OpCreationContext]): R =
    context.withValue(context.copy(graph = graph))(block)

  def createUsing[R](nameScope: String)(block: => R)(implicit context: DynamicVariable[OpCreationContext]): R =
    context.withValue(context.copy(nameScope = s"${context.nameScope}/$nameScope"))(block)
}
