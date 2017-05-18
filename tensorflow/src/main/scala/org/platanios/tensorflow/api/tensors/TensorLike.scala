package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
trait TensorLike extends TensorConvertible {
  val dataType: DataType
  val shape   : Shape

  require(shape.isFullyDefined, s"The shape of a Tensor object must be fully defined. Shape '$shape' is not.")
  require(shape.numElements.get > 0, "Empty tensors are not supported in the TensorFlow Scala API.")

  def rank: Int = shape.rank
  def numElements: Int = shape.numElements.get

  def summarize(maxEntries: Int = numElements): String
}
