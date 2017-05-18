package org.platanios.tensorflow.api.tensors

/** Helper trait for tagging tensor convertible objects so that implicit conversions to tensors can be used.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorConvertible {
  /** Returns the [[Tensor]] that this [[TensorConvertible]] object represents. */
  def toTensor: Tensor
}
