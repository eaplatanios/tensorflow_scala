package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api._

import java.nio.ByteBuffer

import spire.implicits._

/**
  * @author Emmanouil Antonios Platanios
  */
class NumericTensor private[tensors] (
    override val dataType: NumericDataType, override val shape: Shape, override val buffer: ByteBuffer,
    override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    extends FixedSizeTensor(dataType, shape, buffer, order) {
  override def asNumeric: NumericTensor = this
}

class RealNumericTensor private[tensors] (
    override val dataType: RealNumericDataType, override val shape: Shape, override val buffer: ByteBuffer,
    override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    extends NumericTensor(dataType, shape, buffer, order) {
  def +-(tolerance: Double): RealNumericTensorEquality = {
    RealNumericTensorEquality(this, tolerance)
  }

  def ===(that: RealNumericTensorEquality): Boolean = {
    if (this.shape != that.tensor.shape) {
      false
    } else if (this.dataType != that.tensor.dataType) {
      false // TODO: Do we want this?
    } else {
      this.entriesIterator.zip(that.tensor.entriesIterator).forall(p => {
        // TODO: This is very ugly.
        import this.dataType.supportedType
        val v1 = p._1
        val v2 = p._2.asInstanceOf[this.dataType.ScalaType]
        val tol = this.dataType.cast(that.tolerance)
        v1 <= (v2 + tol) && v1 >= (v2 - tol)
      })
    }
  }

  def =!=(that: RealNumericTensorEquality): Boolean = !(this === that)

  override def asNumeric: NumericTensor = this
  override def asRealNumeric: RealNumericTensor = this
}

case class RealNumericTensorEquality(tensor: RealNumericTensor, tolerance: Double)
