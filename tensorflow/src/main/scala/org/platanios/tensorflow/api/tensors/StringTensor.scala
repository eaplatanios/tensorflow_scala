package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.types._

import java.nio.ByteBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
class StringTensor private[tensors] (
    override val shape: Shape, override val buffer: ByteBuffer,
    override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    extends Tensor {
  override val dataType: DataType = STRING

  override private[api] def setElementAtFlattenedIndex[T](
      index: Int, value: T)(implicit evidence: SupportedType[T]): this.type = ???

  override private[api] def getElementAtFlattenedIndex(index: Int): dataType.ScalaType = {
    val offset = INT64.byteSize * numElements + INT64.getElementFromBuffer(buffer, index * INT64.byteSize).toInt
    dataType.getElementFromBuffer(buffer, offset)
  }

  override def fill[T](value: T)(implicit evidence: SupportedType[T]): this.type = ???

  override private[tensors] def newTensor(shape: Shape): Tensor = ???

  override def asNumeric: NumericTensor = {
    throw InvalidDataTypeException(s"Data type '$dataType' of this tensor is not numeric.")
  }

  override def asRealNumeric: RealNumericTensor = {
    throw InvalidDataTypeException(s"Data type '$dataType' of this tensor is not real numeric.")
  }
}
