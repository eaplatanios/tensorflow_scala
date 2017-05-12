package org.platanios.tensorflow.api.tensors

import java.nio.{ByteBuffer, ByteOrder}

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tf.{FixedSizeDataType, NumericDataType, RealNumericDataType, SupportedType}
import org.platanios.tensorflow.api.Exception.InvalidDataTypeException

/**
  * @author Emmanouil Antonios Platanios
  */
class FixedSizeTensor private[tensors] (
    override val dataType: FixedSizeDataType, override val shape: Shape, override val buffer: ByteBuffer,
    override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    extends Tensor {
  override private[api] def setElementAtFlattenedIndex[T](
      index: Int, value: T)(implicit evidence: SupportedType[T]): this.type = {
    dataType.putElementInBuffer(buffer, index * dataType.byteSize, dataType.cast(value))
    this
  }

  override private[api] def getElementAtFlattenedIndex(index: Int): dataType.ScalaType = {
    dataType.getElementFromBuffer(buffer, index * dataType.byteSize)
  }

  override def fill[T](value: T)(implicit evidence: SupportedType[T]): this.type = {
    val castedValue = dataType.cast(value)
    for (index <- flattenedIndexIterator)
      dataType.putElementInBuffer(buffer = buffer, index = index, element = castedValue)
    this
  }

  override private[tensors] def newTensor(shape: Shape): Tensor = FixedSizeTensor.allocate(dataType, shape, order)

  override def asNumeric: NumericTensor = dataType match {
    case d: NumericDataType => new NumericTensor(d, shape, buffer, order)
    case _ => throw InvalidDataTypeException(s"Data type '$dataType' of this tensor is not numeric.")
  }

  override def asRealNumeric: RealNumericTensor = dataType match {
    case d: RealNumericDataType => new RealNumericTensor(d, shape, buffer, order)
    case _ => throw InvalidDataTypeException(s"Data type '$dataType' of this tensor is not real numeric.")
  }
}

object FixedSizeTensor {
  private[api] def allocate(
      dataType: FixedSizeDataType, shape: Shape,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): FixedSizeTensor = {
    val numBytes: Int = dataType.byteSize * shape.numElements.get
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
    new FixedSizeTensor(dataType = dataType, shape = shape, buffer = buffer, order = order)
  }
}
