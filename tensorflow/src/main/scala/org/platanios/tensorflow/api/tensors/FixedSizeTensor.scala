/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.types._

import java.nio.{ByteBuffer, ByteOrder}

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
