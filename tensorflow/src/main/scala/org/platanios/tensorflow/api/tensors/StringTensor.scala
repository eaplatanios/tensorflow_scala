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

  override def reshape(shape: Shape, copyData: Boolean = true): StringTensor = ???

  override def asNumeric: NumericTensor = {
    throw InvalidDataTypeException(s"Data type '$dataType' of this tensor is not numeric.")
  }

  override def asRealNumeric: RealNumericTensor = {
    throw InvalidDataTypeException(s"Data type '$dataType' of this tensor is not real numeric.")
  }
}
