// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.types.{NumericDataType, RealNumericDataType}

import java.nio.{ByteBuffer, ByteOrder}

import spire.implicits._

/**
  * @author Emmanouil Antonios Platanios
  */
class NumericTensor private[tensors] (
    override val dataType: NumericDataType, override val shape: Shape, override val buffer: ByteBuffer,
    override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    extends FixedSizeTensor(dataType, shape, buffer, order) {
  override private[tensors] def newTensor(shape: Shape): Tensor = NumericTensor.allocate(dataType, shape, order)

  override def asNumeric: NumericTensor = this
}

object NumericTensor {
  private[api] def allocate(
      dataType: NumericDataType, shape: Shape, order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): NumericTensor = {
    val numBytes: Int = dataType.byteSize * shape.numElements.get
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
    new NumericTensor(dataType = dataType, shape = shape, buffer = buffer, order = order)
  }
}

class RealNumericTensor private[tensors] (
    override val dataType: RealNumericDataType, override val shape: Shape, override val buffer: ByteBuffer,
    override val order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER)
    extends NumericTensor(dataType, shape, buffer, order) {
  override private[tensors] def newTensor(shape: Shape): Tensor = RealNumericTensor.allocate(dataType, shape, order)

  def +-(tolerance: Double): RealNumericTensor.Equality = {
    RealNumericTensor.Equality(this, tolerance)
  }

  override def equals(that: Any): Boolean = that match {
    case that: RealNumericTensor.Equality =>
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
    case that: Tensor => super.equals(that)
    case _ => false
  }

  override def asNumeric: NumericTensor = this
  override def asRealNumeric: RealNumericTensor = this
}

object RealNumericTensor {
  private[api] def allocate(
      dataType: RealNumericDataType, shape: Shape,
      order: Order = DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER): RealNumericTensor = {
    val numBytes: Int = dataType.byteSize * shape.numElements.get
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(numBytes).order(ByteOrder.nativeOrder)
    new RealNumericTensor(dataType = dataType, shape = shape, buffer = buffer, order = order)
  }

  case class Equality(tensor: RealNumericTensor, tolerance: Double)
}
