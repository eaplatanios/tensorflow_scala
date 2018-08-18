/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors.ops.Basic.stack
import org.platanios.tensorflow.api.types.{DataType, INT32, INT64, SupportedType}

import scala.collection.{TraversableLike, breakOut}

/** Type trait representing types that can be converted to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorConvertible[T] {
  type D <: DataType

  // TODO: Add data type argument.
  /** Converts `value` to a dense tensor. */
  @inline def toTensor(value: T): Tensor[D]
}

object TensorConvertible {
  type Aux[T, DD <: DataType] = TensorConvertible[T] {
    type D = DD
  }

  implicit def fromTensorLike[DR <: DataType, TL[DD <: DataType] <: TensorLike[DD]]: TensorConvertible.Aux[TL[DR], DR] = {
    new TensorConvertible[TL[DR]] {
      override type D = DR

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: TL[DR]): Tensor[DR] = value.toTensor
    }
  }

  implicit val fromShape: TensorConvertible.Aux[Shape, INT32] = {
    new TensorConvertible[Shape] {
      override type D = INT32

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Shape): Tensor[INT32] = value.toTensor
    }
  }

  implicit val fromRange: TensorConvertible.Aux[Range, INT32] = {
    new TensorConvertible[Range] {
      override type D = INT32

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Range): Tensor[INT32] = stack(value.map(_.toTensor))
    }
  }

  implicit def fromSupportedType[T, DD <: DataType](implicit
      evSupported: SupportedType.Aux[T, DD]
  ): TensorConvertible.Aux[T, DD] = {
    new TensorConvertible[T] {
      override type D = DD

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: T): Tensor[DD] = Tensor.fill(evSupported.dataType, Shape())(value)
    }
  }

  implicit def fromArray[T, DD <: DataType](implicit
      ev: TensorConvertible.Aux[T, DD]
  ): TensorConvertible.Aux[Array[T], DD] = {
    new TensorConvertible[Array[T]] {
      override type D = DD

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Array[T]): Tensor[D] = stack(value.map(ev.toTensor))
    }
  }

  implicit def fromTraversable[T, DD <: DataType, CC[A] <: TraversableLike[A, CC[A]]](implicit
      ev: TensorConvertible.Aux[T, DD]
  ): TensorConvertible.Aux[CC[T], DD] = {
    new TensorConvertible[CC[T]] {
      override type D = DD

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: CC[T]): Tensor[D] = stack(value.map(ev.toTensor)(breakOut))
    }
  }
}
