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
import org.platanios.tensorflow.api.tensors.ops.Basic
import org.platanios.tensorflow.api.types.SupportedType

import scala.collection.{TraversableLike, breakOut}

/** Type trait representing types that can be converted to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorConvertible[TC] {
  type T

  // TODO: [TENSORS] Add data type argument.
  /** Converts `value` to a dense tensor. */
  @inline def toTensor(value: TC): Tensor[T]
}

object TensorConvertible {
  type Aux[TC, TT] = TensorConvertible[TC] {
    type T = TT
  }

  implicit def fromTensorLike[TT, TL[A] <: TensorLike[A]]: TensorConvertible.Aux[TL[TT], TT] = {
    new TensorConvertible[TL[TT]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: TL[TT]): Tensor[TT] = {
        value.toTensor
      }
    }
  }

  implicit val fromShape: TensorConvertible.Aux[Shape, Long] = {
    new TensorConvertible[Shape] {
      override type T = Long

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Shape): Tensor[Long] = {
        value.toTensor
      }
    }
  }

  implicit val fromRange: TensorConvertible.Aux[Range, Int] = {
    new TensorConvertible[Range] {
      override type T = Int

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Range): Tensor[Int] = {
        Basic.stack(value.map(_.toTensor))
      }
    }
  }

  implicit def fromSupportedType[TT](implicit
      evSupported: SupportedType[TT]
  ): TensorConvertible.Aux[TT, TT] = {
    new TensorConvertible[TT] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: TT): Tensor[TT] = {
        Tensor.fill(evSupported.dataType, Shape())(value)
      }
    }
  }

  implicit def fromArray[TC, TT](implicit
      ev: TensorConvertible.Aux[TC, TT]
  ): TensorConvertible.Aux[Array[TC], TT] = {
    new TensorConvertible[Array[TC]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: Array[TC]): Tensor[TT] = {
        Basic.stack(value.map(ev.toTensor))
      }
    }
  }

  implicit def fromTraversable[TC, TT, CC[A] <: TraversableLike[A, CC[A]]](implicit
      ev: TensorConvertible.Aux[TC, TT]
  ): TensorConvertible.Aux[CC[TC], TT] = {
    new TensorConvertible[CC[TC]] {
      override type T = TT

      /** Converts `value` to a dense tensor. */
      @inline override def toTensor(value: CC[TC]): Tensor[TT] = {
        Basic.stack(value.map(ev.toTensor)(breakOut))
      }
    }
  }
}
