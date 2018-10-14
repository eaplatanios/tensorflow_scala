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

/** Type trait for defining functions operating on and returning tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorOps[TL[A] <: TensorLike[A]] {
  type T

  /** Applies a unary function to the provided tensor and returns the result.
    *
    * @param  value Tensor-like object to apply the unary op function on.
    * @param  fn    Unary function to apply.
    * @return Resulting tensor-like object that matches the type of `value`.
    */
  @inline def applyUnary[R](value: TL[T], fn: Tensor[T] => Tensor[R]): TL[R]
}

/** Companion object that defines supported tensor ops implicit values. */
object TensorOps {
  type Aux[TL[A] <: TensorLike[A], TT] = TensorOps[TL] {
    type T = TT
  }

  implicit def tensorOps[TT]: TensorOps.Aux[Tensor, TT] = {
    new TensorOps[Tensor] {
      override type T = TT

      @inline override def applyUnary[R](
          value: Tensor[T],
          fn: Tensor[T] => Tensor[R]
      ): Tensor[R] = {
        fn(value)
      }
    }
  }

  implicit def tensorIndexedSlicesOps[TT]: TensorOps.Aux[TensorIndexedSlices, TT] = {
    new TensorOps[TensorIndexedSlices] {
      override type T = TT

      @inline override def applyUnary[R](
          value: TensorIndexedSlices[T],
          fn: Tensor[T] => Tensor[R]
      ): TensorIndexedSlices[R] = {
        value.copy(values = fn(value.values))
      }
    }
  }

  implicit def sparseTensorOps[TT]: TensorOps.Aux[SparseTensor, TT] = {
    new TensorOps[SparseTensor] {
      override type T = TT

      @inline override def applyUnary[R](
          value: SparseTensor[T],
          fn: Tensor[T] => Tensor[R]
      ): SparseTensor[R] = {
        value.copy(values = fn(value.values))
      }
    }
  }

  implicit def valueOps[TT]: TensorOps.Aux[TensorLike, TT] = {
    new TensorOps[TensorLike] {
      override type T = TT

      @inline override def applyUnary[R](
          value: TensorLike[T],
          fn: Tensor[T] => Tensor[R]
      ): TensorLike[R] = {
        value match {
          case t: Tensor[T] => fn(t)
          case t: TensorIndexedSlices[T] => t.copy(values = fn(t.values))
          case t: SparseTensor[T] => t.copy(values = fn(t.values))
        }
      }
    }
  }
}
