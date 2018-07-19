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

import org.platanios.tensorflow.api.types.DataType

/** Type trait for defining functions operating on and returning tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorOps[TL[DD <: DataType] <: TensorLike[DD]] {
  type D <: DataType

  /** Applies a unary function to the provided tensor and returns the result.
    *
    * @param  tensorLike Tensor-like object to apply the unary op function on.
    * @param  function   Unary function to apply.
    * @return Resulting tensor-like object that matches the type of `tensorLike`.
    */
  @inline def applyUnary[DR <: DataType](tensorLike: TL[D], function: Tensor[D] => Tensor[DR]): TL[DR]
}

/** Companion object that defines supported [[TensorOps]] implicit values. */
object TensorOps {
  type Aux[TL[DDD <: DataType] <: TensorLike[DDD], DD <: DataType] = TensorOps[TL] {
    type D = DD
  }

  implicit def tensorOps[DD <: DataType]: TensorOps.Aux[Tensor, DD] = {
    new TensorOps[Tensor] {
      override type D = DD
      @inline override def applyUnary[DR <: DataType](
          tensorLike: Tensor[D],
          function: Tensor[D] => Tensor[DR]
      ): Tensor[DR] = {
        function(tensorLike)
      }
    }
  }

  implicit def tensorIndexedSlicesOps[DD <: DataType]: TensorOps.Aux[TensorIndexedSlices, DD] = {
    new TensorOps[TensorIndexedSlices] {
      override type D = DD
      @inline override def applyUnary[DR <: DataType](
          tensorLike: TensorIndexedSlices[D],
          function: Tensor[D] => Tensor[DR]
      ): TensorIndexedSlices[DR] = {
        tensorLike.copy(values = function(tensorLike.values))
      }
    }
  }

  implicit def sparseTensorOps[DD <: DataType]: TensorOps.Aux[SparseTensor, DD] = {
    new TensorOps[SparseTensor] {
      override type D = DD
      @inline override def applyUnary[DR <: DataType](
          tensorLike: SparseTensor[D],
          function: Tensor[D] => Tensor[DR]
      ): SparseTensor[DR] = {
        tensorLike.copy(values = function(tensorLike.values))
      }
    }
  }

  implicit def tensorLikeOps[DD <: DataType]: TensorOps.Aux[TensorLike, DD] = {
    new TensorOps[TensorLike] {
      override type D = DD
      @inline override def applyUnary[DR <: DataType](
          tensorLike: TensorLike[D],
          function: Tensor[D] => Tensor[DR]
      ): TensorLike[DR] = {
        tensorLike match {
          case t: Tensor[D] => function(t)
          case t: TensorIndexedSlices[D] => t.copy(values = function(t.values))
          case t: SparseTensor[D] => t.copy(values = function(t.values))
        }
      }
    }
  }
}
