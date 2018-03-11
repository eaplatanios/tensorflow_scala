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

package org.platanios.tensorflow.api.implicits.helpers

import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike}

/** Type trait used to map structures of symbolic tensors (i.e., outputs) to structures of tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait OutputToTensor[O] {
  type TensorType
}

object OutputToTensor {
  type Aux[O, T] = OutputToTensor[O] {
    type TensorType = T
  }

  implicit val outputToTensor: Aux[Output, Tensor] = new OutputToTensor[Output] {
    override type TensorType = Tensor
  }

  implicit val outputIndexedSlicesToTensorIndexedSlices: Aux[OutputIndexedSlices, TensorIndexedSlices] = {
    new OutputToTensor[OutputIndexedSlices] {
      override type TensorType = TensorIndexedSlices
    }
  }

  implicit val sparseOutputToSparseTensor: Aux[SparseOutput, SparseTensor] = new OutputToTensor[SparseOutput] {
    override type TensorType = SparseTensor
  }

  implicit def arrayOutputToTensor[O, T](implicit ev: Aux[O, T]): Aux[Array[O], Array[T]] = {
    new OutputToTensor[Array[O]] {
      override type TensorType = Array[T]
    }
  }

  implicit def seqOutputToTensor[O, T, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[O, T]): Aux[CC[O], CC[T]] = {
    new OutputToTensor[CC[O]] {
      override type TensorType = CC[T]
    }
  }

  implicit def mapOutputToTensor[K, O, T, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[O, T]
  ): Aux[CC[K, O], CC[K, T]] = new OutputToTensor[CC[K, O]] {
    override type TensorType = CC[K, T]
  }

  implicit val hnilOutputToTensor: Aux[HNil, HNil] = new OutputToTensor[HNil] {
    override type TensorType = HNil
  }

  implicit def recursiveOutputToTensorConstructor[HO, HT, TO <: HList, TT <: HList](implicit
      evHead: Lazy[Aux[HO, HT]],
      evTail: Aux[TO, TT]
  ): Aux[HO :: TO, HT :: TT] = new OutputToTensor[HO :: TO] {
    override type TensorType = HT :: TT
  }

  implicit def productOutputToTensorConstructor[PO <: Product, PT <: Product, HO <: HList, HT <: HList](implicit
      genO: Generic.Aux[PO, HO],
      evH: Aux[HO, HT],
      tuplerT: Tupler.Aux[HT, PT]
  ): Aux[PO, PT] = new OutputToTensor[PO] {
    override type TensorType = PT
  }
}
