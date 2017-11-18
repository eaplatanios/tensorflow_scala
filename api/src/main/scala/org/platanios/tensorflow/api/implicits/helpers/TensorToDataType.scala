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

package org.platanios.tensorflow.api.implicits.helpers

import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.types.DataType

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike}

/** Type trait used to map structures of tensors to structures of data types.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorToDataType[T] {
  type DataTypeType
}

object TensorToDataType {
  type Aux[T, D] = TensorToDataType[T] {
    type DataTypeType = D
  }

  implicit val tensorToDataType: Aux[Tensor, DataType] = new TensorToDataType[Tensor] {
    override type DataTypeType = DataType
  }

  implicit val tensorIndexedSlicesToDataType: Aux[TensorIndexedSlices, (DataType, DataType, DataType)] = {
    new TensorToDataType[TensorIndexedSlices] {
      override type DataTypeType = (DataType, DataType, DataType)
    }
  }

  implicit val sparseTensorToDataType: Aux[SparseTensor, (DataType, DataType, DataType)] = {
    new TensorToDataType[SparseTensor] {
      override type DataTypeType = (DataType, DataType, DataType)
    }
  }

  implicit def arrayTensorToDataType[T, D](implicit ev: Aux[T, D]): Aux[Array[T], Array[D]] = {
    new TensorToDataType[Array[T]] {
      override type DataTypeType = Array[D]
    }
  }

  implicit def seqTensorToDataType[T, D, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[T, D]): Aux[CC[T], CC[D]] = {
    new TensorToDataType[CC[T]] {
      override type DataTypeType = CC[D]
    }
  }

  implicit def mapTensorToDataType[K, T, D, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[T, D]
  ): Aux[CC[K, T], CC[K, D]] = new TensorToDataType[CC[K, T]] {
    override type DataTypeType = CC[K, D]
  }

  implicit val hnilTensorToDataType: Aux[HNil, HNil] = new TensorToDataType[HNil] {
    override type DataTypeType = HNil
  }

  implicit def recursiveTensorToDataTypeConstructor[HT, HD, TT <: HList, TD <: HList](implicit
      evHead: Lazy[Aux[HT, HD]],
      evTail: Aux[TT, TD]
  ): Aux[HT :: TT, HD :: TD] = new TensorToDataType[HT :: TT] {
    override type DataTypeType = HD :: TD
  }

  implicit def productTensorToDataTypeConstructor[PT <: Product, PD <: Product, HT <: HList, HD <: HList](implicit
      genO: Generic.Aux[PT, HT],
      evH: Aux[HT, HD],
      tuplerT: Tupler.Aux[HD, PD]
  ): Aux[PT, PD] = new TensorToDataType[PT] {
    override type DataTypeType = PD
  }
}
