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

import org.platanios.tensorflow.api.core.types.{DataType, Variant, FLOAT32, VARIANT}
import org.platanios.tensorflow.api.ops.TensorArray
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorToDataType[T] {
  type D

  def dataType(tensor: T): D
}

object TensorToDataType {
  type Aux[T, DD] = TensorToDataType[T] {
    type D = DD
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new TensorToDataType[Unit] {
      override type D = Unit

      override def dataType(output: Unit): Unit = {
        ()
      }
    }
  }

  implicit def fromTensor[T]: Aux[Tensor[T], DataType[T]] = {
    new TensorToDataType[Tensor[T]] {
      override type D = DataType[T]

      override def dataType(output: Tensor[T]): DataType[T] = {
        output.dataType
      }
    }
  }

  implicit def fromTensorIndexedSlices[T]: Aux[TensorIndexedSlices[T], SparseDataType[T]] = {
    new TensorToDataType[TensorIndexedSlices[T]] {
      override type D = SparseDataType[T]

      override def dataType(output: TensorIndexedSlices[T]): SparseDataType[T] = {
        (output.indices.dataType, output.values.dataType, output.denseShape.dataType)
      }
    }
  }

  implicit def fromSparseTensor[T]: Aux[SparseTensor[T], SparseDataType[T]] = {
    new TensorToDataType[SparseTensor[T]] {
      override type D = SparseDataType[T]

      override def dataType(output: SparseTensor[T]): SparseDataType[T] = {
        (output.indices.dataType, output.values.dataType, output.denseShape.dataType)
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], DataType[Float]] = {
    new TensorToDataType[TensorArray[T]] {
      override type D = DataType[Float]

      override def dataType(output: TensorArray[T]): DataType[Float] = {
        FLOAT32
      }
    }
  }

  implicit def fromDataset[T]: Aux[Dataset[T], DataType[Variant]] = {
    new TensorToDataType[Dataset[T]] {
      override type D = DataType[Variant]

      override def dataType(output: Dataset[T]): DataType[Variant] = {
        VARIANT
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: TensorToDataType[T]
  ): TensorToDataType.Aux[Option[T], Option[ev.D]] = {
    new TensorToDataType[Option[T]] {
      override type D = Option[ev.D]

      override def dataType(output: Option[T]): Option[ev.D] = {
        output.map(o => ev.dataType(o))
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: TensorToDataType[T]
  ): TensorToDataType.Aux[Seq[T], Seq[ev.D]] = {
    new TensorToDataType[Seq[T]] {
      override type D = Seq[ev.D]

      override def dataType(output: Seq[T]): Seq[ev.D] = {
        output.map(o => ev.dataType(o))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: TensorToDataType[T]
  ): TensorToDataType.Aux[Map[K, T], Map[K, ev.D]] = {
    new TensorToDataType[Map[K, T]] {
      override type D = Map[K, ev.D]

      override def dataType(output: Map[K, T]): Map[K, ev.D] = {
        output.mapValues(o => ev.dataType(o))
      }
    }
  }

  implicit val fromHNil: TensorToDataType.Aux[HNil, HNil] = {
    new TensorToDataType[HNil] {
      override type D = HNil

      override def dataType(output: HNil): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HD, TT <: HList, TD <: HList](implicit
      evH: Strict[TensorToDataType.Aux[HT, HD]],
      evT: Strict[TensorToDataType.Aux[TT, TD]]
  ): TensorToDataType.Aux[HT :: TT, HD :: TD] = {
    new TensorToDataType[HT :: TT] {
      override type D = HD :: TD

      override def dataType(output: HT :: TT): HD :: TD = {
        evH.value.dataType(output.head) :: evT.value.dataType(output.tail)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PD <: Product, HT <: HList, HD <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[TensorToDataType.Aux[HT, HD]],
      tuplerS: Tupler.Aux[HD, PD],
      genS: Generic.Aux[PD, HD]
  ): TensorToDataType.Aux[PT, PD] = {
    new TensorToDataType[PT] {
      override type D = PD

      override def dataType(output: PT): PD = {
        tuplerS(evT.value.dataType(genT.to(output)))
      }
    }
  }
}
