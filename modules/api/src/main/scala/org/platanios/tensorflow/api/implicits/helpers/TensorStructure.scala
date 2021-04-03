/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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
import shapeless._

import scala.collection.compat._

/** Type trait used to represent nested structures over tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorStructure[V] {
  def tensors(tensor: V): Seq[Tensor[Any]]
  def map(value: V, converter: TensorStructure.Converter): V
}

object TensorStructure {
  trait Converter {
    def apply[T](value: Tensor[T]): Tensor[T] = value
    def apply[T](value: TensorIndexedSlices[T]): TensorIndexedSlices[T] = value
    def apply[T](value: SparseTensor[T]): SparseTensor[T] = value
  }

  def apply[V](implicit ev: TensorStructure[V]): TensorStructure[V] = {
    ev
  }

  implicit val fromUnit: TensorStructure[Unit] = {
    new TensorStructure[Unit] {
      override def tensors(tensor: Unit): Seq[Tensor[Any]] = Seq.empty
      override def map(value: Unit, converter: Converter): Unit = ()
    }
  }

  implicit def fromTensor[T]: TensorStructure[Tensor[T]] = {
    new TensorStructure[Tensor[T]] {
      override def tensors(tensor: Tensor[T]): Seq[Tensor[Any]] = Seq(tensor.asUntyped)
      override def map(value: Tensor[T], converter: Converter): Tensor[T] = converter(value)
    }
  }

  implicit def fromTensorIndexedSlices[T]: TensorStructure[TensorIndexedSlices[T]] = {
    new TensorStructure[TensorIndexedSlices[T]] {
      override def tensors(tensor: TensorIndexedSlices[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices.asUntyped, tensor.values.asUntyped, tensor.denseShape.asUntyped)
      }

      override def map(value: TensorIndexedSlices[T], converter: Converter): TensorIndexedSlices[T] = converter(value)
    }
  }

  implicit def fromSparseTensor[T]: TensorStructure[SparseTensor[T]] = {
    new TensorStructure[SparseTensor[T]] {
      override def tensors(tensor: SparseTensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices.asUntyped, tensor.values.asUntyped, tensor.denseShape.asUntyped)
      }

      override def map(value: SparseTensor[T], converter: Converter): SparseTensor[T] = converter(value)
    }
  }

  implicit def fromOption[T](implicit ev: TensorStructure[T]): TensorStructure[Option[T]] = {
    new TensorStructure[Option[T]] {
      override def tensors(tensor: Option[T]): Seq[Tensor[Any]] = {
        tensor.toSeq.flatMap(ev.tensors)
      }

      override def map(value: Option[T], converter: Converter): Option[T] = value.map(ev.map(_, converter))
    }
  }

  implicit def fromSeq[T](implicit ev: TensorStructure[T]): TensorStructure[Seq[T]] = {
    new TensorStructure[Seq[T]] {
      override def tensors(tensor: Seq[T]): Seq[Tensor[Any]] = {
        tensor.flatMap(ev.tensors)
      }

      override def map(value: Seq[T], converter: Converter): Seq[T] = value.map(ev.map(_, converter))
    }
  }

  implicit def fromMap[K, D](implicit ev: TensorStructure[D]): TensorStructure[Map[K, D]] = {
    new TensorStructure[Map[K, D]] {
      override def tensors(tensor: Map[K, D]): Seq[Tensor[Any]] = {
        tensor.values.flatMap(ev.tensors).toSeq
      }

      override def map(value: Map[K, D], converter: Converter): Map[K, D] = {
        value.view.mapValues(ev.map(_, converter)).toMap
      }
    }
  }

  implicit val fromHNil: TensorStructure[HNil] = {
    new TensorStructure[HNil] {
      override def tensors(tensor: HNil): Seq[Tensor[Any]] = Seq.empty
      override def map(value: HNil, converter: Converter): HNil = HNil
    }
  }

  implicit def fromHList[HT, TT <: HList](implicit
      evH: Strict[TensorStructure[HT]],
      evT: Strict[TensorStructure[TT]]
  ): TensorStructure[HT :: TT] = {
    new TensorStructure[HT :: TT] {
      override def tensors(tensor: HT :: TT): Seq[Tensor[Any]] = {
        evH.value.tensors(tensor.head) ++ evT.value.tensors(tensor.tail)
      }

      override def map(value: HT :: TT, converter: Converter): HT :: TT = {
        evH.value.map(value.head, converter) ::
            evT.value.map(value.tail, converter)
      }
    }
  }

  implicit def fromProduct[PT, HT <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[TensorStructure[HT]]
  ): TensorStructure[PT] = {
    new TensorStructure[PT] {
      override def tensors(tensor: PT): Seq[Tensor[Any]] = {
        evT.value.tensors(genT.to(tensor))
      }

      override def map(value: PT, converter: Converter): PT = {
        genT.from(evT.value.map(genT.to(value), converter))
      }
    }
  }
}
