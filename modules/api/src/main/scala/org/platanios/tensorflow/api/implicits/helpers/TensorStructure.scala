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

/** Type trait used to represent nested structures over tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorStructure[V] {
  def tensors(tensor: V): Seq[Tensor[Any]]
}

object TensorStructure {
  def apply[V](implicit ev: TensorStructure[V]): TensorStructure[V] = {
    ev
  }

  implicit val fromUnit: TensorStructure[Unit] = {
    new TensorStructure[Unit] {
      override def tensors(tensor: Unit): Seq[Tensor[Any]] = {
        Seq.empty
      }
    }
  }

  implicit def fromTensor[T]: TensorStructure[Tensor[T]] = {
    new TensorStructure[Tensor[T]] {
      override def tensors(tensor: Tensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor.asUntyped)
      }
    }
  }

  implicit def fromTensorIndexedSlices[T]: TensorStructure[TensorIndexedSlices[T]] = {
    new TensorStructure[TensorIndexedSlices[T]] {
      override def tensors(tensor: TensorIndexedSlices[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices.asUntyped, tensor.values.asUntyped, tensor.denseShape.asUntyped)
      }
    }
  }

  implicit def fromSparseTensor[T]: TensorStructure[SparseTensor[T]] = {
    new TensorStructure[SparseTensor[T]] {
      override def tensors(tensor: SparseTensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices.asUntyped, tensor.values.asUntyped, tensor.denseShape.asUntyped)
      }
    }
  }

  implicit def fromOption[T](implicit ev: TensorStructure[T]): TensorStructure[Option[T]] = {
    new TensorStructure[Option[T]] {
      override def tensors(tensor: Option[T]): Seq[Tensor[Any]] = {
        tensor.toSeq.flatMap(ev.tensors)
      }
    }
  }

  implicit def fromSeq[T](implicit ev: TensorStructure[T]): TensorStructure[Seq[T]] = {
    new TensorStructure[Seq[T]] {
      override def tensors(tensor: Seq[T]): Seq[Tensor[Any]] = {
        tensor.flatMap(ev.tensors)
      }
    }
  }

  implicit def fromMap[K, D](implicit ev: TensorStructure[D]): TensorStructure[Map[K, D]] = {
    new TensorStructure[Map[K, D]] {
      override def tensors(tensor: Map[K, D]): Seq[Tensor[Any]] = {
        tensor.values.flatMap(ev.tensors).toSeq
      }
    }
  }

  implicit val fromHNil: TensorStructure[HNil] = {
    new TensorStructure[HNil] {
      override def tensors(tensor: HNil): Seq[Tensor[Any]] = {
        Seq.empty
      }
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
    }
  }

  implicit def fromProduct[PT <: Product, HT <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[TensorStructure[HT]]
  ): TensorStructure[PT] = {
    new TensorStructure[PT] {
      override def tensors(tensor: PT): Seq[Tensor[Any]] = {
        evT.value.tensors(genT.to(tensor))
      }
    }
  }
}
