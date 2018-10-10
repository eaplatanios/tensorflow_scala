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

import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.reflect.ClassTag

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorToOutput[T] {
  type O

  def tensors(tensor: T): Seq[Tensor[Any]]
  def toOutput(tensor: T): O
}

object TensorToOutput {
  type Aux[T, OO] = TensorToOutput[T] {
    type O = OO
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new TensorToOutput[Unit] {
      override type O = Unit

      override def tensors(tensor: Unit): Seq[Tensor[Any]] = {
        Seq.empty
      }

      override def toOutput(tensor: Unit): Unit = {
        ()
      }
    }
  }

  implicit def fromTensor[T: TF]: Aux[Tensor[T], Output[T]] = {
    new TensorToOutput[Tensor[T]] {
      override type O = Output[T]

      override def tensors(tensor: Tensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor)
      }

      override def toOutput(tensor: Tensor[T]): Output[T] = {
        tensor.toOutput
      }
    }
  }

  implicit def fromTensorIndexedSlices[T]: Aux[TensorIndexedSlices[T], OutputIndexedSlices[T]] = {
    new TensorToOutput[TensorIndexedSlices[T]] {
      override type O = OutputIndexedSlices[T]

      override def tensors(tensor: TensorIndexedSlices[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices, tensor.values, tensor.denseShape)
      }

      override def toOutput(tensor: TensorIndexedSlices[T]): OutputIndexedSlices[T] = {
        OutputIndexedSlices(
          indices = tensor.indices,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape)
      }
    }
  }

  implicit def fromSparseTensor[T]: Aux[SparseTensor[T], SparseOutput[T]] = {
    new TensorToOutput[SparseTensor[T]] {
      override type O = SparseOutput[T]

      override def tensors(tensor: SparseTensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices, tensor.values, tensor.denseShape)
      }

      override def toOutput(tensor: SparseTensor[T]): SparseOutput[T] = {
        SparseOutput(
          indices = tensor.indices,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape)
      }
    }
  }

  implicit def fromOption[T, OO](implicit ev: Aux[T, OO]): Aux[Option[T], Option[OO]] = {
    new TensorToOutput[Option[T]] {
      override type O = Option[OO]

      override def tensors(tensor: Option[T]): Seq[Tensor[Any]] = {
        tensor.toSeq.flatMap(ev.tensors)
      }

      override def toOutput(tensor: Option[T]): Option[OO] = {
        tensor.map(ev.toOutput)
      }
    }
  }

  implicit def fromArray[T, OO: ClassTag](implicit
      ev: Aux[T, OO]
  ): Aux[Array[T], Array[OO]] = {
    new TensorToOutput[Array[T]] {
      override type O = Array[OO]

      override def tensors(tensor: Array[T]): Seq[Tensor[Any]] = {
        tensor.flatMap(ev.tensors).toSeq
      }

      override def toOutput(tensor: Array[T]): Array[OO] = {
        tensor.map(ev.toOutput)
      }
    }
  }

  implicit def fromSeq[T, OO](implicit ev: Aux[T, OO]): Aux[Seq[T], Seq[OO]] = {
    new TensorToOutput[Seq[T]] {
      override type O = Seq[OO]

      override def tensors(tensor: Seq[T]): Seq[Tensor[Any]] = {
        tensor.flatMap(ev.tensors)
      }

      override def toOutput(tensor: Seq[T]): Seq[OO] = {
        tensor.map(ev.toOutput)
      }
    }
  }

  implicit def fromMap[K, T, OO](implicit ev: Aux[T, OO]): Aux[Map[K, T], Map[K, OO]] = {
    new TensorToOutput[Map[K, T]] {
      override type O = Map[K, OO]

      override def tensors(tensor: Map[K, T]): Seq[Tensor[Any]] = {
        tensor.values.flatMap(ev.tensors).toSeq
      }

      override def toOutput(tensor: Map[K, T]): Map[K, OO] = {
        tensor.mapValues(ev.toOutput)
      }
    }
  }

  implicit val fromHNil: Aux[HNil, HNil] = {
    new TensorToOutput[HNil] {
      override type O = HNil

      override def tensors(tensor: HNil): Seq[Tensor[Any]] = {
        Seq.empty
      }

      override def toOutput(tensor: HNil): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HO, TT <: HList, TO <: HList](implicit
      evH: Strict[Aux[HT, HO]],
      evT: Aux[TT, TO]
  ): Aux[HT :: TT, HO :: TO] = {
    new TensorToOutput[HT :: TT] {
      override type O = HO :: TO

      override def tensors(tensor: HT :: TT): Seq[Tensor[Any]] = {
        evH.value.tensors(tensor.head) ++
            evT.tensors(tensor.tail)
      }

      override def toOutput(tensor: HT :: TT): HO :: TO = {
        evH.value.toOutput(tensor.head) ::
            evT.toOutput(tensor.tail)
      }
    }
  }

  implicit def fromCoproduct[HT, HO, TT <: Coproduct, TO <: Coproduct](implicit
      evH: Strict[Aux[HT, HO]],
      evT: Aux[TT, TO]
  ): Aux[HT :+: TT, HO :+: TO] = {
    new TensorToOutput[HT :+: TT] {
      override type O = HO :+: TO

      override def tensors(tensor: HT :+: TT): Seq[Tensor[Any]] = {
        tensor match {
          case Inl(h) => evH.value.tensors(h)
          case Inr(t) => evT.tensors(t)
        }
      }

      override def toOutput(tensor: HT :+: TT): HO :+: TO = {
        tensor match {
          case Inl(h) => Inl(evH.value.toOutput(h))
          case Inr(t) => Inr(evT.toOutput(t))
        }
      }
    }
  }

  implicit def fromProduct[PT <: Product, PO <: Product, HT <: HList, HO <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[Aux[HT, HO]],
      tuplerO: Tupler.Aux[HO, PO]
  ): Aux[PT, PO] = {
    new TensorToOutput[PT] {
      override type O = PO

      override def tensors(tensor: PT): Seq[Tensor[Any]] = {
        evT.value.tensors(genT.to(tensor))
      }

      override def toOutput(tensor: PT): PO = {
        tuplerO(evT.value.toOutput(genT.to(tensor)))
      }
    }
  }
}
