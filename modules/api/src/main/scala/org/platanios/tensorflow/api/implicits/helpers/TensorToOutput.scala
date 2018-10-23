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

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorToOutput[T] {
  type O

  def output(tensor: T): O
  def tensors(tensor: T): Seq[Tensor[Any]]
}

object TensorToOutput {
  type Aux[T, OO] = TensorToOutput[T] {
    type O = OO
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new TensorToOutput[Unit] {
      override type O = Unit

      override def output(tensor: Unit): Unit = {
        ()
      }

      override def tensors(tensor: Unit): Seq[Tensor[Any]] = {
        Seq.empty
      }
    }
  }

  implicit def fromTensor[T]: Aux[Tensor[T], Output[T]] = {
    new TensorToOutput[Tensor[T]] {
      override type O = Output[T]

      override def output(tensor: Tensor[T]): Output[T] = {
        tensor.toOutput
      }

      override def tensors(tensor: Tensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor)
      }
    }
  }

  implicit def fromTensorIndexedSlices[T]: Aux[TensorIndexedSlices[T], OutputIndexedSlices[T]] = {
    new TensorToOutput[TensorIndexedSlices[T]] {
      override type O = OutputIndexedSlices[T]

      override def output(tensor: TensorIndexedSlices[T]): OutputIndexedSlices[T] = {
        OutputIndexedSlices(
          indices = tensor.indices,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape)
      }

      override def tensors(tensor: TensorIndexedSlices[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices, tensor.values, tensor.denseShape)
      }
    }
  }

  implicit def fromSparseTensor[T]: Aux[SparseTensor[T], SparseOutput[T]] = {
    new TensorToOutput[SparseTensor[T]] {
      override type O = SparseOutput[T]

      override def output(tensor: SparseTensor[T]): SparseOutput[T] = {
        SparseOutput(
          indices = tensor.indices,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape)
      }

      override def tensors(tensor: SparseTensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices, tensor.values, tensor.denseShape)
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: TensorToOutput[T]
  ): TensorToOutput.Aux[Option[T], Option[ev.O]] = {
    new TensorToOutput[Option[T]] {
      override type O = Option[ev.O]

      override def output(tensor: Option[T]): Option[ev.O] = {
        tensor.map(t => ev.output(t))
      }

      override def tensors(tensor: Option[T]): Seq[Tensor[Any]] = {
        tensor.toSeq.flatMap(ev.tensors)
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: TensorToOutput[T]
  ): TensorToOutput.Aux[Seq[T], Seq[ev.O]] = {
    new TensorToOutput[Seq[T]] {
      override type O = Seq[ev.O]

      override def output(tensor: Seq[T]): Seq[ev.O] = {
        tensor.map(t => ev.output(t))
      }

      override def tensors(tensor: Seq[T]): Seq[Tensor[Any]] = {
        tensor.flatMap(ev.tensors)
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: TensorToOutput[T]
  ): TensorToOutput.Aux[Map[K, T], Map[K, ev.O]] = {
    new TensorToOutput[Map[K, T]] {
      override type O = Map[K, ev.O]

      override def output(tensor: Map[K, T]): Map[K, ev.O] = {
        tensor.mapValues(t => ev.output(t))
      }

      override def tensors(tensor: Map[K, T]): Seq[Tensor[Any]] = {
        tensor.values.flatMap(ev.tensors).toSeq
      }
    }
  }

  implicit val fromHNil: TensorToOutput.Aux[HNil, HNil] = {
    new TensorToOutput[HNil] {
      override type O = HNil

      override def output(tensor: HNil): HNil = {
        HNil
      }

      override def tensors(tensor: HNil): Seq[Tensor[Any]] = {
        Seq.empty
      }
    }
  }

  implicit def fromHList[HT, HO, TT <: HList, TO <: HList](implicit
      evH: Strict[TensorToOutput.Aux[HT, HO]],
      evT: TensorToOutput.Aux[TT, TO]
  ): TensorToOutput.Aux[HT :: TT, HO :: TO] = {
    new TensorToOutput[HT :: TT] {
      override type O = HO :: TO

      override def output(tensor: HT :: TT): HO :: TO = {
        evH.value.output(tensor.head) :: evT.output(tensor.tail)
      }

      override def tensors(tensor: HT :: TT): Seq[Tensor[Any]] = {
        evH.value.tensors(tensor.head) ++ evT.tensors(tensor.tail)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PO <: Product, HT <: HList, HO <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: TensorToOutput.Aux[HT, HO],
      tuplerO: Tupler.Aux[HO, PO],
      genO: Generic.Aux[PO, HO]
  ): TensorToOutput.Aux[PT, PO] = {
    new TensorToOutput[PT] {
      override type O = PO

      override def output(tensor: PT): PO = {
        genO.from(evT.output(genT.to(tensor)))
      }

      override def tensors(tensor: PT): Seq[Tensor[Any]] = {
        evT.tensors(genT.to(tensor))
      }
    }
  }

  implicit def fromCoproduct[HT, HO, TT <: Coproduct, TO <: Coproduct](implicit
      evH: Strict[TensorToOutput.Aux[HT, HO]],
      evT: TensorToOutput.Aux[TT, TO]
  ): TensorToOutput.Aux[HT :+: TT, HO :+: TO] = {
    new TensorToOutput[HT :+: TT] {
      override type O = HO :+: TO

      override def output(tensor: HT :+: TT): HO :+: TO = {
        tensor match {
          case Inl(h) => Inl(evH.value.output(h))
          case Inr(t) => Inr(evT.output(t))
        }
      }

      override def tensors(tensor: HT :+: TT): Seq[Tensor[Any]] = {
        tensor match {
          case Inl(h) => evH.value.tensors(h)
          case Inr(t) => evT.tensors(t)
        }
      }
    }
  }
}
