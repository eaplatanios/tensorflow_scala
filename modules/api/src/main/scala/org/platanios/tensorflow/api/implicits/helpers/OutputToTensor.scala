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

import org.platanios.tensorflow.api.core.types.Variant
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput, TensorArray}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of outputs to structures of tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait OutputToTensor[T] {
  type V

  def size(output: T): Int
  def decodeTensor(output: T, tensors: Seq[Tensor[Any]]): (V, Seq[Tensor[Any]])
}

object OutputToTensor {
  def apply[T](implicit ev: OutputToTensor[T]): Aux[T, ev.V] = {
    ev.asInstanceOf[Aux[T, ev.V]]
  }

  type Aux[T, VV] = OutputToTensor[T] {
    type V = VV
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new OutputToTensor[Unit] {
      override type V = Unit

      override def size(output: Unit): Int = {
        0
      }

      override def decodeTensor(
          output: Unit,
          tensors: Seq[Tensor[Any]]
      ): (Unit, Seq[Tensor[Any]]) = {
        ((), tensors)
      }
    }
  }

  implicit def fromOutput[T]: Aux[Output[T], Tensor[T]] = {
    new OutputToTensor[Output[T]] {
      override type V = Tensor[T]

      override def size(output: Output[T]): Int = {
        1
      }

      override def decodeTensor(
          output: Output[T],
          tensors: Seq[Tensor[Any]]
      ): (Tensor[T], Seq[Tensor[Any]]) = {
        (tensors.head.asInstanceOf[Tensor[T]], tensors.tail)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: Aux[OutputIndexedSlices[T], TensorIndexedSlices[T]] = {
    new OutputToTensor[OutputIndexedSlices[T]] {
      override type V = TensorIndexedSlices[T]

      override def size(output: OutputIndexedSlices[T]): Int = {
        3
      }

      override def decodeTensor(
          output: OutputIndexedSlices[T],
          tensors: Seq[Tensor[Any]]
      ): (TensorIndexedSlices[T], Seq[Tensor[Any]]) = {
        (TensorIndexedSlices[T](
          indices = tensors(0).asInstanceOf[Tensor[Long]],
          values = tensors(1).asInstanceOf[Tensor[T]],
          denseShape = tensors(2).asInstanceOf[Tensor[Long]]
        ), tensors.drop(3))
      }
    }
  }

  implicit def fromSparseOutput[T]: Aux[SparseOutput[T], SparseTensor[T]] = {
    new OutputToTensor[SparseOutput[T]] {
      override type V = SparseTensor[T]

      override def size(output: SparseOutput[T]): Int = {
        3
      }

      override def decodeTensor(
          output: SparseOutput[T],
          tensors: Seq[Tensor[Any]]
      ): (SparseTensor[T], Seq[Tensor[Any]]) = {
        (SparseTensor[T](
          indices = tensors(0).asInstanceOf[Tensor[Long]],
          values = tensors(1).asInstanceOf[Tensor[T]],
          denseShape = tensors(2).asInstanceOf[Tensor[Long]]
        ), tensors.drop(3))
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], Tensor[Float]] = {
    new OutputToTensor[TensorArray[T]] {
      override type V = Tensor[Float]

      override def size(output: TensorArray[T]): Int = {
        1
      }

      override def decodeTensor(
          output: TensorArray[T],
          tensors: Seq[Tensor[Any]]
      ): (Tensor[Float], Seq[Tensor[Any]]) = {
        (tensors.head.asInstanceOf[Tensor[Float]], tensors.tail)
      }
    }
  }

  implicit def fromDataset[T]: Aux[Dataset[T], Tensor[Variant]] = {
    new OutputToTensor[Dataset[T]] {
      override type V = Tensor[Variant]

      override def size(output: Dataset[T]): Int = {
        1
      }

      override def decodeTensor(
          output: Dataset[T],
          tensors: Seq[Tensor[Any]]
      ): (Tensor[Variant], Seq[Tensor[Any]]) = {
        (tensors.head.asInstanceOf[Tensor[Variant]], tensors.tail)
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: OutputToTensor[T]
  ): OutputToTensor.Aux[Option[T], Option[ev.V]] = {
    new OutputToTensor[Option[T]] {
      override type V = Option[ev.V]

      override def size(output: Option[T]): Int = {
        output.map(ev.size).sum
      }

      override def decodeTensor(
          output: Option[T],
          tensors: Seq[Tensor[Any]]
      ): (Option[ev.V], Seq[Tensor[Any]]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeTensor(o, tensors)
            (Some(result), remaining)
          case None => (None, tensors)
        }
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: OutputToTensor[T]
  ): OutputToTensor.Aux[Seq[T], Seq[ev.V]] = {
    new OutputToTensor[Seq[T]] {
      override type V = Seq[ev.V]

      override def size(output: Seq[T]): Int = {
        output.map(ev.size).sum
      }

      override def decodeTensor(
          output: Seq[T],
          tensors: Seq[Tensor[Any]]
      ): (Seq[ev.V], Seq[Tensor[Any]]) = {
        val n = size(output)
        (output
            .zip(Collections.segment(tensors.take(n), output.map(ev.size)))
            .map(f => ev.decodeTensor(f._1, f._2)._1), tensors.drop(n))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: OutputToTensor[T]
  ): OutputToTensor.Aux[Map[K, T], Map[K, ev.V]] = {
    new OutputToTensor[Map[K, T]] {
      override type V = Map[K, ev.V]

      override def size(output: Map[K, T]): Int = {
        output.values.map(ev.size).sum
      }

      override def decodeTensor(
          output: Map[K, T],
          tensors: Seq[Tensor[Any]]
      ): (Map[K, ev.V], Seq[Tensor[Any]]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(tensors.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.decodeTensor(f._1, f._2)._1)).toMap, tensors.drop(n))
      }
    }
  }

  implicit val fromHNil: OutputToTensor.Aux[HNil, HNil] = {
    new OutputToTensor[HNil] {
      override type V = HNil

      override def size(output: HNil): Int = {
        0
      }

      override def decodeTensor(
          output: HNil,
          tensors: Seq[Tensor[Any]]
      ): (HNil, Seq[Tensor[Any]]) = {
        (HNil, tensors)
      }
    }
  }

  implicit def fromHList[HT, HV, TT <: HList, TV <: HList](implicit
      evH: Strict[OutputToTensor.Aux[HT, HV]],
      evT: Strict[OutputToTensor.Aux[TT, TV]]
  ): OutputToTensor.Aux[HT :: TT, HV :: TV] = {
    new OutputToTensor[HT :: TT] {
      override type V = HV :: TV

      override def size(output: HT :: TT): Int = {
        evH.value.size(output.head) + evT.value.size(output.tail)
      }

      override def decodeTensor(
          output: HT :: TT,
          tensors: Seq[Tensor[Any]]
      ): (HV :: TV, Seq[Tensor[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeTensor(output.head, tensors)
        val (tailOut, tailRemaining) = evT.value.decodeTensor(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PV <: Product, HT <: HList, HV <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[OutputToTensor.Aux[HT, HV]],
      tuplerV: Tupler.Aux[HV, PV],
      genV: Generic.Aux[PV, HV]
  ): OutputToTensor.Aux[PT, PV] = {
    new OutputToTensor[PT] {
      override type V = PV

      override def size(output: PT): Int = {
        evT.value.size(genT.to(output))
      }

      override def decodeTensor(
          output: PT,
          tensors: Seq[Tensor[Any]]
      ): (PV, Seq[Tensor[Any]]) = {
        val (out, remaining) = evT.value.decodeTensor(genT.to(output), tensors)
        (genV.from(out), remaining)
      }
    }
  }
}
