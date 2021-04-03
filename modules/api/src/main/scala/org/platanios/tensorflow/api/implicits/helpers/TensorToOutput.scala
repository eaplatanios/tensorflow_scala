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

import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.compat._

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorToOutput[T] {
  type O

  def tensorStructure: TensorStructure[T]
  def outputStructure: OutputStructure[O]
  def output(tensor: T): O
}

object TensorToOutput extends TensorToOutputLowPriorityImplicits {
  type Aux[T, OO] = TensorToOutput[T] {
    type O = OO
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new TensorToOutput[Unit] {
      override type O = Unit

      override def tensorStructure: TensorStructure[Unit] = TensorStructure.fromUnit
      override def outputStructure: OutputStructure[Unit] = OutputStructure.fromUnit
      override def output(tensor: Unit): Unit = ()
    }
  }

  implicit def fromTensor[T]: Aux[Tensor[T], Output[T]] = {
    new TensorToOutput[Tensor[T]] {
      override type O = Output[T]

      override def tensorStructure: TensorStructure[Tensor[T]] = TensorStructure.fromTensor[T]
      override def outputStructure: OutputStructure[Output[T]] = OutputStructure.fromOutput[T]
      override def output(tensor: Tensor[T]): Output[T] = tensor.toOutput
    }
  }

  implicit def fromTensorIndexedSlices[T]: Aux[TensorIndexedSlices[T], OutputIndexedSlices[T]] = {
    new TensorToOutput[TensorIndexedSlices[T]] {
      override type O = OutputIndexedSlices[T]

      override def tensorStructure: TensorStructure[TensorIndexedSlices[T]] = {
        TensorStructure.fromTensorIndexedSlices[T]
      }

      override def outputStructure: OutputStructure[OutputIndexedSlices[T]] = {
        OutputStructure.fromOutputIndexedSlices[T]
      }

      override def output(tensor: TensorIndexedSlices[T]): OutputIndexedSlices[T] = {
        OutputIndexedSlices(
          indices = tensor.indices.toOutput,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape.toOutput)
      }
    }
  }

  implicit def fromSparseTensor[T]: Aux[SparseTensor[T], SparseOutput[T]] = {
    new TensorToOutput[SparseTensor[T]] {
      override type O = SparseOutput[T]

      override def tensorStructure: TensorStructure[SparseTensor[T]] = {
        TensorStructure.fromSparseTensor[T]
      }

      override def outputStructure: OutputStructure[SparseOutput[T]] = {
        OutputStructure.fromSparseOutput[T]
      }

      override def output(tensor: SparseTensor[T]): SparseOutput[T] = {
        SparseOutput(
          indices = tensor.indices.toOutput,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape.toOutput)
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: TensorToOutput[T]
  ): TensorToOutput.Aux[Option[T], Option[ev.O]] = {
    new TensorToOutput[Option[T]] {
      override type O = Option[ev.O]

      override def tensorStructure: TensorStructure[Option[T]] = {
        TensorStructure.fromOption[T](ev.tensorStructure)
      }

      override def outputStructure: OutputStructure[Option[ev.O]] = {
        OutputStructure.fromOption[ev.O](ev.outputStructure)
      }

      override def output(tensor: Option[T]): Option[ev.O] = {
        tensor.map(t => ev.output(t))
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: TensorToOutput[T]
  ): TensorToOutput.Aux[Seq[T], Seq[ev.O]] = {
    new TensorToOutput[Seq[T]] {
      override type O = Seq[ev.O]

      override def tensorStructure: TensorStructure[Seq[T]] = {
        TensorStructure.fromSeq[T](ev.tensorStructure)
      }

      override def outputStructure: OutputStructure[Seq[ev.O]] = {
        OutputStructure.fromSeq[ev.O](ev.outputStructure)
      }

      override def output(tensor: Seq[T]): Seq[ev.O] = {
        tensor.map(t => ev.output(t))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: TensorToOutput[T]
  ): TensorToOutput.Aux[Map[K, T], Map[K, ev.O]] = {
    new TensorToOutput[Map[K, T]] {
      override type O = Map[K, ev.O]

      override def tensorStructure: TensorStructure[Map[K, T]] = {
        TensorStructure.fromMap[K, T](ev.tensorStructure)
      }

      override def outputStructure: OutputStructure[Map[K, ev.O]] = {
        OutputStructure.fromMap[K, ev.O](ev.outputStructure)
      }

      override def output(tensor: Map[K, T]): Map[K, ev.O] = {
        tensor.view.mapValues(t => ev.output(t)).toMap
      }
    }
  }

  implicit val fromHNil: TensorToOutput.Aux[HNil, HNil] = {
    new TensorToOutput[HNil] {
      override type O = HNil

      override def tensorStructure: TensorStructure[HNil] = TensorStructure.fromHNil
      override def outputStructure: OutputStructure[HNil] = OutputStructure.fromHNil
      override def output(tensor: HNil): HNil = HNil
    }
  }

  implicit def fromHList[HT, HO, TT <: HList, TO <: HList](implicit
      evH: Strict[TensorToOutput.Aux[HT, HO]],
      evT: Strict[TensorToOutput.Aux[TT, TO]]
  ): TensorToOutput.Aux[HT :: TT, HO :: TO] = {
    new TensorToOutput[HT :: TT] {
      override type O = HO :: TO

      override def tensorStructure: TensorStructure[HT :: TT] = {
        TensorStructure.fromHList[HT, TT](evH.value.tensorStructure, evT.value.tensorStructure)
      }

      override def outputStructure: OutputStructure[HO :: TO] = {
        OutputStructure.fromHList[HO, TO](evH.value.outputStructure, evT.value.outputStructure)
      }

      override def output(tensor: HT :: TT): HO :: TO = {
        evH.value.output(tensor.head) :: evT.value.output(tensor.tail)
      }
    }
  }

  implicit def fromKnownProduct[PT <: Product, PO, HT <: HList, HO <: HList](implicit
      genT: Generic.Aux[PT, HT],
      genO: Generic.Aux[PO, HO],
      evT: Strict[TensorToOutput.Aux[HT, HO]]
  ): TensorToOutput.Aux[PT, PO] = {
    new TensorToOutput[PT] {
      override type O = PO

      override def tensorStructure: TensorStructure[PT] = {
        TensorStructure.fromProduct[PT, HT](genT, evT.value.tensorStructure)
      }

      override def outputStructure: OutputStructure[PO] = {
        OutputStructure.fromProduct[PO, HO](genO, evT.value.outputStructure)
      }

      override def output(tensor: PT): PO = {
        genO.from(evT.value.output(genT.to(tensor)))
      }
    }
  }
}

trait TensorToOutputLowPriorityImplicits {
  implicit def fromProduct[PT <: Product, PO, HT <: HList, HO <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[TensorToOutput.Aux[HT, HO]],
      tuplerO: Tupler.Aux[HO, PO],
      genO: Generic.Aux[PO, HO]
  ): TensorToOutput.Aux[PT, PO] = {
    TensorToOutput.fromKnownProduct
  }
}
