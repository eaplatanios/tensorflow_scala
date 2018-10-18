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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.{DataType, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor

import shapeless._
import shapeless.ops.hlist.Tupler

/** Represents types that have a "zero" value (e.g., RNN states).
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Zero[T] {
  type V // Tensor value
  type D // Data type
  type S // Shape

  val structure: NestedStructure.Aux[T, V, D, S]

  def asAux(): Zero.Aux[T, V, D, S] = {
    this.asInstanceOf[Zero.Aux[T, V, D, S]]
  }

  /** Generates a zero value of type `T`. */
  def zero(
      batchSize: Output[Int],
      shape: S,
      name: String = "Zero"
  ): T
}

object Zero extends ZeroLowPriority {
  type SparseDataType[T] = (DataType[Long], DataType[T], DataType[Long])
  type SparseShape = (Shape, Shape, Shape)

  def apply[T](implicit ev: Zero[T]): Zero.Aux[T, ev.V, ev.D, ev.S] = {
    ev.asAux()
  }

  type Aux[T, VV, DD, SS] = Zero[T] {
    type V = VV
    type D = DD
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit, Unit, Unit] = {
    new Zero[Unit] {
      override type V = Unit
      override type D = Unit
      override type S = Unit

      override val structure: NestedStructure.Aux[Unit, Unit, Unit, Unit] = {
        NestedStructure.fromUnit
      }

      override def zero(
          batchSize: Output[Int],
          shape: Unit,
          name: String = "Zero"
      ): Unit = {
        ()
      }
    }
  }

  implicit def fromOutput[T: TF]: Aux[Output[T], Tensor[T], DataType[T], Shape] = {
    new Zero[Output[T]] {
      override type V = Tensor[T]
      override type D = DataType[T]
      override type S = Shape

      override val structure: NestedStructure.Aux[Output[T], Tensor[T], DataType[T], Shape] = {
        NestedStructure.fromOutput[T]
      }

      override def zero(
          batchSize: Output[Int],
          shape: Shape,
          name: String = "Zero"
      ): Output[T] = {
        val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
        Op.nameScope(name) {
          val fullShape = Basic.concatenate(Seq(
            batchSize.expandDims(0).castTo[Long],
            shape.toOutput
          ), axis = 0)
          val zero = Basic.zeros[T](fullShape)
          zero.setShape(Shape(staticBatchSize) ++ shape)
          zero
        }
      }
    }
  }

  // TODO: [TYPES] !!! What about OutputIndexedSlices and TensorIndexedSlices?

  implicit def fromOption[T](implicit
      ev: Zero[T]
  ): Zero.Aux[Option[T], Option[ev.V], Option[ev.D], Option[ev.S]] = {
    new Zero[Option[T]] {
      override type V = Option[ev.V]
      override type D = Option[ev.D]
      override type S = Option[ev.S]

      override val structure: NestedStructure.Aux[Option[T], Option[ev.V], Option[ev.D], Option[ev.S]] = {
        NestedStructure.fromOption[T](ev.structure)
      }

      override def zero(
          batchSize: Output[Int],
          shape: Option[ev.S],
          name: String
      ): Option[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: Zero[T]
  ): Zero.Aux[Seq[T], Seq[ev.V], Seq[ev.D], Seq[ev.S]] = {
    new Zero[Seq[T]] {
      override type V = Seq[ev.V]
      override type D = Seq[ev.D]
      override type S = Seq[ev.S]

      override val structure: NestedStructure.Aux[Seq[T], Seq[ev.V], Seq[ev.D], Seq[ev.S]] = {
        NestedStructure.fromSeq[T](ev.structure)
      }

      override def zero(
          batchSize: Output[Int],
          shape: Seq[ev.S],
          name: String
      ): Seq[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: Zero[T]
  ): Zero.Aux[Map[K, T], Map[K, ev.V], Map[K, ev.D], Map[K, ev.S]] = {
    new Zero[Map[K, T]] {
      override type V = Map[K, ev.V]
      override type D = Map[K, ev.D]
      override type S = Map[K, ev.S]

      override val structure: NestedStructure.Aux[Map[K, T], Map[K, ev.V], Map[K, ev.D], Map[K, ev.S]] = {
        NestedStructure.fromMap[K, T](ev.structure)
      }

      override def zero(
          batchSize: Output[Int],
          shape: Map[K, ev.S],
          name: String
      ): Map[K, T] = {
        Op.nameScope(name) {
          shape.mapValues(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit val fromHNil: Zero.Aux[HNil, HNil, HNil, HNil] = {
    new Zero[HNil] {
      override type V = HNil
      override type D = HNil
      override type S = HNil

      override val structure: NestedStructure.Aux[HNil, HNil, HNil, HNil] = {
        NestedStructure.fromHNil
      }

      override def zero(
          batchSize: Output[Int],
          shape: HNil,
          name: String = "Zero"
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HV, HD, HS, TT <: HList, TV <: HList, TD <: HList, TS <: HList](implicit
      evH: Strict[Zero.Aux[HT, HV, HD, HS]],
      evT: Zero.Aux[TT, TV, TD, TS]
  ): Zero.Aux[HT :: TT, HV :: TV, HD :: TD, HS :: TS] = {
    new Zero[HT :: TT] {
      override type V = HV :: TV
      override type D = HD :: TD
      override type S = HS :: TS

      implicit val evStructureH: NestedStructure.Aux[HT, HV, HD, HS] = evH.value.structure
      implicit val evStructureT: NestedStructure.Aux[TT, TV, TD, TS] = evT.structure

      override val structure: NestedStructure.Aux[HT :: TT, HV :: TV, HD :: TD, HS :: TS] = {
        NestedStructure.fromHList[HT, HV, HD, HS, TT, TV, TD, TS]
      }

      override def zero(
          batchSize: Output[Int],
          shape: HS :: TS,
          name: String = "Zero"
      ): HT :: TT = {
        Op.nameScope(name) {
          evH.value.zero(batchSize, shape.head) ::
              evT.zero(batchSize, shape.tail)
        }
      }
    }
  }

  implicit def fromProduct[PT <: Product, PV <: Product, PD <: Product, PS <: Product, HT <: HList, HV <: HList, HD <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evZeroH: Zero.Aux[HT, HV, HD, HS],
      tuplerV: Tupler.Aux[HV, PV],
      tuplerD: Tupler.Aux[HD, PD],
      tuplerS: Tupler.Aux[HS, PS],
      genV: Generic.Aux[PV, HV],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS]
  ): Zero.Aux[PT, PV, PD, PS] = {
    new Zero[PT] {
      override type V = PV
      override type D = PD
      override type S = PS

      implicit val evStructureH: NestedStructure.Aux[HT, HV, HD, HS] = evZeroH.structure

      override val structure: NestedStructure.Aux[PT, PV, PD, PS] = {
        NestedStructure.fromProduct[PT, PV, PD, PS, HT, HV, HD, HS]
      }

      override def zero(
          batchSize: Output[Int],
          shape: PS,
          name: String = "Zero"
      ): PT = {
        genT.from(evZeroH.zero(batchSize, genS.to(shape), name))
      }
    }
  }
}

trait ZeroLowPriority {
  implicit def fromCoproduct[HT, HV, HD, HS, TT <: Coproduct, TV <: Coproduct, TD <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[Zero.Aux[HT, HV, HD, HS]],
      evT: Zero.Aux[TT, TV, TD, TS]
  ): Zero.Aux[HT :+: TT, HV :+: TV, HD :+: TD, HS :+: TS] = {
    new Zero[HT :+: TT] {
      override type V = HV :+: TV
      override type D = HD :+: TD
      override type S = HS :+: TS

      implicit val evStructureH: NestedStructure.Aux[HT, HV, HD, HS] = evH.value.structure
      implicit val evStructureT: NestedStructure.Aux[TT, TV, TD, TS] = evT.structure

      override val structure: NestedStructure.Aux[HT :+: TT, HV :+: TV, HD :+: TD, HS :+: TS] = {
        NestedStructure.fromCoproduct[HT, HV, HD, HS, TT, TV, TD, TS]
      }

      override def zero(
          batchSize: Output[Int],
          shape: HS :+: TS,
          name: String
      ): HT :+: TT = {
        shape match {
          case Inl(h) => Inl(evH.value.zero(batchSize, h, name))
          case Inr(t) => Inr(evT.zero(batchSize, t, name))
        }
      }
    }
  }
}
