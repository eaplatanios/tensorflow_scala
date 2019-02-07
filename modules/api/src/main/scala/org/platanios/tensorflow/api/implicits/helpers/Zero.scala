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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._

import shapeless._
import shapeless.ops.hlist.Tupler

/** Represents types that have a "zero" value (e.g., RNN states).
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Zero[T] {
  type S // Shape type

  def evOutputToShape: OutputToShape.Aux[T, S]

  /** Generates a zero value of type `T`. */
  def zero(
      batchSize: Output[Int],
      shape: S,
      name: String = "Zero"
  ): T
}

object Zero {
  def apply[T](implicit ev: Zero[T]): Zero.Aux[T, ev.S] = {
    ev.asInstanceOf[Zero.Aux[T, ev.S]]
  }

  type Aux[T, SS] = Zero[T] {
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new Zero[Unit] {
      override type S = Unit

      override def evOutputToShape: OutputToShape.Aux[Unit, Unit] = {
        OutputToShape.fromUnit
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

  implicit def fromOutput[T: TF]: Aux[Output[T], Shape] = {
    new Zero[Output[T]] {
      override type S = Shape

      override def evOutputToShape: OutputToShape.Aux[Output[T], Shape] = {
        OutputToShape.fromOutput[T]
      }

      override def zero(
          batchSize: Output[Int],
          shape: Shape,
          name: String = "Zero"
      ): Output[T] = {
        val staticBatchSize = Output.constantValue(batchSize).map(_.scalar).getOrElse(-1)
        Op.nameScope(name) {
          val fullShape = Basic.concatenate(Seq(
            batchSize.expandDims(0),
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
  ): Zero.Aux[Option[T], Option[ev.S]] = {
    new Zero[Option[T]] {
      override type S = Option[ev.S]

      override def evOutputToShape: OutputToShape.Aux[Option[T], Option[ev.S]] = {
        OutputToShape.fromOption[T](ev.evOutputToShape)
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
  ): Zero.Aux[Seq[T], Seq[ev.S]] = {
    new Zero[Seq[T]] {
      override type S = Seq[ev.S]

      override def evOutputToShape: OutputToShape.Aux[Seq[T], Seq[ev.S]] = {
        OutputToShape.fromSeq[T](ev.evOutputToShape)
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
  ): Zero.Aux[Map[K, T], Map[K, ev.S]] = {
    new Zero[Map[K, T]] {
      override type S = Map[K, ev.S]

      override def evOutputToShape: OutputToShape.Aux[Map[K, T], Map[K, ev.S]] = {
        OutputToShape.fromMap[K, T](ev.evOutputToShape)
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

  implicit val fromHNil: Zero.Aux[HNil, HNil] = {
    new Zero[HNil] {
      override type S = HNil

      override def evOutputToShape: OutputToShape.Aux[HNil, HNil] = {
        OutputToShape.fromHNil
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

  implicit def fromHList[HT, HS, TT <: HList, TS <: HList](implicit
      evH: Strict[Zero.Aux[HT, HS]],
      evT: Strict[Zero.Aux[TT, TS]]
  ): Zero.Aux[HT :: TT, HS :: TS] = {
    new Zero[HT :: TT] {
      override type S = HS :: TS

      override def evOutputToShape: OutputToShape.Aux[HT :: TT, HS :: TS] = {
        OutputToShape.fromHList[HT, HS, TT, TS](evH.value.evOutputToShape, evT.value.evOutputToShape)
      }

      override def zero(
          batchSize: Output[Int],
          shape: HS :: TS,
          name: String = "Zero"
      ): HT :: TT = {
        Op.nameScope(name) {
          evH.value.zero(batchSize, shape.head) ::
              evT.value.zero(batchSize, shape.tail)
        }
      }
    }
  }

  implicit def fromProduct[PT <: Product, PS <: Product, HT <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evH: Strict[Zero.Aux[HT, HS]],
      tuplerS: Tupler.Aux[HS, PS],
      genS: Generic.Aux[PS, HS]
  ): Zero.Aux[PT, PS] = {
    new Zero[PT] {
      override type S = PS

      override def evOutputToShape: OutputToShape.Aux[PT, PS] = {
        OutputToShape.fromProduct[PT, PS, HT, HS](genT, evH.value.evOutputToShape, tuplerS, genS)
      }

      override def zero(
          batchSize: Output[Int],
          shape: PS,
          name: String = "Zero"
      ): PT = {
        genT.from(evH.value.zero(batchSize, genS.to(shape), name))
      }
    }
  }
}
