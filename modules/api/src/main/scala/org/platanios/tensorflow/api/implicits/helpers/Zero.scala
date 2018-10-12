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
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.reflect.ClassTag

/** Represents types that have a "zero" value (e.g., RNN states).
  *
  * @author Emmanouil Antonios Platanios
  */
trait Zero[T] {
  type S // Shape type

  /** Generates a zero value of type `T`. */
  def zero(
      batchSize: Output[Int],
      shape: S,
      name: String = "Zero"
  ): T
}

object Zero {
  def apply[T, S](implicit ev: Zero.Aux[T, S]): Zero.Aux[T, S] = {
    ev
  }

  type Aux[T, SS] = Zero[T] {
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new Zero[Unit] {
      override type S = Unit

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

  implicit def fromOption[T, SS](implicit
      ev: Aux[T, SS]
  ): Aux[Option[T], Option[SS]] = {
    new Zero[Option[T]] {
      override type S = Option[SS]

      override def zero(
          batchSize: Output[Int],
          shape: Option[SS],
          name: String
      ): Option[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit def fromArray[T: ClassTag, SS: ClassTag](implicit
      ev: Aux[T, SS]
  ): Aux[Array[T], Array[SS]] = {
    new Zero[Array[T]] {
      override type S = Array[SS]

      override def zero(
          batchSize: Output[Int],
          shape: Array[SS],
          name: String
      ): Array[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit def fromSeq[T, SS](implicit
      ev: Aux[T, SS]
  ): Aux[Seq[T], Seq[SS]] = {
    new Zero[Seq[T]] {
      override type S = Seq[SS]

      override def zero(
          batchSize: Output[Int],
          shape: Seq[SS],
          name: String
      ): Seq[T] = {
        Op.nameScope(name) {
          shape.map(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit def fromMap[K, T, SS](implicit
      ev: Aux[T, SS]
  ): Aux[Map[K, T], Map[K, SS]] = {
    new Zero[Map[K, T]] {
      override type S = Map[K, SS]

      override def zero(
          batchSize: Output[Int],
          shape: Map[K, SS],
          name: String
      ): Map[K, T] = {
        Op.nameScope(name) {
          shape.mapValues(ev.zero(batchSize, _))
        }
      }
    }
  }

  implicit val fromHNil: Aux[HNil, HNil] = {
    new Zero[HNil] {
      override type S = HNil

      override def zero(
          batchSize: Output[Int],
          shape: HNil,
          name: String = "Zero"
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[H, HS, T <: HList, TS <: HList](implicit
      evH: Strict[Aux[H, HS]],
      evT: Aux[T, TS]
  ): Aux[H :: T, HS :: TS] = {
    new Zero[H :: T] {
      override type S = HS :: TS

      override def zero(
          batchSize: Output[Int],
          shape: HS :: TS,
          name: String = "Zero"
      ): H :: T = {
        Op.nameScope(name) {
          evH.value.zero(batchSize, shape.head) ::
              evT.zero(batchSize, shape.tail)
        }
      }
    }
  }

  implicit def fromCoproduct[H, HS, T <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[Aux[H, HS]],
      evT: Aux[T, TS]
  ): Aux[H :+: T, HS :+: TS] = {
    new Zero[H :+: T] {
      override type S = HS :+: TS

      override def zero(
          batchSize: Output[Int],
          shape: HS :+: TS,
          name: String
      ): H :+: T = {
        shape match {
          case Inl(h) => Inl(evH.value.zero(batchSize, h, name))
          case Inr(t) => Inr(evT.zero(batchSize, t, name))
        }
      }
    }
  }

  implicit def fromProduct[P <: Product, PS <: Product, L <: HList, LS <: HList](implicit
      genP: Generic.Aux[P, L],
      evL: Strict[Aux[L, LS]],
      tuplerS: Tupler.Aux[LS, PS],
      genS: Generic.Aux[PS, LS]
  ): Aux[P, PS] = {
    new Zero[P] {
      override type S = PS

      override def zero(
          batchSize: Output[Int],
          shape: PS,
          name: String = "Zero"
      ): P = {
        genP.from(evL.value.zero(batchSize, genS.to(shape), name))
      }
    }
  }
}
