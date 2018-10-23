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
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._

/** Type trait used to represent nested structures over shapes.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait ShapeStructure[S] {
  def size(shape: S): Int
  def shapes(shape: S): Seq[Shape]
  def decodeShape(shape: S, shapes: Seq[Shape]): (S, Seq[Shape])
}

object ShapeStructure {
  implicit val fromUnit: ShapeStructure[Unit] = {
    new ShapeStructure[Unit] {
      override def size(shape: Unit): Int = {
        0
      }

      override def shapes(shape: Unit): Seq[Shape] = {
        Seq.empty
      }

      override def decodeShape(
          shape: Unit,
          shapes: Seq[Shape]
      ): (Unit, Seq[Shape]) = {
        ((), shapes)
      }
    }
  }

  implicit def fromOutput[T]: ShapeStructure[Shape] = {
    new ShapeStructure[Shape] {
      override def size(shape: Shape): Int = {
        1
      }

      override def shapes(shape: Shape): Seq[Shape] = {
        Seq(shape)
      }

      override def decodeShape(
          shape: Shape,
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes.tail)
      }
    }
  }

  implicit def fromOption[D](implicit ev: ShapeStructure[D]): ShapeStructure[Option[D]] = {
    new ShapeStructure[Option[D]] {
      override def size(shape: Option[D]): Int = {
        shape.map(ev.size).sum
      }

      override def shapes(shape: Option[D]): Seq[Shape] = {
        shape.toSeq.flatMap(ev.shapes)
      }

      override def decodeShape(
          shape: Option[D],
          shapes: Seq[Shape]
      ): (Option[D], Seq[Shape]) = {
        shape match {
          case Some(d) =>
            val (result, remaining) = ev.decodeShape(d, shapes)
            (Some(result), remaining)
          case None => (None, shapes)
        }
      }
    }
  }

  implicit def fromSeq[D](implicit
      ev: ShapeStructure[D]
  ): ShapeStructure[Seq[D]] = {
    new ShapeStructure[Seq[D]] {
      override def size(shape: Seq[D]): Int = {
        shape.map(ev.size).sum
      }

      override def shapes(shape: Seq[D]): Seq[Shape] = {
        shape.flatMap(ev.shapes)
      }

      override def decodeShape(
          shape: Seq[D],
          shapes: Seq[Shape]
      ): (Seq[D], Seq[Shape]) = {
        val n = size(shape)
        (shape
            .zip(Collections.segment(shapes.take(n), shape.map(ev.size)))
            .map(f => ev.decodeShape(f._1, f._2)._1), shapes.drop(n))
      }
    }
  }

  implicit def fromMap[K, D](implicit
      ev: ShapeStructure[D]
  ): ShapeStructure[Map[K, D]] = {
    new ShapeStructure[Map[K, D]] {
      override def size(shape: Map[K, D]): Int = {
        shape.values.map(ev.size).sum
      }

      override def shapes(shape: Map[K, D]): Seq[Shape] = {
        shape.values.flatMap(ev.shapes).toSeq
      }

      override def decodeShape(
          shape: Map[K, D],
          shapes: Seq[Shape]
      ): (Map[K, D], Seq[Shape]) = {
        val n = size(shape)
        (shape.keys.zip(
          shape.values
              .zip(Collections.segment(shapes.take(n), shape.values.map(ev.size).toSeq))
              .map(f => ev.decodeShape(f._1, f._2)._1)).toMap, shapes.drop(n))
      }
    }
  }

  implicit val fromHNil: ShapeStructure[HNil] = {
    new ShapeStructure[HNil] {
      override def size(shape: HNil): Int = {
        0
      }

      override def shapes(shape: HNil): Seq[Shape] = {
        Seq.empty
      }

      override def decodeShape(
          shape: HNil,
          shapes: Seq[Shape]
      ): (HNil, Seq[Shape]) = {
        (HNil, shapes)
      }
    }
  }

  implicit def fromHList[HS, TS <: HList](implicit
      evH: Strict[ShapeStructure[HS]],
      evT: ShapeStructure[TS]
  ): ShapeStructure[HS :: TS] = {
    new ShapeStructure[HS :: TS] {
      override def size(shape: HS :: TS): Int = {
        evH.value.size(shape.head) + evT.size(shape.tail)
      }

      override def shapes(shape: HS :: TS): Seq[Shape] = {
        evH.value.shapes(shape.head) ++ evT.shapes(shape.tail)
      }

      override def decodeShape(
          shape: HS :: TS,
          shapes: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = evH.value.decodeShape(shape.head, shapes)
        val (tailOut, tailRemaining) = evT.decodeShape(shape.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }
    }
  }

  implicit def fromProduct[PS <: Product, HS <: HList](implicit
      genD: Generic.Aux[PS, HS],
      evD: ShapeStructure[HS]
  ): ShapeStructure[PS] = {
    new ShapeStructure[PS] {
      override def size(shape: PS): Int = {
        evD.size(genD.to(shape))
      }

      override def shapes(shape: PS): Seq[Shape] = {
        evD.shapes(genD.to(shape))
      }

      override def decodeShape(
          shape: PS,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evD.decodeShape(genD.to(shape), shapes)
        (genD.from(out), remaining)
      }
    }
  }

  implicit def fromCoproduct[HS, TS <: Coproduct](implicit
      evH: Strict[ShapeStructure[HS]],
      evT: ShapeStructure[TS]
  ): ShapeStructure[HS :+: TS] = {
    new ShapeStructure[HS :+: TS] {
      override def size(shape: HS :+: TS): Int = {
        shape match {
          case Inl(h) => evH.value.size(h)
          case Inr(t) => evT.size(t)
        }
      }

      override def shapes(shape: HS :+: TS): Seq[Shape] = {
        shape match {
          case Inl(h) => evH.value.shapes(h)
          case Inr(t) => evT.shapes(t)
        }
      }

      override def decodeShape(
          shape: HS :+: TS,
          shapes: Seq[Shape]
      ): (HS :+: TS, Seq[Shape]) = {
        shape match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeShape(h, shapes)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeShape(t, shapes)
            (Inr(result), remaining)
        }
      }
    }
  }
}
