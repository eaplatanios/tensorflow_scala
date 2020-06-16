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
import org.platanios.tensorflow.api.core.types.DataType
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of data types to structures of shapes.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataTypeToShape[D] {
  type S

  def sizeFromDataType(dataType: D): Int
  def decodeShape(dataType: D, shapes: Seq[Shape]): (S, Seq[Shape])
}

object DataTypeToShape extends DataTypeToShapeLowPriorityImplicits {
  def apply[D](implicit ev: DataTypeToShape[D]): Aux[D, ev.S] = {
    ev.asInstanceOf[Aux[D, ev.S]]
  }

  type Aux[D, SS] = DataTypeToShape[D] {
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new DataTypeToShape[Unit] {
      override type S = Unit

      override def sizeFromDataType(dataType: Unit): Int = {
        0
      }

      override def decodeShape(
          dataType: Unit,
          shapes: Seq[Shape]
      ): (Unit, Seq[Shape]) = {
        ((), shapes)
      }
    }
  }

  implicit def fromDataType[T]: Aux[DataType[T], Shape] = {
    new DataTypeToShape[DataType[T]] {
      override type S = Shape

      override def sizeFromDataType(dataType: DataType[T]): Int = {
        1
      }

      override def decodeShape(
          dataType: DataType[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes)
      }
    }
  }

  implicit def fromOption[D](implicit
      ev: DataTypeToShape[D]
  ): DataTypeToShape.Aux[Option[D], Option[ev.S]] = {
    new DataTypeToShape[Option[D]] {
      override type S = Option[ev.S]

      override def sizeFromDataType(dataType: Option[D]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def decodeShape(
          dataType: Option[D],
          shapes: Seq[Shape]
      ): (Option[ev.S], Seq[Shape]) = {
        dataType match {
          case Some(o) =>
            val (result, remaining) = ev.decodeShape(o, shapes)
            (Some(result), remaining)
          case None => (None, shapes)
        }
      }
    }
  }

  implicit def fromSeq[D](implicit
      ev: DataTypeToShape[D]
  ): DataTypeToShape.Aux[Seq[D], Seq[ev.S]] = {
    new DataTypeToShape[Seq[D]] {
      override type S = Seq[ev.S]

      override def sizeFromDataType(dataType: Seq[D]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def decodeShape(
          dataType: Seq[D],
          shapes: Seq[Shape]
      ): (Seq[ev.S], Seq[Shape]) = {
        val n = sizeFromDataType(dataType)
        (dataType
            .zip(Collections.segment(shapes.take(n), dataType.map(ev.sizeFromDataType)))
            .map(f => ev.decodeShape(f._1, f._2)._1), shapes.drop(n))
      }
    }
  }

  implicit def fromMap[K, D](implicit
      ev: DataTypeToShape[D]
  ): DataTypeToShape.Aux[Map[K, D], Map[K, ev.S]] = {
    new DataTypeToShape[Map[K, D]] {
      override type S = Map[K, ev.S]

      override def sizeFromDataType(dataType: Map[K, D]): Int = {
        dataType.values.map(ev.sizeFromDataType).sum
      }

      override def decodeShape(
          dataType: Map[K, D],
          shapes: Seq[Shape]
      ): (Map[K, ev.S], Seq[Shape]) = {
        val n = sizeFromDataType(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(shapes.take(n), dataType.values.map(ev.sizeFromDataType).toSeq))
              .map(f => ev.decodeShape(f._1, f._2)._1)).toMap, shapes.drop(n))
      }
    }
  }

  implicit val fromHNil: DataTypeToShape.Aux[HNil, HNil] = {
    new DataTypeToShape[HNil] {
      override type S = HNil

      override def sizeFromDataType(dataType: HNil): Int = {
        0
      }

      override def decodeShape(
          dataType: HNil,
          shapes: Seq[Shape]
      ): (HNil, Seq[Shape]) = {
        (HNil, shapes)
      }
    }
  }

  implicit def fromHList[HD, HS, TD <: HList, TS <: HList](implicit
      evH: Strict[DataTypeToShape.Aux[HD, HS]],
      evT: Strict[DataTypeToShape.Aux[TD, TS]]
  ): DataTypeToShape.Aux[HD :: TD, HS :: TS] = {
    new DataTypeToShape[HD :: TD] {
      override type S = HS :: TS

      override def sizeFromDataType(dataType: HD :: TD): Int = {
        evH.value.sizeFromDataType(dataType.head) + evT.value.sizeFromDataType(dataType.tail)
      }

      override def decodeShape(
          dataType: HD :: TD,
          shapes: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = evH.value.decodeShape(dataType.head, shapes)
        val (tailOut, tailRemaining) = evT.value.decodeShape(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }
    }
  }

  implicit def fromKnownProduct[PD <: Product, PS, HD <: HList, HS <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evD: Strict[DataTypeToShape.Aux[HD, HS]],
      genS: Generic.Aux[PS, HS]
  ): DataTypeToShape.Aux[PD, PS] = {
    new DataTypeToShape[PD] {
      override type S = PS

      override def sizeFromDataType(dataType: PD): Int = {
        evD.value.sizeFromDataType(genD.to(dataType))
      }

      override def decodeShape(
          dataType: PD,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evD.value.decodeShape(genD.to(dataType), shapes)
        (genS.from(out), remaining)
      }
    }
  }
}

trait DataTypeToShapeLowPriorityImplicits {
  implicit def fromProduct[PD <: Product, PS, HD <: HList, HS <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evD: Strict[DataTypeToShape.Aux[HD, HS]],
      tuplerS: Tupler.Aux[HS, PS],
      genS: Generic.Aux[PS, HS]
  ): DataTypeToShape.Aux[PD, PS] = {
    DataTypeToShape.fromKnownProduct
  }
}
