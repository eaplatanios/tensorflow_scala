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
import org.platanios.tensorflow.api.core.types.Variant
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput, TensorArray}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait OutputToShape[T] {
  type S

  def outputStructure: OutputStructure[T]
  def shapeStructure: ShapeStructure[S]

  def sizeFromOutput(output: T): Int
  def shape(output: T): S
  def decodeShape(output: T, shapes: Seq[Shape]): (S, Seq[Shape])

  def map(
      value: T,
      shape: Option[S],
      converter: OutputStructure.Converter
  ): T
}

object OutputToShape {
  def apply[T](implicit ev: OutputToShape[T]): OutputToShape.Aux[T, ev.S] = {
    ev.asInstanceOf[OutputToShape.Aux[T, ev.S]]
  }

  type Aux[T, SS] = OutputToShape[T] {
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new OutputToShape[Unit] {
      override type S = Unit

      override def outputStructure: OutputStructure[Unit] = {
        OutputStructure.fromUnit
      }

      override def shapeStructure: ShapeStructure[Unit] = {
        ShapeStructure.fromUnit
      }

      override def sizeFromOutput(output: Unit): Int = {
        0
      }

      override def shape(output: Unit): Unit = {
        ()
      }

      override def decodeShape(
          output: Unit,
          shapes: Seq[Shape]
      ): (Unit, Seq[Shape]) = {
        ((), shapes)
      }

      def map(
          value: Unit,
          shape: Option[Unit],
          converter: OutputStructure.Converter
      ): Unit = {
        ()
      }
    }
  }

  implicit def fromOutput[T]: Aux[Output[T], Shape] = {
    new OutputToShape[Output[T]] {
      override type S = Shape

      override def outputStructure: OutputStructure[Output[T]] = {
        OutputStructure.fromOutput[T]
      }

      override def shapeStructure: ShapeStructure[Shape] = {
        ShapeStructure.fromOutput[T]
      }

      override def sizeFromOutput(output: Output[T]): Int = {
        1
      }

      override def shape(output: Output[T]): Shape = {
        output.shape
      }

      override def decodeShape(
          output: Output[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes)
      }

      override def map(
          value: Output[T],
          shape: Option[S],
          converter: OutputStructure.Converter
      ): Output[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: Aux[OutputIndexedSlices[T], SparseShape] = {
    new OutputToShape[OutputIndexedSlices[T]] {
      override type S = SparseShape

      override def outputStructure: OutputStructure[OutputIndexedSlices[T]] = {
        OutputStructure.fromOutputIndexedSlices[T]
      }

      override def shapeStructure: ShapeStructure[SparseShape] = {
        implicitly[ShapeStructure[SparseShape]]
      }

      override def sizeFromOutput(output: OutputIndexedSlices[T]): Int = {
        3
      }

      override def shape(output: OutputIndexedSlices[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }

      override def decodeShape(
          output: OutputIndexedSlices[T],
          shapes: Seq[Shape]
      ): (SparseShape, Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def map(
          value: OutputIndexedSlices[T],
          shape: Option[SparseShape],
          converter: OutputStructure.Converter
      ): OutputIndexedSlices[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromSparseOutput[T]: Aux[SparseOutput[T], SparseShape] = {
    new OutputToShape[SparseOutput[T]] {
      override type S = SparseShape

      override def outputStructure: OutputStructure[SparseOutput[T]] = {
        OutputStructure.fromSparseOutput[T]
      }

      override def shapeStructure: ShapeStructure[SparseShape] = {
        implicitly[ShapeStructure[SparseShape]]
      }

      override def sizeFromOutput(output: SparseOutput[T]): Int = {
        3
      }

      override def shape(output: SparseOutput[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }

      override def decodeShape(
          output: SparseOutput[T],
          shapes: Seq[Shape]
      ): (SparseShape, Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def map(
          value: SparseOutput[T],
          shape: Option[SparseShape],
          converter: OutputStructure.Converter
      ): SparseOutput[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], Shape] = {
    new OutputToShape[TensorArray[T]] {
      override type S = Shape

      override def outputStructure: OutputStructure[TensorArray[T]] = {
        OutputStructure.fromTensorArray[T]
      }

      override def shapeStructure: ShapeStructure[Shape] = {
        ShapeStructure.fromOutput[Float]
      }

      override def sizeFromOutput(output: TensorArray[T]): Int = {
        1
      }

      override def shape(output: TensorArray[T]): Shape = {
        Shape()
      }

      override def decodeShape(
          output: TensorArray[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes)
      }

      override def map(
          value: TensorArray[T],
          shape: Option[S],
          converter: OutputStructure.Converter
      ): TensorArray[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromDataset[T: OutputStructure, DD, SS](implicit
      evOutputToDataType: OutputToDataType.Aux[T, DD],
      evOutputToShape: OutputToShape.Aux[T, SS],
      evDataTypeToShape: DataTypeToShape.Aux[DD, SS]
  ): Aux[Dataset[T], Shape] = {
    new OutputToShape[Dataset[T]] {
      override type S = Shape

      override def outputStructure: OutputStructure[Dataset[T]] = {
        OutputStructure.fromDataset[T, DD, SS]
      }

      override def shapeStructure: ShapeStructure[Shape] = {
        ShapeStructure.fromOutput[Variant]
      }

      override def sizeFromOutput(output: Dataset[T]): Int = {
        1
      }

      override def shape(output: Dataset[T]): Shape = {
        Shape()
      }

      override def decodeShape(
          output: Dataset[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes.tail)
      }

      override def map(
          value: Dataset[T],
          shape: Option[Shape],
          converter: OutputStructure.Converter
      ): Dataset[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: OutputToShape[T]
  ): OutputToShape.Aux[Option[T], Option[ev.S]] = {
    new OutputToShape[Option[T]] {
      override type S = Option[ev.S]

      override def outputStructure: OutputStructure[Option[T]] = {
        OutputStructure.fromOption[T](ev.outputStructure)
      }

      override def shapeStructure: ShapeStructure[Option[ev.S]] = {
        ShapeStructure.fromOption[ev.S](ev.shapeStructure)
      }

      override def sizeFromOutput(output: Option[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def shape(output: Option[T]): Option[ev.S] = {
        output.map(o => ev.shape(o))
      }

      override def decodeShape(
          output: Option[T],
          shapes: Seq[Shape]
      ): (Option[ev.S], Seq[Shape]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeShape(o, shapes)
            (Some(result), remaining)
          case None => (None, shapes)
        }
      }

      override def map(
          value: Option[T],
          shape: Option[Option[ev.S]],
          converter: OutputStructure.Converter
      ): Option[T] = {
        (value, shape) match {
          case (Some(v), Some(s)) => Some(ev.map(v, s, converter))
          case _ => None
        }
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: OutputToShape[T]
  ): OutputToShape.Aux[Seq[T], Seq[ev.S]] = {
    new OutputToShape[Seq[T]] {
      override type S = Seq[ev.S]

      override def outputStructure: OutputStructure[Seq[T]] = {
        OutputStructure.fromSeq[T](ev.outputStructure)
      }

      override def shapeStructure: ShapeStructure[Seq[ev.S]] = {
        ShapeStructure.fromSeq[ev.S](ev.shapeStructure)
      }

      override def sizeFromOutput(output: Seq[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def shape(output: Seq[T]): Seq[ev.S] = {
        output.map(o => ev.shape(o))
      }

      override def decodeShape(
          output: Seq[T],
          shapes: Seq[Shape]
      ): (Seq[ev.S], Seq[Shape]) = {
        val n = sizeFromOutput(output)
        (output
            .zip(Collections.segment(shapes.take(n), output.map(ev.sizeFromOutput)))
            .map(f => ev.decodeShape(f._1, f._2)._1), shapes.drop(n))
      }

      override def map(
          value: Seq[T],
          shape: Option[Seq[ev.S]],
          converter: OutputStructure.Converter
      ): Seq[T] = {
        val shapes = shape.map(_.map(Option(_))).getOrElse(value.map(_ => None))
        value.zip(shapes).map(p => ev.map(p._1, p._2, converter))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: OutputToShape[T]
  ): OutputToShape.Aux[Map[K, T], Map[K, ev.S]] = {
    new OutputToShape[Map[K, T]] {
      override type S = Map[K, ev.S]

      override def outputStructure: OutputStructure[Map[K, T]] = {
        OutputStructure.fromMap[K, T](ev.outputStructure)
      }

      override def shapeStructure: ShapeStructure[Map[K, ev.S]] = {
        ShapeStructure.fromMap[K, ev.S](ev.shapeStructure)
      }

      override def sizeFromOutput(output: Map[K, T]): Int = {
        output.values.map(ev.sizeFromOutput).sum
      }

      override def shape(output: Map[K, T]): Map[K, ev.S] = {
        output.mapValues(o => ev.shape(o))
      }

      override def decodeShape(
          output: Map[K, T],
          shapes: Seq[Shape]
      ): (Map[K, ev.S], Seq[Shape]) = {
        val n = sizeFromOutput(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(shapes.take(n), output.values.map(ev.sizeFromOutput).toSeq))
              .map(f => ev.decodeShape(f._1, f._2)._1)).toMap, shapes.drop(n))
      }

      override def map(
          value: Map[K, T],
          shape: Option[Map[K, ev.S]],
          converter: OutputStructure.Converter
      ): Map[K, T] = {
        val shapes = shape.map(_.mapValues(Option(_))).getOrElse(value.mapValues(_ => None))
        (value.keys ++ shapes.keys).map(k => k -> ev.map(value(k), shapes(k), converter)).toMap
      }
    }
  }

  implicit val fromHNil: OutputToShape.Aux[HNil, HNil] = {
    new OutputToShape[HNil] {
      override type S = HNil

      override def outputStructure: OutputStructure[HNil] = {
        OutputStructure[HNil]
      }

      override def shapeStructure: ShapeStructure[HNil] = {
        ShapeStructure.fromHNil
      }

      override def sizeFromOutput(output: HNil): Int = {
        0
      }

      override def shape(output: HNil): HNil = {
        HNil
      }

      override def decodeShape(
          output: HNil,
          shapes: Seq[Shape]
      ): (HNil, Seq[Shape]) = {
        (HNil, shapes)
      }

      override def map(
          value: HNil,
          shape: Option[HNil],
          converter: OutputStructure.Converter
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HS, TT <: HList, TS <: HList](implicit
      evH: Strict[OutputToShape.Aux[HT, HS]],
      evT: Strict[OutputToShape.Aux[TT, TS]]
  ): OutputToShape.Aux[HT :: TT, HS :: TS] = {
    new OutputToShape[HT :: TT] {
      override type S = HS :: TS

      override def outputStructure: OutputStructure[HT :: TT] = {
        implicit val evOutputToShapeH: OutputStructure[HT] = evH.value.outputStructure
        implicit val evOutputToShapeT: OutputStructure[TT] = evT.value.outputStructure
        OutputStructure[HT :: TT]
      }

      override def shapeStructure: ShapeStructure[HS :: TS] = {
        ShapeStructure.fromHList[HS, TS](evH.value.shapeStructure, evT.value.shapeStructure)
      }

      override def sizeFromOutput(output: HT :: TT): Int = {
        evH.value.sizeFromOutput(output.head) + evT.value.sizeFromOutput(output.tail)
      }

      override def shape(output: HT :: TT): HS :: TS = {
        evH.value.shape(output.head) :: evT.value.shape(output.tail)
      }

      override def decodeShape(
          output: HT :: TT,
          shapes: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = evH.value.decodeShape(output.head, shapes)
        val (tailOut, tailRemaining) = evT.value.decodeShape(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def map(
          value: HT :: TT,
          shape: Option[HS :: TS],
          converter: OutputStructure.Converter
      ): HT :: TT = {
        evH.value.map(value.head, shape.map(_.head), converter) ::
            evT.value.map(value.tail, shape.map(_.tail), converter)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PS <: Product, HT <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[OutputToShape.Aux[HT, HS]],
      tuplerS: Tupler.Aux[HS, PS],
      genS: Generic.Aux[PS, HS]
  ): OutputToShape.Aux[PT, PS] = {
    new OutputToShape[PT] {
      override type S = PS

      override def outputStructure: OutputStructure[PT] = {
        implicit val evOutputToShapeT: OutputStructure[HT] = evT.value.outputStructure
        OutputStructure[PT]
      }

      override def shapeStructure: ShapeStructure[PS] = {
        ShapeStructure.fromProduct[PS, HS](genS, evT.value.shapeStructure)
      }

      override def sizeFromOutput(output: PT): Int = {
        evT.value.sizeFromOutput(genT.to(output))
      }

      override def shape(output: PT): PS = {
        tuplerS(evT.value.shape(genT.to(output)))
      }

      override def decodeShape(
          output: PT,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evT.value.decodeShape(genT.to(output), shapes)
        (genS.from(out), remaining)
      }

      override def map(
          value: PT,
          shape: Option[PS],
          converter: OutputStructure.Converter
      ): PT = {
        genT.from(evT.value.map(genT.to(value), shape.map(genS.to), converter))
      }
    }
  }
}
