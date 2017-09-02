/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.ops.io

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.breakOut
import scala.collection.generic.CanBuildFrom
import scala.collection.{MapLike, SeqLike, mutable}
import scala.language.higherKinds
import scala.reflect.ClassTag

/** Data can be emitted by [[Dataset]]s (i.e., the element types of all [[Dataset]]s are [[Data]]).
  *
  * Currently supported data types are:
  *   - Single [[Tensor]].
  *   - Sequences of other [[Data]] (e.g., Seq`s, `List`s, etc.).
  *     - Sequences that are not homogeneous are not supported (e.g., `Seq(data1, Seq(data1, data2))`).
  *     - Note that, for that reason, even though `Seq(List(data1), List(data1, data2))` is supported,
  *       `Seq(Seq(data1), List(data1, data2))` is not.
  *     - A sequence containing both [[Output]]s and [[SparseOutput]]s, for example, is considered heterogeneous.
  *       For such cases, it is advisable to use tuples.
  *   - Arrays of other [[Data]].
  *   - Maps with arbitrary key types and [[Data]] value types.
  *   - Products of other [[Data]] (e.g., tuples).
  *     - Note that with tuples, heterogeneous types are supported, due to the tuple type being a kind of heterogeneous
  *       collection.
  * Internally, the data emitted by a [[Dataset]] will be de-duplicated to prevent redundant computation.
  *
  * This trait guarantees that the output data types and shapes of a [[Dataset]] will match the structure of the
  * corresponding data. For example, if a `Seq(List(data1), List(data1, data2))` is provided as a [[Dataset]] element
  * type, then the dataset output data types will have the following structure `Seq(List(type1), List(type1, type2))`,
  * and similarly for the output shapes.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Data[T] {
  type OutputType
  type DataTypes
  type Shapes

  def outputDataTypes(data: T): DataTypes
  def outputShapes(data: T): Shapes
  def flattenedOutputs(data: T): Seq[Output]
  def flattenedOutputDataTypes(dataTypes: DataTypes): Seq[DataType]
  def flattenedOutputShapes(shapes: Shapes): Seq[Shape]
  def numberOfOutputs(dataTypes: DataTypes): Int

  def unflattenOutputs(dataTypes: DataTypes, s: Seq[Output]): OutputType = segmentOutputs(dataTypes, s)._1
  def unflattenDataTypes(dataTypes: DataTypes, s: Seq[DataType]): DataTypes = segmentDataTypes(dataTypes, s)._1
  def unflattenShapes(dataTypes: DataTypes, s: Seq[Shape]): Shapes = segmentShapes(dataTypes, s)._1

  def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output])
  def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType])
  def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape])

  def dataToString(data: T): String
  def dataTypesToString(dataTypes: DataTypes): String
  def shapesToString(shapes: Shapes): String
}

object Data {
  private[io] def process[T, O, D, S](data: T)(implicit ev: Aux[T, O, D, S]): (
      Seq[Output], Seq[DataType], Seq[Shape], Seq[Output] => O, Seq[DataType] => D, Seq[Shape] => S) = {
    val flattenedOutputs = ev.flattenedOutputs(data)
    val (uniqueOutputs, indices) = Data.uniquifyOutputs(flattenedOutputs)
    val uniqueOutputDataTypes = uniqueOutputs.map(_.dataType)
    val uniqueOutputShapes = uniqueOutputs.map(_.shape)
    val outputDataTypes = ev.outputDataTypes(data)
    val unflattenOutputs = (o: Seq[Output]) => ev.unflattenOutputs(outputDataTypes, indices.map(o(_)))
    val unflattenDataTypes = (d: Seq[DataType]) => ev.unflattenDataTypes(outputDataTypes, indices.map(d(_)))
    val unflattenShapes = (s: Seq[Shape]) => ev.unflattenShapes(outputDataTypes, indices.map(s(_)))
    (uniqueOutputs, uniqueOutputDataTypes, uniqueOutputShapes, unflattenOutputs, unflattenDataTypes, unflattenShapes)
  }

  private[Data] def uniquifyOutputs(outputs: Seq[Output]): (Seq[Output], Seq[Int]) = {
    val uniqueOutputs = mutable.ArrayBuffer.empty[Output]
    val seenOutputs = mutable.Map.empty[Output, Int]
    val indices = outputs.map(f => seenOutputs.getOrElseUpdate(f, {uniqueOutputs += f; uniqueOutputs.length - 1}))
    (uniqueOutputs, indices)
  }

  type Aux[T, O, D, S] = Data[T] {
    type OutputType = O
    type DataTypes = D
    type Shapes = S
  }

  def apply[T, O, D, S](implicit ev: Aux[T, O, D, S]): Aux[T, O, D, S] = ev

  implicit val tensorData: Aux[Tensor, Output, DataType, Shape] = new Data[Tensor] {
    override type OutputType = Output
    override type DataTypes = DataType
    override type Shapes = Shape

    override def outputDataTypes(data: Tensor): DataType = data.dataType
    override def outputShapes(data: Tensor): Shape = data.shape
    override def flattenedOutputs(data: Tensor): Seq[Output] = Seq(data.toOutput)
    override def flattenedOutputDataTypes(dataTypes: DataType): Seq[DataType] = Seq(dataTypes)
    override def flattenedOutputShapes(shapes: Shape): Seq[Shape] = Seq(shapes)
    override def numberOfOutputs(dataTypes: DataType): Int = 1

    override def segmentOutputs(dataTypes: DataType, s: Seq[Output]): (Output, Seq[Output]) = (s.head, s.tail)
    override def segmentDataTypes(dataTypes: DataType, s: Seq[DataType]): (DataType, Seq[DataType]) = (s.head, s.tail)
    override def segmentShapes(dataTypes: DataType, s: Seq[Shape]): (Shape, Seq[Shape]) = (s.head, s.tail)

    override def dataToString(data: Tensor): String = data.toString
    override def dataTypesToString(dataTypes: DataType): String = dataTypes.toString
    override def shapesToString(shapes: Shape): String = shapes.toString
  }

  // TODO: [DATASETS] "output", "outputIndexedSlicesData", and "sparseOutputData".

  implicit def dataArray[T: ClassTag, O: ClassTag, D: ClassTag, S: ClassTag](implicit
      ev: Aux[T, O, D, S]
  ): Aux[Array[T], Array[O], Array[D], Array[S]] = {
    new Data[Array[T]] {
      override type OutputType = Array[O]
      override type DataTypes = Array[D]
      override type Shapes = Array[S]

      override def outputDataTypes(data: Array[T]): Array[D] = data.map(ev.outputDataTypes)
      override def outputShapes(data: Array[T]): Array[S] = data.map(ev.outputShapes)
      override def flattenedOutputs(data: Array[T]): Seq[Output] = data.flatMap(ev.flattenedOutputs).toSeq

      override def flattenedOutputDataTypes(dataTypes: Array[D]): Seq[DataType] = {
        dataTypes.flatMap(ev.flattenedOutputDataTypes).toSeq
      }

      override def flattenedOutputShapes(shapes: Array[S]): Seq[Shape] = shapes.flatMap(ev.flattenedOutputShapes).toSeq
      override def numberOfOutputs(dataTypes: Array[D]): Int = dataTypes.map(ev.numberOfOutputs).sum

      override def segmentOutputs(dataTypes: Array[D], s: Seq[Output]): (Array[O], Seq[Output]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes.zip(Collections.segment(s.take(n), dataTypes.map(ev.numberOfOutputs).toSeq))
            .map(f => ev.unflattenOutputs(f._1, f._2)), s.drop(n))
      }

      override def segmentDataTypes(dataTypes: Array[D], s: Seq[DataType]): (Array[D], Seq[DataType]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes.zip(Collections.segment(s.take(n), dataTypes.map(ev.numberOfOutputs).toSeq))
            .map(f => ev.unflattenDataTypes(f._1, f._2)), s.drop(n))
      }

      override def segmentShapes(dataTypes: Array[D], s: Seq[Shape]): (Array[S], Seq[Shape]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes.zip(Collections.segment(s.take(n), dataTypes.map(ev.numberOfOutputs).toSeq))
            .map(f => ev.unflattenShapes(f._1, f._2)), s.drop(n))
      }

      override def dataToString(data: Array[T]): String = {
        s"{${data.map(ev.dataToString).mkString(", ")}}"
      }

      override def dataTypesToString(dataTypes: Array[D]): String = {
        s"{${dataTypes.map(ev.dataTypesToString).mkString(", ")}}"
      }

      override def shapesToString(shapes: Array[S]): String = {
        s"{${shapes.map(ev.shapesToString).mkString(", ")}}"
      }
    }
  }

  implicit def dataSeq[T, O, D, S, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, O, D, S],
      cbfTD: CanBuildFrom[CC[T], D, CC[D]],
      cbfTS: CanBuildFrom[CC[T], S, CC[S]],
      cbfDT: CanBuildFrom[CC[D], O, CC[O]],
      cbfDD: CanBuildFrom[CC[D], D, CC[D]],
      cbfDS: CanBuildFrom[CC[D], S, CC[S]]
  ): Aux[CC[T], CC[O], CC[D], CC[S]] = {
    new Data[CC[T]] {
      override type OutputType = CC[O]
      override type DataTypes = CC[D]
      override type Shapes = CC[S]

      override def outputDataTypes(data: CC[T]): CC[D] = data.map(ev.outputDataTypes).to[CC](cbfTD)
      override def outputShapes(data: CC[T]): CC[S] = data.map(ev.outputShapes).to[CC](cbfTS)
      override def flattenedOutputs(data: CC[T]): Seq[Output] = data.flatMap(ev.flattenedOutputs).toSeq

      override def flattenedOutputDataTypes(dataTypes: CC[D]): Seq[DataType] = {
        dataTypes.flatMap(ev.flattenedOutputDataTypes).toSeq
      }

      override def flattenedOutputShapes(shapes: CC[S]): Seq[Shape] = shapes.flatMap(ev.flattenedOutputShapes).toSeq
      override def numberOfOutputs(dataTypes: CC[D]): Int = dataTypes.map(ev.numberOfOutputs).sum

      override def segmentOutputs(dataTypes: CC[D], s: Seq[Output]): (CC[O], Seq[Output]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes
            .zip(Collections.segment(s.take(n), dataTypes.map(ev.numberOfOutputs).toSeq))(breakOut)
            .map(f => ev.unflattenOutputs(f._1, f._2)).to[CC](cbfDT), s.drop(n))
      }

      override def segmentDataTypes(dataTypes: CC[D], s: Seq[DataType]): (CC[D], Seq[DataType]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes
            .zip(Collections.segment(s.take(n), dataTypes.map(ev.numberOfOutputs).toSeq))(breakOut)
            .map(f => ev.unflattenDataTypes(f._1, f._2)).to[CC](cbfDD), s.drop(n))
      }

      override def segmentShapes(dataTypes: CC[D], s: Seq[Shape]): (CC[S], Seq[Shape]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes
            .zip(Collections.segment(s.take(n), dataTypes.map(ev.numberOfOutputs).toSeq))(breakOut)
            .map(f => ev.unflattenShapes(f._1, f._2)).to[CC](cbfDS), s.drop(n))
      }

      override def dataToString(data: CC[T]): String = {
        s"[${data.map(ev.dataToString).mkString(", ")}]"
      }

      override def dataTypesToString(dataTypes: CC[D]): String = {
        s"[${dataTypes.map(ev.dataTypesToString).mkString(", ")}]"
      }

      override def shapesToString(shapes: CC[S]): String = {
        s"[${shapes.map(ev.shapesToString).mkString(", ")}]"
      }
    }
  }

  implicit def dataMap[K, T, O, D, S, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[T, O, D, S]
  ): Aux[CC[K, T], CC[K, O], Map[K, D], Map[K, S]] = {
    new Data[CC[K, T]] {
      // TODO: [DATASETS] Return CC type instead of Map.
      // TODO: [DATASETS] Make sure key-value pairs order is handled correctly here.
      override type OutputType = CC[K, O]
      override type DataTypes = Map[K, D]
      override type Shapes = Map[K, S]

      override def outputDataTypes(data: CC[K, T]): Map[K, D] = data.mapValues(ev.outputDataTypes)
      override def outputShapes(data: CC[K, T]): Map[K, S] = data.mapValues(ev.outputShapes)
      override def flattenedOutputs(data: CC[K, T]): Seq[Output] = data.values.flatMap(ev.flattenedOutputs).toSeq

      override def flattenedOutputDataTypes(dataTypes: Map[K, D]): Seq[DataType] = {
        dataTypes.values.flatMap(ev.flattenedOutputDataTypes).toSeq
      }

      override def flattenedOutputShapes(shapes: Map[K, S]): Seq[Shape] = {
        shapes.values.flatMap(ev.flattenedOutputShapes).toSeq
      }

      override def numberOfOutputs(dataTypes: Map[K, D]): Int = dataTypes.values.map(ev.numberOfOutputs).sum

      override def segmentOutputs(dataTypes: Map[K, D], s: Seq[Output]): (CC[K, O], Seq[Output]) = {
        val n = numberOfOutputs(dataTypes)
        // TODO: [DATASETS] !!! Fix this hacky solution for the return type.
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.numberOfOutputs).toSeq))
              .map(f => ev.unflattenOutputs(f._1, f._2))).toMap.asInstanceOf[CC[K, O]], s.drop(n))
      }

      override def segmentDataTypes(dataTypes: Map[K, D], s: Seq[DataType]): (Map[K, D], Seq[DataType]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.numberOfOutputs).toSeq))
              .map(f => ev.unflattenDataTypes(f._1, f._2))).toMap, s.drop(n))
      }

      override def segmentShapes(dataTypes: Map[K, D], s: Seq[Shape]): (Map[K, S], Seq[Shape]) = {
        val n = numberOfOutputs(dataTypes)
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.numberOfOutputs).toSeq))
              .map(f => ev.unflattenShapes(f._1, f._2))).toMap, s.drop(n))
      }

      override def dataToString(data: CC[K, T]): String = {
        s"{${data.map(d => s"${d._1.toString} -> ${ev.dataToString(d._2)}").mkString(", ")}}"
      }

      override def dataTypesToString(dataTypes: Map[K, D]): String = {
        s"{${dataTypes.map(d => s"${d._1.toString} -> ${ev.dataTypesToString(d._2)}").mkString(", ")}}"
      }

      override def shapesToString(shapes: Map[K, S]): String = {
        s"{${shapes.map(d => s"${d._1.toString} -> ${ev.shapesToString(d._2)}").mkString(", ")}}"
      }
    }
  }

  implicit val hnil: Aux[HNil, HNil, HNil, HNil] = new Data[HNil] {
    override type OutputType = HNil
    override type DataTypes = HNil
    override type Shapes = HNil

    override def outputDataTypes(data: HNil): HNil = HNil
    override def outputShapes(data: HNil): HNil = HNil
    override def flattenedOutputs(data: HNil): Seq[Output] = Seq.empty
    override def flattenedOutputDataTypes(dataTypes: HNil): Seq[DataType] = Seq.empty
    override def flattenedOutputShapes(shapes: HNil): Seq[Shape] = Seq.empty
    override def numberOfOutputs(dataTypes: HNil): Int = 0

    override def segmentOutputs(dataTypes: HNil, s: Seq[Output]): (HNil, Seq[Output]) = (HNil, s)
    override def segmentDataTypes(dataTypes: HNil, s: Seq[DataType]): (HNil, Seq[DataType]) = (HNil, s)
    override def segmentShapes(dataTypes: HNil, s: Seq[Shape]): (HNil, Seq[Shape]) = (HNil, s)

    override def dataToString(data: HNil): String = ""
    override def dataTypesToString(dataTypes: HNil): String = ""
    override def shapesToString(shapes: HNil): String = ""
  }

  implicit def recursiveConstructor[HT, HO, HD, HS, TT <: HList, TO <: HList, TD <: HList, TS <: HList](implicit
      dataHead: Lazy[Aux[HT, HO, HD, HS]],
      dataTail: Aux[TT, TO, TD, TS]
  ): Aux[HT :: TT, HO :: TO, HD :: TD, HS :: TS] = new Data[HT :: TT] {
    override type OutputType = HO :: TO
    override type DataTypes = HD :: TD
    override type Shapes = HS :: TS

    override def outputDataTypes(data: HT :: TT): HD :: TD = {
      dataHead.value.outputDataTypes(data.head) :: dataTail.outputDataTypes(data.tail)
    }

    override def outputShapes(data: HT :: TT): HS :: TS = {
      dataHead.value.outputShapes(data.head) :: dataTail.outputShapes(data.tail)
    }

    override def flattenedOutputs(data: HT :: TT): Seq[Output] = {
      dataHead.value.flattenedOutputs(data.head) ++ dataTail.flattenedOutputs(data.tail)
    }

    override def flattenedOutputDataTypes(dataTypes: HD :: TD): Seq[DataType] = {
      dataHead.value.flattenedOutputDataTypes(dataTypes.head) ++ dataTail.flattenedOutputDataTypes(dataTypes.tail)
    }

    override def flattenedOutputShapes(shapes: HS :: TS): Seq[Shape] = {
      dataHead.value.flattenedOutputShapes(shapes.head) ++ dataTail.flattenedOutputShapes(shapes.tail)
    }

    override def numberOfOutputs(dataTypes: HD :: TD): Int = {
      dataHead.value.numberOfOutputs(dataTypes.head) + dataTail.numberOfOutputs(dataTypes.tail)
    }

    override def segmentOutputs(dataTypes: HD :: TD, s: Seq[Output]): (HO :: TO, Seq[Output]) = {
      val (headOut, headRemaining) = dataHead.value.segmentOutputs(dataTypes.head, s)
      val (tailOut, tailRemaining) = dataTail.segmentOutputs(dataTypes.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def segmentDataTypes(dataTypes: HD :: TD, s: Seq[DataType]): (HD :: TD, Seq[DataType]) = {
      val (headOut, headRemaining) = dataHead.value.segmentDataTypes(dataTypes.head, s)
      val (tailOut, tailRemaining) = dataTail.segmentDataTypes(dataTypes.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def segmentShapes(dataTypes: HD :: TD, s: Seq[Shape]): (HS :: TS, Seq[Shape]) = {
      val (headOut, headRemaining) = dataHead.value.segmentShapes(dataTypes.head, s)
      val (tailOut, tailRemaining) = dataTail.segmentShapes(dataTypes.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def dataToString(data: HT :: TT): String = {
      val headPart = dataHead.value.dataToString(data.head)
      val tailPart = dataTail.dataToString(data.tail)
      if (headPart == "")
        tailPart
      else if (tailPart == "")
        headPart
      else
        s"$headPart, $tailPart"
    }

    override def dataTypesToString(dataTypes: HD :: TD): String = {
      val headPart = dataHead.value.dataTypesToString(dataTypes.head)
      val tailPart = dataTail.dataTypesToString(dataTypes.tail)
      if (headPart == "")
        tailPart
      else if (tailPart == "")
        headPart
      else
        s"$headPart, $tailPart"
    }

    override def shapesToString(shapes: HS :: TS): String = {
      val headPart = dataHead.value.shapesToString(shapes.head)
      val tailPart = dataTail.shapesToString(shapes.tail)
      if (headPart == "")
        tailPart
      else if (tailPart == "")
        headPart
      else
        s"$headPart, $tailPart"
    }
  }

  // This also covers `OutputIndexedSlices` and `SparseOutput` as they are case classes (i.e., products).
  implicit def productConstructor[
  PT <: Product, PO <: Product, PD <: Product, PS <: Product,
  LT <: HList, LO <: HList, LD <: HList, LS <: HList](implicit
      genT: Generic.Aux[PT, LT],
      genD: Generic.Aux[PD, LD],
      genS: Generic.Aux[PS, LS],
      dataL: Lazy[Aux[LT, LO, LD, LS]],
      tuplerO: Tupler.Aux[LO, PO],
      tuplerD: Tupler.Aux[LD, PD],
      tuplerS: Tupler.Aux[LS, PS]
  ): Aux[PT, PO, PD, PS] = new Data[PT] {
    override type OutputType = PO
    override type DataTypes = PD
    override type Shapes = PS

    override def outputDataTypes(data: PT): PD = tuplerD(dataL.value.outputDataTypes(genT.to(data)))
    override def outputShapes(data: PT): PS = tuplerS(dataL.value.outputShapes(genT.to(data)))
    override def flattenedOutputs(data: PT): Seq[Output] = dataL.value.flattenedOutputs(genT.to(data))

    override def flattenedOutputDataTypes(dataTypes: PD): Seq[DataType] = {
      dataL.value.flattenedOutputDataTypes(genD.to(dataTypes))
    }

    override def flattenedOutputShapes(shapes: PS): Seq[Shape] = {
      dataL.value.flattenedOutputShapes(genS.to(shapes))
    }

    override def numberOfOutputs(dataTypes: PD): Int = dataL.value.numberOfOutputs(genD.to(dataTypes))

    override def segmentOutputs(dataTypes: PD, s: Seq[Output]): (PO, Seq[Output]) = {
      val (out, remaining) = dataL.value.segmentOutputs(genD.to(dataTypes), s)
      (tuplerO(out), remaining)
    }

    override def segmentDataTypes(dataTypes: PD, s: Seq[DataType]): (PD, Seq[DataType]) = {
      val (out, remaining) = dataL.value.segmentDataTypes(genD.to(dataTypes), s)
      (tuplerD(out), remaining)
    }

    override def segmentShapes(dataTypes: PD, s: Seq[Shape]): (PS, Seq[Shape]) = {
      val (out, remaining) = dataL.value.segmentShapes(genD.to(dataTypes), s)
      (tuplerS(out), remaining)
    }

    override def dataToString(data: PT): String = dataL.value.dataToString(genT.to(data))
    override def dataTypesToString(dataTypes: PD): String = dataL.value.dataTypesToString(genD.to(dataTypes))
    override def shapesToString(shapes: PS): String = dataL.value.shapesToString(genS.to(shapes))
  }
}
