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
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
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
  *   - Single [[Output]], [[OutputIndexedSlices]], [[SparseOutput]] object.
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
  type DataTypes
  type Shapes

  def outputDataTypes(data: T): DataTypes
  def outputShapes(data: T): Shapes
  def flattenedOutputs(data: T): Seq[Output]
  def numberOfOutputs(dataTypes: DataTypes): Int

  def unflattenOutputs(dataTypes: DataTypes, s: Seq[Output]): T = segmentOutputs(dataTypes, s)._1
  def unflattenDataTypes(dataTypes: DataTypes, s: Seq[DataType]): DataTypes = segmentDataTypes(dataTypes, s)._1
  def unflattenShapes(dataTypes: DataTypes, s: Seq[Shape]): Shapes = segmentShapes(dataTypes, s)._1

  def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (T, Seq[Output])
  def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType])
  def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape])
}

object Data {
  private[io] def process[T, D, S](data: T)(implicit ev: Aux[T, D, S]): (
      Seq[Output], Seq[DataType], Seq[Shape], Seq[Output] => T, Seq[DataType] => D, Seq[Shape] => S) = {
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

  type Aux[T, D, S] = Data[T] {type DataTypes = D; type Shapes = S}

  def apply[T, D, S](implicit ev: Aux[T, D, S]): Aux[T, D, S] = ev

  implicit val outputData: Aux[Output, DataType, Shape] = new Data[Output] {
    override type DataTypes = DataType
    override type Shapes = Shape

    override def outputDataTypes(data: Output): DataType = data.dataType
    override def outputShapes(data: Output): Shape = data.shape
    override def flattenedOutputs(data: Output): Seq[Output] = Seq(data)
    override def numberOfOutputs(dataTypes: DataType): Int = 1
    override def segmentOutputs(dataTypes: DataType, s: Seq[Output]): (Output, Seq[Output]) = (s.head, s.tail)
    override def segmentDataTypes(dataTypes: DataType, s: Seq[DataType]): (DataType, Seq[DataType]) = (s.head, s.tail)
    override def segmentShapes(dataTypes: DataType, s: Seq[Shape]): (Shape, Seq[Shape]) = (s.head, s.tail)
  }

  implicit def dataArray[T: ClassTag, D: ClassTag, S: ClassTag](implicit
      ev: Aux[T, D, S]
  ): Aux[Array[T], Array[D], Array[S]] = {
    new Data[Array[T]] {
      override type DataTypes = Array[D]
      override type Shapes = Array[S]

      override def outputDataTypes(data: Array[T]): Array[D] = data.map(ev.outputDataTypes)
      override def outputShapes(data: Array[T]): Array[S] = data.map(ev.outputShapes)
      override def flattenedOutputs(data: Array[T]): Seq[Output] = data.flatMap(ev.flattenedOutputs).toSeq
      override def numberOfOutputs(dataTypes: Array[D]): Int = dataTypes.map(ev.numberOfOutputs).sum

      override def segmentOutputs(dataTypes: Array[D], s: Seq[Output]): (Array[T], Seq[Output]) = {
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
    }
  }

  implicit def dataSeq[T, D, S, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, D, S],
      cbfTD: CanBuildFrom[CC[T], D, CC[D]],
      cbfTS: CanBuildFrom[CC[T], S, CC[S]],
      cbfDT: CanBuildFrom[CC[D], T, CC[T]],
      cbfDD: CanBuildFrom[CC[D], D, CC[D]],
      cbfDS: CanBuildFrom[CC[D], S, CC[S]]
  ): Aux[CC[T], CC[D], CC[S]] = {
    new Data[CC[T]] {
      override type DataTypes = CC[D]
      override type Shapes = CC[S]

      override def outputDataTypes(data: CC[T]): CC[D] = data.map(ev.outputDataTypes).to[CC](cbfTD)
      override def outputShapes(data: CC[T]): CC[S] = data.map(ev.outputShapes).to[CC](cbfTS)
      override def flattenedOutputs(data: CC[T]): Seq[Output] = data.flatMap(ev.flattenedOutputs).toSeq
      override def numberOfOutputs(dataTypes: CC[D]): Int = dataTypes.map(ev.numberOfOutputs).sum

      override def segmentOutputs(dataTypes: CC[D], s: Seq[Output]): (CC[T], Seq[Output]) = {
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
    }
  }

  implicit def dataMap[K, T, D, S, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[T, D, S],
  ): Aux[CC[K, T], Map[K, D], Map[K, S]] = {
    new Data[CC[K, T]] {
      // TODO: [DATASETS] Return CC type instead of Map.
      // TODO: [DATASETS] Make sure key-value pairs order is handled correctly here.
      override type DataTypes = Map[K, D]
      override type Shapes = Map[K, S]

      override def outputDataTypes(data: CC[K, T]): Map[K, D] = data.mapValues(ev.outputDataTypes)
      override def outputShapes(data: CC[K, T]): Map[K, S] = data.mapValues(ev.outputShapes)
      override def flattenedOutputs(data: CC[K, T]): Seq[Output] = data.values.flatMap(ev.flattenedOutputs).toSeq
      override def numberOfOutputs(dataTypes: Map[K, D]): Int = dataTypes.values.map(ev.numberOfOutputs).sum

      override def segmentOutputs(dataTypes: Map[K, D], s: Seq[Output]): (CC[K, T], Seq[Output]) = {
        val n = numberOfOutputs(dataTypes)
        // TODO: [DATASETS] !!! Fix this hacky solution for the return type.
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.numberOfOutputs).toSeq))
              .map(f => ev.unflattenOutputs(f._1, f._2))).toMap.asInstanceOf[CC[K, T]], s.drop(n))
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
    }
  }

  implicit val hnil: Aux[HNil, HNil, HNil] = new Data[HNil] {
    override type DataTypes = HNil
    override type Shapes = HNil

    override def outputDataTypes(data: HNil): HNil = HNil
    override def outputShapes(data: HNil): HNil = HNil
    override def flattenedOutputs(data: HNil): Seq[Output] = Seq.empty
    override def numberOfOutputs(dataTypes: HNil): Int = 0

    override def segmentOutputs(dataTypes: HNil, s: Seq[Output]): (HNil, Seq[Output]) = (HNil, s)
    override def segmentDataTypes(dataTypes: HNil, s: Seq[DataType]): (HNil, Seq[DataType]) = (HNil, s)
    override def segmentShapes(dataTypes: HNil, s: Seq[Shape]): (HNil, Seq[Shape]) = (HNil, s)
  }

  implicit def recursiveConstructor[HT, HD, HS, TT <: HList, TD <: HList, TS <: HList](implicit
      dataHead: Lazy[Aux[HT, HD, HS]],
      dataTail: Aux[TT, TD, TS]
  ): Aux[HT :: TT, HD :: TD, HS :: TS] = new Data[HT :: TT] {
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

    override def numberOfOutputs(dataTypes: HD :: TD): Int = {
      dataHead.value.numberOfOutputs(dataTypes.head) + dataTail.numberOfOutputs(dataTypes.tail)
    }

    override def segmentOutputs(dataTypes: HD :: TD, s: Seq[Output]): (HT :: TT, Seq[Output]) = {
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
  }

  // This also covers `OutputIndexedSlices` and `SparseOutput` as they are case classes (i.e., products).
  implicit def productConstructor[PT <: Product, PD, PS, LT <: HList, LD <: HList, LS <: HList](implicit
      genT: Generic.Aux[PT, LT],
      genD: Generic.Aux[PD, LD],
      dataL: Lazy[Aux[LT, LD, LS]],
      tuplerT: Tupler.Aux[LT, PT],
      tuplerD: Tupler.Aux[LD, PD],
      tuplerS: Tupler.Aux[LS, PS]
  ): Aux[PT, PD, PS] = new Data[PT] {
    override type DataTypes = PD
    override type Shapes = PS

    override def outputDataTypes(data: PT): PD = tuplerD(dataL.value.outputDataTypes(genT.to(data)))
    override def outputShapes(data: PT): PS = tuplerS(dataL.value.outputShapes(genT.to(data)))
    override def flattenedOutputs(data: PT): Seq[Output] = dataL.value.flattenedOutputs(genT.to(data))
    override def numberOfOutputs(dataTypes: PD): Int = dataL.value.numberOfOutputs(genD.to(dataTypes))

    override def segmentOutputs(dataTypes: PD, s: Seq[Output]): (PT, Seq[Output]) = {
      val (out, remaining) = dataL.value.segmentOutputs(genD.to(dataTypes), s)
      (tuplerT(out), remaining)
    }

    override def segmentDataTypes(dataTypes: PD, s: Seq[DataType]): (PD, Seq[DataType]) = {
      val (out, remaining) = dataL.value.segmentDataTypes(genD.to(dataTypes), s)
      (tuplerD(out), remaining)
    }

    override def segmentShapes(dataTypes: PD, s: Seq[Shape]): (PS, Seq[Shape]) = {
      val (out, remaining) = dataL.value.segmentShapes(genD.to(dataTypes), s)
      (tuplerS(out), remaining)
    }
  }
}
