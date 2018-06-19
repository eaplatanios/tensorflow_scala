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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.types.{DataType, INT64}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.generic.CanBuildFrom
import scala.collection.{MapLike, SeqLike, breakOut, mutable}
import scala.language.higherKinds
import scala.reflect.ClassTag

// TODO: Separate into readers and transformations.
// TODO: paddedBatchAndDropRemainder
// TODO: denseToSparseBatch
// TODO: listFiles

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

  def size(dataTypes: DataTypes): Int

  def dataTypesFromT(data: T): DataTypes
  def dataTypesFromO(data: OutputType): DataTypes

  def shapesFromT(data: T): Shapes
  def shapesFromO(data: OutputType): Shapes

  def flattenedTensors(data: T): Seq[Tensor]

  def flattenedOutputsFromT(data: T): Seq[Output]
  def flattenedOutputsFromO(data: OutputType): Seq[Output]

  def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType]
  def flattenedShapes(shapes: Shapes): Seq[Shape]

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
  private[io] def process[T, O, D, S](data: T)(implicit
      ev: Aux[T, O, D, S]
  ): (Seq[Output], Seq[DataType], Seq[Shape], Seq[Output] => O, Seq[DataType] => D, Seq[Shape] => S) = {
    val flattenedOutputs = ev.flattenedOutputsFromT(data)
    val (uniqueOutputs, indices) = Data.uniquifyOutputs(flattenedOutputs)
    val uniqueDataTypes = uniqueOutputs.map(_.dataType)
    val uniqueShapes = uniqueOutputs.map(_.shape)
    val dataTypes = ev.dataTypesFromT(data)
    val unflattenOutputs = (o: Seq[Output]) => ev.unflattenOutputs(dataTypes, indices.map(o(_)))
    val unflattenDataTypes = (d: Seq[DataType]) => ev.unflattenDataTypes(dataTypes, indices.map(d(_)))
    val unflattenShapes = (s: Seq[Shape]) => ev.unflattenShapes(dataTypes, indices.map(s(_)))
    (uniqueOutputs, uniqueDataTypes, uniqueShapes, unflattenOutputs, unflattenDataTypes, unflattenShapes)
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

  implicit def tensorData[D <: DataType]: Aux[Tensor, Output, D, Shape] = new Data[Tensor] {
    override type OutputType = Output
    override type DataTypes = D
    override type Shapes = Shape

    override def size(dataTypes: DataTypes): Int = 1

    override def dataTypesFromT(data: Tensor): D = data.dataType.asInstanceOf[D]
    override def dataTypesFromO(data: Output): D = data.dataType.asInstanceOf[D]

    override def shapesFromT(data: Tensor): Shape = data.shape
    override def shapesFromO(data: Output): Shape = data.shape

    override def flattenedTensors(data: Tensor): Seq[Tensor] = Seq(data)

    override def flattenedOutputsFromT(data: Tensor): Seq[Output] = Seq(data.toOutput)
    override def flattenedOutputsFromO(data: Output): Seq[Output] = Seq(data)

    override def flattenedDataTypes(dataTypes: D): Seq[DataType] = Seq(dataTypes: DataType)
    override def flattenedShapes(shapes: Shape): Seq[Shape] = Seq(shapes)

    override def segmentOutputs(dataTypes: D, s: Seq[Output]): (Output, Seq[Output]) = (s.head, s.tail)
    override def segmentDataTypes(dataTypes: D, s: Seq[DataType]): (D, Seq[DataType]) = (s.head.asInstanceOf[D], s.tail)
    override def segmentShapes(dataTypes: D, s: Seq[Shape]): (Shape, Seq[Shape]) = (s.head, s.tail)

    override def dataToString(data: Tensor): String = data.toString
    override def dataTypesToString(dataTypes: D): String = dataTypes.toString
    override def shapesToString(shapes: Shape): String = shapes.toString
  }

  implicit val tensorIndexedSlicesData: Aux[
      TensorIndexedSlices, OutputIndexedSlices, (INT64, DataType, INT64), (Shape, Shape, Shape)] = {
    new Data[TensorIndexedSlices] {
      override type OutputType = OutputIndexedSlices
      override type DataTypes = (INT64, DataType, INT64)
      override type Shapes = (Shape, Shape, Shape)

      override def size(dataTypes: DataTypes): Int = 3

      override def dataTypesFromT(data: TensorIndexedSlices): DataTypes = (INT64, data.dataType, INT64)
      override def dataTypesFromO(data: OutputIndexedSlices): DataTypes = (INT64, data.dataType, INT64)

      override def shapesFromT(data: TensorIndexedSlices): Shapes = {
        val indicesShape = data.indices.shape
        val denseShapeShape = data.denseShape.shape
        val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
        (Shape(-1, rank), Shape(-1), Shape(rank))
      }

      override def shapesFromO(data: OutputIndexedSlices): Shapes = {
        val indicesShape = data.indices.shape
        val denseShapeShape = data.denseShape.shape
        val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
        (Shape(-1, rank), Shape(-1), Shape(rank))
      }

      override def flattenedTensors(data: TensorIndexedSlices): Seq[Tensor] = {
        Seq(data.indices, data.values, data.denseShape)
      }

      override def flattenedOutputsFromT(data: TensorIndexedSlices): Seq[Output] = {
        flattenedTensors(data).map(_.toOutput)
      }

      override def flattenedOutputsFromO(data: OutputIndexedSlices): Seq[Output] = {
        Seq(data.indices, data.values, data.denseShape)
      }

      override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = {
        Seq(dataTypes._1, dataTypes._2, dataTypes._3)
      }

      override def flattenedShapes(shapes: Shapes): Seq[Shape] = Seq(shapes._1, shapes._2, shapes._3)

      override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputIndexedSlices, Seq[Output]) = {
        (OutputIndexedSlices(s(0), s(1), s(2)), s.drop(3))
      }

      override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
        ((s(0).asInstanceOf[INT64], s(1), s(2).asInstanceOf[INT64]), s.drop(3))
      }

      override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
        ((s(0), s(1), s(2)), s.drop(3))
      }

      override def dataToString(data: TensorIndexedSlices): String = data.toString
      override def dataTypesToString(dataTypes: DataTypes): String = s"${dataTypes._1}:${dataTypes._2}:${dataTypes._3}"
      override def shapesToString(shapes: Shapes): String = s"${shapes._1}:${shapes._2}:${shapes._3}"
    }
  }

  implicit val sparseTensorData: Aux[
      SparseTensor, SparseOutput, (INT64, DataType, INT64), (Shape, Shape, Shape)] = new Data[SparseTensor] {
    override type OutputType = SparseOutput
    override type DataTypes = (INT64, DataType, INT64)
    override type Shapes = (Shape, Shape, Shape)

    override def size(dataTypes: DataTypes): Int = 3

    override def dataTypesFromT(data: SparseTensor): DataTypes = (INT64, data.dataType, INT64)
    override def dataTypesFromO(data: SparseOutput): DataTypes = (INT64, data.dataType, INT64)

    override def shapesFromT(data: SparseTensor): Shapes = {
      val indicesShape = data.indices.shape
      val denseShapeShape = data.denseShape.shape
      val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
      (Shape(-1, rank), Shape(-1), Shape(rank))
    }

    override def shapesFromO(data: SparseOutput): Shapes = {
      val indicesShape = data.indices.shape
      val denseShapeShape = data.denseShape.shape
      val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
      (Shape(-1, rank), Shape(-1), Shape(rank))
    }

    override def flattenedTensors(data: SparseTensor): Seq[Tensor] = Seq(data.indices, data.values, data.denseShape)

    override def flattenedOutputsFromT(data: SparseTensor): Seq[Output] = {
      flattenedTensors(data).map(_.toOutput)
    }

    override def flattenedOutputsFromO(data: SparseOutput): Seq[Output] = {
      Seq(data.indices, data.values, data.denseShape)
    }

    override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = {
      Seq(dataTypes._1, dataTypes._2, dataTypes._3)
    }

    override def flattenedShapes(shapes: Shapes): Seq[Shape] = Seq(shapes._1, shapes._2, shapes._3)

    override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = {
      (SparseOutput(s(0), s(1), s(2)), s.drop(3))
    }

    override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
      ((s(0).asInstanceOf[INT64], s(1), s(2).asInstanceOf[INT64]), s.drop(3))
    }

    override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
      ((s(0), s(1), s(2)), s.drop(3))
    }

    override def dataToString(data: SparseTensor): String = data.toString
    override def dataTypesToString(dataTypes: DataTypes): String = s"${dataTypes._1}, ${dataTypes._2}, ${dataTypes._3}"
    override def shapesToString(shapes: Shapes): String = s"${shapes._1}, ${shapes._2}, ${shapes._3}"
  }

  implicit def dataArray[T: ClassTag, O: ClassTag, D: ClassTag, S: ClassTag](implicit
      ev: Aux[T, O, D, S]
  ): Aux[Array[T], Array[O], Array[D], Array[S]] = {
    new Data[Array[T]] {
      override type OutputType = Array[O]
      override type DataTypes = Array[D]
      override type Shapes = Array[S]

      override def size(dataTypes: DataTypes): Int = dataTypes.map(ev.size).sum

      override def dataTypesFromT(data: Array[T]): DataTypes = data.map(ev.dataTypesFromT)
      override def dataTypesFromO(data: Array[O]): DataTypes = data.map(ev.dataTypesFromO)

      override def shapesFromT(data: Array[T]): Shapes = data.map(ev.shapesFromT)
      override def shapesFromO(data: Array[O]): Shapes = data.map(ev.shapesFromO)

      override def flattenedTensors(data: Array[T]): Seq[Tensor] = data.flatMap(ev.flattenedTensors).toSeq

      override def flattenedOutputsFromT(data: Array[T]): Seq[Output] = data.flatMap(ev.flattenedOutputsFromT).toSeq
      override def flattenedOutputsFromO(data: Array[O]): Seq[Output] = data.flatMap(ev.flattenedOutputsFromO).toSeq

      override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = {
        dataTypes.flatMap(ev.flattenedDataTypes).toSeq
      }

      override def flattenedShapes(shapes: Shapes): Seq[Shape] = shapes.flatMap(ev.flattenedShapes).toSeq

      override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = {
        val n = size(dataTypes)
        (dataTypes.zip(Collections.segment(s.take(n), dataTypes.map(ev.size).toSeq))
            .map(f => ev.unflattenOutputs(f._1, f._2)), s.drop(n))
      }

      override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
        val n = size(dataTypes)
        (dataTypes.zip(Collections.segment(s.take(n), dataTypes.map(ev.size).toSeq))
            .map(f => ev.unflattenDataTypes(f._1, f._2)), s.drop(n))
      }

      override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
        val n = size(dataTypes)
        (dataTypes.zip(Collections.segment(s.take(n), dataTypes.map(ev.size).toSeq))
            .map(f => ev.unflattenShapes(f._1, f._2)), s.drop(n))
      }

      override def dataToString(data: Array[T]): String = {
        s"{${data.map(ev.dataToString).mkString(", ")}}"
      }

      override def dataTypesToString(dataTypes: DataTypes): String = {
        s"{${dataTypes.map(ev.dataTypesToString).mkString(", ")}}"
      }

      override def shapesToString(shapes: Shapes): String = {
        s"{${shapes.map(ev.shapesToString).mkString(", ")}}"
      }
    }
  }

  implicit def dataSeq[T, O, D, S, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, O, D, S],
      cbfTD: CanBuildFrom[CC[T], D, CC[D]],
      cbfOD: CanBuildFrom[CC[O], D, CC[D]],
      cbfTS: CanBuildFrom[CC[T], S, CC[S]],
      cbfOS: CanBuildFrom[CC[O], S, CC[S]],
      cbfO: CanBuildFrom[Nothing, O, CC[O]],
      cbfD: CanBuildFrom[Nothing, D, CC[D]],
      cbfS: CanBuildFrom[Nothing, S, CC[S]]
  ): Aux[CC[T], CC[O], CC[D], CC[S]] = {
    new Data[CC[T]] {
      override type OutputType = CC[O]
      override type DataTypes = CC[D]
      override type Shapes = CC[S]

      override def size(dataTypes: DataTypes): Int = dataTypes.map(ev.size)(breakOut).sum

      override def dataTypesFromT(data: CC[T]): DataTypes = data.map(ev.dataTypesFromT)
      override def dataTypesFromO(data: CC[O]): DataTypes = data.map(ev.dataTypesFromO)

      override def shapesFromT(data: CC[T]): Shapes = data.map(ev.shapesFromT)
      override def shapesFromO(data: CC[O]): Shapes = data.map(ev.shapesFromO)

      override def flattenedTensors(data: CC[T]): Seq[Tensor] = data.flatMap(ev.flattenedTensors)(breakOut)

      override def flattenedOutputsFromT(data: CC[T]): Seq[Output] = data.flatMap(ev.flattenedOutputsFromT)(breakOut)
      override def flattenedOutputsFromO(data: CC[O]): Seq[Output] = data.flatMap(ev.flattenedOutputsFromO)(breakOut)

      override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = {
        dataTypes.flatMap(ev.flattenedDataTypes)(breakOut)
      }

      override def flattenedShapes(shapes: Shapes): Seq[Shape] = {
        shapes.flatMap(ev.flattenedShapes)(breakOut)
      }

      override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = {
        val n = size(dataTypes)
        (dataTypes
            .zip(Collections.segment(s.take(n), dataTypes.map(ev.size)(breakOut)))(breakOut)
            .map(f => ev.unflattenOutputs(f._1, f._2)).to[CC](cbfO), s.drop(n))
      }

      override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
        val n = size(dataTypes)
        (dataTypes
            .zip(Collections.segment(s.take(n), dataTypes.map(ev.size)(breakOut)))(breakOut)
            .map(f => ev.unflattenDataTypes(f._1, f._2)).to[CC](cbfD), s.drop(n))
      }

      override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
        val n = size(dataTypes)
        (dataTypes
            .zip(Collections.segment(s.take(n), dataTypes.map(ev.size)(breakOut)))(breakOut)
            .map(f => ev.unflattenShapes(f._1, f._2)).to[CC](cbfS), s.drop(n))
      }

      override def dataToString(data: CC[T]): String = {
        s"[${data.map(ev.dataToString)(breakOut).mkString(", ")}]"
      }

      override def dataTypesToString(dataTypes: DataTypes): String = {
        s"[${dataTypes.map(ev.dataTypesToString)(breakOut).mkString(", ")}]"
      }

      override def shapesToString(shapes: Shapes): String = {
        s"[${shapes.map(ev.shapesToString)(breakOut).mkString(", ")}]"
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

      override def size(dataTypes: DataTypes): Int = dataTypes.values.map(ev.size).sum

      override def dataTypesFromT(data: CC[K, T]): DataTypes = data.mapValues(ev.dataTypesFromT)
      override def dataTypesFromO(data: CC[K, O]): DataTypes = data.mapValues(ev.dataTypesFromO)

      override def shapesFromT(data: CC[K, T]): Shapes = data.mapValues(ev.shapesFromT)
      override def shapesFromO(data: CC[K, O]): Shapes = data.mapValues(ev.shapesFromO)

      override def flattenedTensors(data: CC[K, T]): Seq[Tensor] = data.values.flatMap(ev.flattenedTensors).toSeq

      override def flattenedOutputsFromT(data: CC[K, T]): Seq[Output] = {
        data.values.flatMap(ev.flattenedOutputsFromT).toSeq
      }

      override def flattenedOutputsFromO(data: CC[K, O]): Seq[Output] = {
        data.values.flatMap(ev.flattenedOutputsFromO).toSeq
      }

      override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = {
        dataTypes.values.flatMap(ev.flattenedDataTypes).toSeq
      }

      override def flattenedShapes(shapes: Shapes): Seq[Shape] = {
        shapes.values.flatMap(ev.flattenedShapes).toSeq
      }

      override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = {
        val n = size(dataTypes)
        // TODO: [DATASETS] !!! Fix this hacky solution for the return type.
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.size).toSeq))
              .map(f => ev.unflattenOutputs(f._1, f._2))).toMap.asInstanceOf[CC[K, O]], s.drop(n))
      }

      override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
        val n = size(dataTypes)
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.size).toSeq))
              .map(f => ev.unflattenDataTypes(f._1, f._2))).toMap, s.drop(n))
      }

      override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
        val n = size(dataTypes)
        (dataTypes.keys.zip(
          dataTypes.values
              .zip(Collections.segment(s.take(n), dataTypes.values.map(ev.size).toSeq))
              .map(f => ev.unflattenShapes(f._1, f._2))).toMap, s.drop(n))
      }

      override def dataToString(data: CC[K, T]): String = {
        s"{${data.map(d => s"${d._1.toString} -> ${ev.dataToString(d._2)}").mkString(", ")}}"
      }

      override def dataTypesToString(dataTypes: DataTypes): String = {
        s"{${dataTypes.map(d => s"${d._1.toString} -> ${ev.dataTypesToString(d._2)}").mkString(", ")}}"
      }

      override def shapesToString(shapes: Shapes): String = {
        s"{${shapes.map(d => s"${d._1.toString} -> ${ev.shapesToString(d._2)}").mkString(", ")}}"
      }
    }
  }

  implicit val hnil: Aux[HNil, HNil, HNil, HNil] = new Data[HNil] {
    override type OutputType = HNil
    override type DataTypes = HNil
    override type Shapes = HNil

    override def size(dataTypes: DataTypes): Int = 0

    override def dataTypesFromT(data: HNil): DataTypes = HNil
    override def dataTypesFromO(data: HNil): DataTypes = HNil

    override def shapesFromT(data: HNil): Shapes = HNil
    override def shapesFromO(data: HNil): Shapes = HNil

    override def flattenedTensors(data: HNil): Seq[Tensor] = Seq.empty

    override def flattenedOutputsFromT(data: HNil): Seq[Output] = Seq.empty
    override def flattenedOutputsFromO(data: HNil): Seq[Output] = Seq.empty

    override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = Seq.empty
    override def flattenedShapes(shapes: Shapes): Seq[Shape] = Seq.empty

    override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = (HNil, s)
    override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = (HNil, s)
    override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = (HNil, s)

    override def dataToString(data: HNil): String = ""
    override def dataTypesToString(dataTypes: DataTypes): String = ""
    override def shapesToString(shapes: Shapes): String = ""
  }

  implicit def recursiveConstructor[HT, HO, HD, HS, TT <: HList, TO <: HList, TD <: HList, TS <: HList](implicit
      dataHead: Lazy[Aux[HT, HO, HD, HS]],
      dataTail: Aux[TT, TO, TD, TS]
  ): Aux[HT :: TT, HO :: TO, HD :: TD, HS :: TS] = new Data[HT :: TT] {
    override type OutputType = HO :: TO
    override type DataTypes = HD :: TD
    override type Shapes = HS :: TS

    override def size(dataTypes: DataTypes): Int = dataHead.value.size(dataTypes.head) + dataTail.size(dataTypes.tail)

    override def dataTypesFromT(data: HT :: TT): DataTypes = {
      dataHead.value.dataTypesFromT(data.head) :: dataTail.dataTypesFromT(data.tail)
    }

    override def dataTypesFromO(data: HO :: TO): DataTypes = {
      dataHead.value.dataTypesFromO(data.head) :: dataTail.dataTypesFromO(data.tail)
    }

    override def shapesFromT(data: HT :: TT): Shapes = {
      dataHead.value.shapesFromT(data.head) :: dataTail.shapesFromT(data.tail)
    }

    override def shapesFromO(data: HO :: TO): Shapes = {
      dataHead.value.shapesFromO(data.head) :: dataTail.shapesFromO(data.tail)
    }

    override def flattenedTensors(data: HT :: TT): Seq[Tensor] = {
      dataHead.value.flattenedTensors(data.head) ++ dataTail.flattenedTensors(data.tail)
    }

    override def flattenedOutputsFromT(data: HT :: TT): Seq[Output] = {
      dataHead.value.flattenedOutputsFromT(data.head) ++ dataTail.flattenedOutputsFromT(data.tail)
    }

    override def flattenedOutputsFromO(data: HO :: TO): Seq[Output] = {
      dataHead.value.flattenedOutputsFromO(data.head) ++ dataTail.flattenedOutputsFromO(data.tail)
    }

    override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = {
      dataHead.value.flattenedDataTypes(dataTypes.head) ++ dataTail.flattenedDataTypes(dataTypes.tail)
    }

    override def flattenedShapes(shapes: Shapes): Seq[Shape] = {
      dataHead.value.flattenedShapes(shapes.head) ++ dataTail.flattenedShapes(shapes.tail)
    }

    override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = {
      val (headOut, headRemaining) = dataHead.value.segmentOutputs(dataTypes.head, s)
      val (tailOut, tailRemaining) = dataTail.segmentOutputs(dataTypes.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
      val (headOut, headRemaining) = dataHead.value.segmentDataTypes(dataTypes.head, s)
      val (tailOut, tailRemaining) = dataTail.segmentDataTypes(dataTypes.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
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

    override def dataTypesToString(dataTypes: DataTypes): String = {
      val headPart = dataHead.value.dataTypesToString(dataTypes.head)
      val tailPart = dataTail.dataTypesToString(dataTypes.tail)
      if (headPart == "")
        tailPart
      else if (tailPart == "")
        headPart
      else
        s"$headPart, $tailPart"
    }

    override def shapesToString(shapes: Shapes): String = {
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
  implicit def productConstructor[PT, PO, PD, PS, HT <: HList, HO <: HList, HD <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      dataL: Aux[HT, HO, HD, HS],
      tuplerO: Tupler.Aux[HO, PO],
      tuplerD: Tupler.Aux[HD, PD],
      tuplerS: Tupler.Aux[HS, PS],
      genO: Generic.Aux[PO, HO],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS]
  ): Aux[PT, PO, PD, PS] = new Data[PT] {
    override type OutputType = PO
    override type DataTypes = PD
    override type Shapes = PS

    override def size(dataTypes: DataTypes): Int = dataL.size(genD.to(dataTypes))

    override def dataTypesFromT(data: PT): DataTypes = tuplerD(dataL.dataTypesFromT(genT.to(data)))
    override def dataTypesFromO(data: PO): DataTypes = tuplerD(dataL.dataTypesFromO(genO.to(data)))

    override def shapesFromT(data: PT): Shapes = tuplerS(dataL.shapesFromT(genT.to(data)))
    override def shapesFromO(data: PO): Shapes = tuplerS(dataL.shapesFromO(genO.to(data)))

    override def flattenedTensors(data: PT): Seq[Tensor] = dataL.flattenedTensors(genT.to(data))

    override def flattenedOutputsFromT(data: PT): Seq[Output] = dataL.flattenedOutputsFromT(genT.to(data))
    override def flattenedOutputsFromO(data: PO): Seq[Output] = dataL.flattenedOutputsFromO(genO.to(data))
    override def flattenedDataTypes(dataTypes: DataTypes): Seq[DataType] = dataL.flattenedDataTypes(genD.to(dataTypes))
    override def flattenedShapes(shapes: Shapes): Seq[Shape] = dataL.flattenedShapes(genS.to(shapes))

    override def segmentOutputs(dataTypes: DataTypes, s: Seq[Output]): (OutputType, Seq[Output]) = {
      val (out, remaining) = dataL.segmentOutputs(genD.to(dataTypes), s)
      (tuplerO(out), remaining)
    }

    override def segmentDataTypes(dataTypes: DataTypes, s: Seq[DataType]): (DataTypes, Seq[DataType]) = {
      val (out, remaining) = dataL.segmentDataTypes(genD.to(dataTypes), s)
      (tuplerD(out), remaining)
    }

    override def segmentShapes(dataTypes: DataTypes, s: Seq[Shape]): (Shapes, Seq[Shape]) = {
      val (out, remaining) = dataL.segmentShapes(genD.to(dataTypes), s)
      (tuplerS(out), remaining)
    }

    override def dataToString(data: PT): String = dataL.dataToString(genT.to(data))
    override def dataTypesToString(dataTypes: DataTypes): String = dataL.dataTypesToString(genD.to(dataTypes))
    override def shapesToString(shapes: Shapes): String = dataL.shapesToString(genS.to(shapes))
  }
}
