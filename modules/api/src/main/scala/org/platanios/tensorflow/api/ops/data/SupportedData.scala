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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT64}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.generic.CanBuildFrom
import scala.collection.{MapLike, SeqLike, breakOut, mutable}
import scala.language.higherKinds
import scala.reflect.ClassTag

// TODO: [DATA] Separate into readers and transformations.
// TODO: [DATA] paddedBatchAndDropRemainder
// TODO: [DATA] denseToSparseBatch
// TODO: [DATA] listFiles

/** Data can be emitted by [[Dataset]]s (i.e., the element types of all [[Dataset]]s are [[SupportedData]]).
  *
  * Currently supported data types are:
  *   - Single [[Tensor]].
  *   - Sequences of other [[SupportedData]] (e.g., `Seq`s, `List`s, etc.).
  *     - Sequences that are not homogeneous are not supported (e.g., `Seq(data1, Seq(data1, data2))`).
  *     - Note that, for that reason, even though `Seq(List(data1), List(data1, data2))` is supported,
  *       `Seq(Seq(data1), List(data1, data2))` is not.
  *     - A sequence containing both [[Output]]s and [[SparseOutput]]s, for example, is considered heterogeneous.
  *       For such cases, it is advisable to use tuples.
  *   - Arrays of other [[SupportedData]].
  *   - Maps with arbitrary key types and [[SupportedData]] value types.
  *   - Products of other [[SupportedData]] (e.g., tuples).
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
trait SupportedData[O] {
  type D
  type S

  def size(dataType: D): Int

  def dataType(output: O): D
  def shape(output: O): S

  def outputs(output: O): Seq[Output[Any]]

  def dataTypes(dataType: D): Seq[DataType[Any]]
  def shapes(shape: S): Seq[Shape]

  def decodeOutput(dataType: D, outputs: Seq[Output[Any]]): (O, Seq[Output[Any]])
  def decodeDataType(dataType: D, dataTypes: Seq[DataType[Any]]): (D, Seq[DataType[Any]])
  def decodeShape(dataType: D, shapes: Seq[Shape]): (S, Seq[Shape])

  def dataTypeToString(dataType: D): String
  def shapeToString(shape: S): String
}

object SupportedData {
  type Aux[O, DD, SS] = SupportedData[O] {
    type D = DD
    type S = SS
  }

  implicit def outputEvidence[T]: Aux[Output[T], DataType[T], Shape] = {
    new SupportedData[Output[T]] {
      override type D = DataType[T]
      override type S = Shape

      override def size(dataType: DataType[T]): Int = {
        1
      }

      override def dataType(output: Output[T]): DataType[T] = {
        output.dataType
      }

      override def shape(output: Output[T]): Shape = {
        output.shape
      }

      override def outputs(output: Output[T]): Seq[Output[Any]] = {
        Seq(output)
      }

      override def dataTypes(dataType: DataType[T]): Seq[DataType[Any]] = {
        Seq(dataType)
      }

      override def shapes(shape: Shape): Seq[Shape] = {
        Seq(shape)
      }

      override def decodeOutput(
          dataType: DataType[T],
          outputs: Seq[Output[Any]]
      ): (Output[T], Seq[Output[Any]]) = {
        (outputs.head.asInstanceOf[Output[T]], outputs.tail)
      }

      override def decodeDataType(
          dataType: DataType[T],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[T], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[T]], dataTypes.tail)
      }

      override def decodeShape(
          dataType: DataType[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes.tail)
      }

      override def dataTypeToString(dataType: DataType[T]): String = {
        dataType.toString
      }

      override def shapeToString(shape: Shape): String = {
        shape.toString
      }
    }
  }

  implicit def outputIndexedSlicesEvidence[T]: Aux[OutputIndexedSlices[T], (INT64, DataType[T], INT64), (Shape, Shape, Shape)] = {
    new SupportedData[OutputIndexedSlices[T]] {
      override type D = (INT64, DataType[T], INT64)
      override type S = (Shape, Shape, Shape)

      override def size(dataType: (INT64, DataType[T], INT64)): Int = {
        3
      }

      override def dataType(output: OutputIndexedSlices[T]): (INT64, DataType[T], INT64) = {
        (INT64, output.dataType, INT64)
      }

      override def shape(output: OutputIndexedSlices[T]): (Shape, Shape, Shape) = {
        val indicesShape = output.indices.shape
        val denseShapeShape = output.denseShape.shape
        val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
        (Shape(-1, rank), Shape(-1), Shape(rank))
      }

      override def outputs(output: OutputIndexedSlices[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def dataTypes(dataType: (INT64, DataType[T], INT64)): Seq[DataType[Any]] = {
        Seq(dataType._1, dataType._2, dataType._3)
      }

      override def shapes(shape: (Shape, Shape, Shape)): Seq[Shape] = {
        Seq(shape._1, shape._2, shape._3)
      }

      override def decodeOutput(
          dataType: (INT64, DataType[T], INT64),
          outputs: Seq[Output[Any]]
      ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
        (OutputIndexedSlices(
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]]), outputs.drop(3))
      }

      override def decodeDataType(
          dataType: (INT64, DataType[T], INT64),
          dataTypes: Seq[DataType[Any]]
      ): ((INT64, DataType[T], INT64), Seq[DataType[Any]]) = {
        ((dataTypes(0).asInstanceOf[DataType[Long]],
            dataTypes(1).asInstanceOf[DataType[T]],
            dataTypes(2).asInstanceOf[DataType[Long]]), dataTypes.drop(3))
      }

      override def decodeShape(
          dataType: (INT64, DataType[T], INT64),
          shapes: Seq[Shape]
      ): ((Shape, Shape, Shape), Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def dataTypeToString(dataType: (INT64, DataType[T], INT64)): String = {
        s"${dataType._1}:${dataType._2}:${dataType._3}"
      }

      override def shapeToString(shape: (Shape, Shape, Shape)): String = {
        s"${shape._1}:${shape._2}:${shape._3}"
      }
    }
  }

  implicit def sparseOutputEvidence[T]: Aux[SparseOutput[T], (INT64, DataType[T], INT64), (Shape, Shape, Shape)] = {
    new SupportedData[SparseOutput[T]] {
      override type D = (INT64, DataType[T], INT64)
      override type S = (Shape, Shape, Shape)

      override def size(dataType: (INT64, DataType[T], INT64)): Int = {
        3
      }

      override def dataType(output: SparseOutput[T]): (INT64, DataType[T], INT64) = {
        (INT64, output.dataType, INT64)
      }

      override def shape(output: SparseOutput[T]): (Shape, Shape, Shape) = {
        val indicesShape = output.indices.shape
        val denseShapeShape = output.denseShape.shape
        val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
        (Shape(-1, rank), Shape(-1), Shape(rank))
      }

      override def outputs(output: SparseOutput[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def dataTypes(dataType: (INT64, DataType[T], INT64)): Seq[DataType[Any]] = {
        Seq(dataType._1, dataType._2, dataType._3)
      }

      override def shapes(shape: (Shape, Shape, Shape)): Seq[Shape] = {
        Seq(shape._1, shape._2, shape._3)
      }

      override def decodeOutput(
          dataType: (INT64, DataType[T], INT64),
          outputs: Seq[Output[Any]]
      ): (SparseOutput[T], Seq[Output[Any]]) = {
        (SparseOutput(
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]]), outputs.drop(3))
      }

      override def decodeDataType(
          dataType: (INT64, DataType[T], INT64),
          dataTypes: Seq[DataType[Any]]
      ): ((INT64, DataType[T], INT64), Seq[DataType[Any]]) = {
        ((dataTypes(0).asInstanceOf[DataType[Long]],
            dataTypes(1).asInstanceOf[DataType[T]],
            dataTypes(2).asInstanceOf[DataType[Long]]), dataTypes.drop(3))
      }

      override def decodeShape(
          dataType: (INT64, DataType[T], INT64),
          shapes: Seq[Shape]
      ): ((Shape, Shape, Shape), Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def dataTypeToString(dataType: (INT64, DataType[T], INT64)): String = {
        s"${dataType._1}:${dataType._2}:${dataType._3}"
      }

      override def shapeToString(shape: (Shape, Shape, Shape)): String = {
        s"${shape._1}:${shape._2}:${shape._3}"
      }
    }
  }

  implicit def arrayEvidence[T: ClassTag, DD: ClassTag, SS: ClassTag](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Array[T], Array[DD], Array[SS]] = {
    new SupportedData[Array[T]] {
      override type D = Array[DD]
      override type S = Array[SS]

      override def size(dataType: Array[DD]): Int = {
        dataType.map(ev.size).sum
      }

      override def dataType(output: Array[T]): Array[DD] = {
        output.map(ev.dataType)
      }

      override def shape(output: Array[T]): Array[SS] = {
        output.map(ev.shape)
      }

      override def outputs(output: Array[T]): Seq[Output[Any]] = {
        output.flatMap(ev.outputs).toSeq
      }

      override def dataTypes(dataType: Array[DD]): Seq[DataType[Any]] = {
        dataType.flatMap(ev.dataTypes).toSeq
      }

      override def shapes(shape: Array[SS]): Seq[Shape] = {
        shape.flatMap(ev.shapes).toSeq
      }

      override def decodeOutput(
          dataType: Array[DD],
          outputs: Seq[Output[Any]]
      ): (Array[T], Seq[Output[Any]]) = {
        val n = size(dataType)
        (dataType.zip(Collections.segment(outputs.take(n), dataType.map(ev.size).toSeq))
            .map(f => ev.decodeOutput(f._1, f._2)._1), outputs.drop(n))
      }

      override def decodeDataType(
          dataType: Array[DD],
          dataTypes: Seq[DataType[Any]]
      ): (Array[DD], Seq[DataType[Any]]) = {
        val n = size(dataType)
        (dataType.zip(Collections.segment(dataTypes.take(n), dataType.map(ev.size).toSeq))
            .map(f => ev.decodeDataType(f._1, f._2)._1), dataTypes.drop(n))
      }

      override def decodeShape(
          dataType: Array[DD],
          shapes: Seq[Shape]
      ): (Array[SS], Seq[Shape]) = {
        val n = size(dataType)
        (dataType.zip(Collections.segment(shapes.take(n), dataType.map(ev.size).toSeq))
            .map(f => ev.decodeShape(f._1, f._2)._1), shapes.drop(n))
      }

      override def dataTypeToString(dataType: Array[DD]): String = {
        s"{${dataType.map(ev.dataTypeToString).mkString(", ")}}"
      }

      override def shapeToString(shape: Array[SS]): String = {
        s"{${shape.map(ev.shapeToString).mkString(", ")}}"
      }
    }
  }

  implicit def seqEvidence[T, DD, SS, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, DD, SS],
      cbfTD: CanBuildFrom[CC[T], DD, CC[DD]],
      cbfTS: CanBuildFrom[CC[T], SS, CC[SS]],
      cbfT: CanBuildFrom[Nothing, T, CC[T]],
      cbfD: CanBuildFrom[Nothing, DD, CC[DD]],
      cbfS: CanBuildFrom[Nothing, SS, CC[SS]]
  ): Aux[CC[T], CC[DD], CC[SS]] = {
    new SupportedData[CC[T]] {
      override type D = CC[DD]
      override type S = CC[SS]

      override def size(dataType: CC[DD]): Int = {
        dataType.map(ev.size)(breakOut).sum
      }

      override def dataType(output: CC[T]): CC[DD] = {
        output.map(ev.dataType)
      }

      override def shape(output: CC[T]): CC[SS] = {
        output.map(ev.shape)
      }

      override def outputs(output: CC[T]): Seq[Output[Any]] = {
        output.flatMap(ev.outputs)(breakOut)
      }

      override def dataTypes(dataType: CC[DD]): Seq[DataType[Any]] = {
        dataType.flatMap(ev.dataTypes)(breakOut)
      }

      override def shapes(shape: CC[SS]): Seq[Shape] = {
        shape.flatMap(ev.shapes)(breakOut)
      }

      override def decodeOutput(
          dataType: CC[DD],
          outputs: Seq[Output[Any]]
      ): (CC[T], Seq[Output[Any]]) = {
        val n = size(dataType)
        (dataType
            .zip(Collections.segment(outputs.take(n), dataType.map(ev.size)(breakOut)))(breakOut)
            .map(f => ev.decodeOutput(f._1, f._2)._1).to[CC](cbfT), outputs.drop(n))
      }

      override def decodeDataType(
          dataType: CC[DD],
          dataTypes: Seq[DataType[Any]]
      ): (CC[DD], Seq[DataType[Any]]) = {
        val n = size(dataType)
        (dataType
            .zip(Collections.segment(dataTypes.take(n), dataType.map(ev.size)(breakOut)))(breakOut)
            .map(f => ev.decodeDataType(f._1, f._2)._1).to[CC](cbfD), dataTypes.drop(n))
      }

      override def decodeShape(
          dataType: CC[DD],
          shapes: Seq[Shape]
      ): (CC[SS], Seq[Shape]) = {
        val n = size(dataType)
        (dataType
            .zip(Collections.segment(shapes.take(n), dataType.map(ev.size)(breakOut)))(breakOut)
            .map(f => ev.decodeShape(f._1, f._2)._1).to[CC](cbfS), shapes.drop(n))
      }

      override def dataTypeToString(dataType: CC[DD]): String = {
        s"{${dataType.map(ev.dataTypeToString)(breakOut).mkString(", ")}}"
      }

      override def shapeToString(shape: CC[SS]): String = {
        s"{${shape.map(ev.shapeToString)(breakOut).mkString(", ")}}"
      }
    }
  }


  implicit def mapEvidence[K, T, DD, SS, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[T, DD, SS]
  ): Aux[CC[K, T], Map[K, DD], Map[K, SS]] = {
    new SupportedData[CC[K, T]] {
      override type D = Map[K, DD]
      override type S = Map[K, SS]

      override def size(dataType: Map[K, DD]): Int = {
        dataType.values.map(ev.size).sum
      }

      override def dataType(output: CC[K, T]): Map[K, DD] = {
        output.mapValues(ev.dataType)
      }

      override def shape(output: CC[K, T]): Map[K, SS] = {
        output.mapValues(ev.shape)
      }

      override def outputs(output: CC[K, T]): Seq[Output[Any]] = {
        output.values.flatMap(ev.outputs).toSeq
      }

      override def dataTypes(dataType: Map[K, DD]): Seq[DataType[Any]] = {
        dataType.values.flatMap(ev.dataTypes).toSeq
      }

      override def shapes(shape: Map[K, SS]): Seq[Shape] = {
        shape.values.flatMap(ev.shapes).toSeq
      }

      override def decodeOutput(
          dataType: Map[K, DD],
          outputs: Seq[Output[Any]]
      ): (CC[K, T], Seq[Output[Any]]) = {
        val n = size(dataType)
        // TODO: [DATASETS] !!! Fix this hacky solution for the return type.
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(outputs.take(n), dataType.values.map(ev.size).toSeq))
              .map(f => ev.decodeOutput(f._1, f._2)._1)).toMap.asInstanceOf[CC[K, T]], outputs.drop(n))
      }

      override def decodeDataType(
          dataType: Map[K, DD],
          dataTypes: Seq[DataType[Any]]
      ): (Map[K, DD], Seq[DataType[Any]]) = {
        val n = size(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(dataTypes.take(n), dataType.values.map(ev.size).toSeq))
              .map(f => ev.decodeDataType(f._1, f._2)._1)).toMap.asInstanceOf[Map[K, DD]], dataTypes.drop(n))
      }

      override def decodeShape(
          dataType: Map[K, DD],
          shapes: Seq[Shape]
      ): (Map[K, SS], Seq[Shape]) = {
        val n = size(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(shapes.take(n), dataType.values.map(ev.size).toSeq))
              .map(f => ev.decodeShape(f._1, f._2)._1)).toMap.asInstanceOf[Map[K, SS]], shapes.drop(n))
      }

      override def dataTypeToString(dataType: Map[K, DD]): String = {
        s"{${dataType.map(d => s"${d._1.toString} -> ${ev.dataTypeToString(d._2)}").mkString(", ")}}"
      }

      override def shapeToString(shape: Map[K, SS]): String = {
        s"{${shape.map(d => s"${d._1.toString} -> ${ev.shapeToString(d._2)}").mkString(", ")}}"
      }
    }
  }

  implicit val hnilEvidence: Aux[HNil, HNil, HNil] = {
    new SupportedData[HNil] {
      override type D = HNil
      override type S = HNil

      override def size(dataType: HNil): Int = {
        0
      }

      override def dataType(output: HNil): HNil = {
        HNil
      }

      override def shape(output: HNil): HNil = {
        HNil
      }

      override def outputs(output: HNil): Seq[Output[Any]] = {
        Seq.empty
      }

      override def dataTypes(dataType: HNil): Seq[DataType[Any]] = {
        Seq.empty
      }

      override def shapes(shape: HNil): Seq[Shape] = {
        Seq.empty
      }

      override def decodeOutput(
          dataType: HNil,
          outputs: Seq[Output[Any]]
      ): (HNil, Seq[Output[Any]]) = {
        (HNil, outputs)
      }

      override def decodeDataType(
          dataType: HNil,
          dataTypes: Seq[DataType[Any]]
      ): (HNil, Seq[DataType[Any]]) = {
        (HNil, dataTypes)
      }

      override def decodeShape(
          dataType: HNil,
          shapes: Seq[Shape]
      ): (HNil, Seq[Shape]) = {
        (HNil, shapes)
      }

      override def dataTypeToString(dataType: HNil): String = {
        ""
      }

      override def shapeToString(shape: HNil): String = {
        ""
      }
    }
  }

  implicit def recursiveEvidence[HT, HD, HS, TT <: HList, TD <: HList, TS <: HList](implicit
      dataHead: Lazy[Aux[HT, HD, HS]],
      dataTail: Aux[TT, TD, TS]
  ): Aux[HT :: TT, HD :: TD, HS :: TS] = {
    new SupportedData[HT :: TT] {
      override type D = HD :: TD
      override type S = HS :: TS

      override def size(dataType: HD :: TD): Int = {
        dataHead.value.size(dataType.head) +
            dataTail.size(dataType.tail)
      }

      override def dataType(output: HT :: TT): HD :: TD = {
        dataHead.value.dataType(output.head) ::
            dataTail.dataType(output.tail)
      }

      override def shape(output: HT :: TT): HS :: TS = {
        dataHead.value.shape(output.head) ::
            dataTail.shape(output.tail)
      }

      override def outputs(output: HT :: TT): Seq[Output[Any]] = {
        dataHead.value.outputs(output.head) ++
            dataTail.outputs(output.tail)
      }

      override def dataTypes(dataType: HD :: TD): Seq[DataType[Any]] = {
        dataHead.value.dataTypes(dataType.head) ++
            dataTail.dataTypes(dataType.tail)
      }

      override def shapes(shape: HS :: TS): Seq[Shape] = {
        dataHead.value.shapes(shape.head) ++
            dataTail.shapes(shape.tail)
      }

      override def decodeOutput(
          dataType: HD :: TD,
          outputs: Seq[Output[Any]]
      ): (HT :: TT, Seq[Output[Any]]) = {
        val (headOut, headRemaining) = dataHead.value.decodeOutput(dataType.head, outputs)
        val (tailOut, tailRemaining) = dataTail.decodeOutput(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def decodeDataType(
          dataType: HD :: TD,
          dataTypes: Seq[DataType[Any]]
      ): (HD :: TD, Seq[DataType[Any]]) = {
        val (headOut, headRemaining) = dataHead.value.decodeDataType(dataType.head, dataTypes)
        val (tailOut, tailRemaining) = dataTail.decodeDataType(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def decodeShape(
          dataType: HD :: TD,
          shapes: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = dataHead.value.decodeShape(dataType.head, shapes)
        val (tailOut, tailRemaining) = dataTail.decodeShape(dataType.tail, shapes)
        (headOut :: tailOut, tailRemaining)
      }

      override def dataTypeToString(dataType: HD :: TD): String = {
        val headPart = dataHead.value.dataTypeToString(dataType.head)
        val tailPart = dataTail.dataTypeToString(dataType.tail)
        if (headPart == "")
          tailPart
        else if (tailPart == "")
          headPart
        else
          s"$headPart, $tailPart"
      }

      override def shapeToString(shape: HS :: TS): String = {
        val headPart = dataHead.value.shapeToString(shape.head)
        val tailPart = dataTail.shapeToString(shape.tail)
        if (headPart == "")
          tailPart
        else if (tailPart == "")
          headPart
        else
          s"$headPart, $tailPart"
      }
    }
  }

  implicit def productEvidence[PT, PD, PS, HT <: HList, HD <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Aux[HT, HD, HS],
      tuplerT: Tupler.Aux[HT, PT],
      tuplerD: Tupler.Aux[HD, PD],
      tuplerS: Tupler.Aux[HS, PS],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS]
  ): Aux[PT, PD, PS] = {
    new SupportedData[PT] {
      override type D = PD
      override type S = PS

      override def size(dataType: PD): Int = {
        evT.size(genD.to(dataType))
      }

      override def dataType(output: PT): PD = {
        tuplerD(evT.dataType(genT.to(output)))
      }

      override def shape(output: PT): PS = {
        tuplerS(evT.shape(genT.to(output)))
      }

      override def outputs(output: PT): Seq[Output[Any]] = {
        evT.outputs(genT.to(output))
      }

      override def dataTypes(dataType: PD): Seq[DataType[Any]] = {
        evT.dataTypes(genD.to(dataType))
      }

      override def shapes(shape: PS): Seq[Shape] = {
        evT.shapes(genS.to(shape))
      }

      override def decodeOutput(
          dataType: PD,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.decodeOutput(genD.to(dataType), outputs)
        (tuplerT(out), remaining)
      }

      override def decodeDataType(
          dataType: PD,
          dataTypes: Seq[DataType[Any]]
      ): (PD, Seq[DataType[Any]]) = {
        val (out, remaining) = evT.decodeDataType(genD.to(dataType), dataTypes)
        (tuplerD(out), remaining)
      }

      override def decodeShape(
          dataType: PD,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evT.decodeShape(genD.to(dataType), shapes)
        (tuplerS(out), remaining)
      }

      override def dataTypeToString(dataType: PD): String = {
        evT.dataTypeToString(genD.to(dataType))
      }

      override def shapeToString(shape: PS): String = {
        evT.shapeToString(genS.to(shape))
      }
    }
  }
}
