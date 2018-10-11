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
import org.platanios.tensorflow.api.core.types.{DataType, TF, VARIANT, Variant}
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.{Output, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.language.higherKinds
import scala.reflect.ClassTag

/** Data that can be emitted by [[Dataset]]s (i.e., the element types of all [[Dataset]]s are [[OutputStructure]]).
  *
  * Currently supported data types are:
  *   - Single [[Tensor]].
  *   - Sequences of other [[OutputStructure]] (e.g., `Seq`s, `List`s, etc.).
  *     - Sequences that are not homogeneous are not supported (e.g., `Seq(data1, Seq(data1, data2))`).
  *     - Note that, for that reason, even though `Seq(List(data1), List(data1, data2))` is supported,
  *       `Seq(Seq(data1), List(data1, data2))` is not.
  *     - A sequence containing both [[Output]]s and [[SparseOutput]]s, for example, is considered heterogeneous.
  *       For such cases, it is advisable to use tuples.
  *   - Arrays of other [[OutputStructure]].
  *   - Maps with arbitrary key types and [[OutputStructure]] value types.
  *   - Products of other [[OutputStructure]] (e.g., tuples).
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
trait OutputStructure[T] {
  type D
  type S

  def sizeFromOutput(output: T): Int
  def sizeFromDataType(dataType: D): Int

  def dataType(output: T): D
  def shape(output: T): S

  def outputs(output: T): Seq[Output[Any]]

  def dataTypes(dataType: D): Seq[DataType[Any]]
  def shapes(shape: S): Seq[Shape]

  def decodeOutputFromOutput(output: T, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])

  def decodeOutputFromDataType(dataType: D, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])
  def decodeDataTypeFromDataType(dataType: D, dataTypes: Seq[DataType[Any]]): (D, Seq[DataType[Any]])
  def decodeShapeFromDataType(dataType: D, shapes: Seq[Shape]): (S, Seq[Shape])

  def dataTypeToString(dataType: D): String
  def shapeToString(shape: S): String
}

object OutputStructure {
  type Aux[T, DD, SS] = OutputStructure[T] {
    type D = DD
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit, Unit] = {
    new OutputStructure[Unit] {
      override type D = Unit
      override type S = Unit

      override def sizeFromOutput(output: Unit): Int = {
        0
      }

      override def sizeFromDataType(dataType: Unit): Int = {
        0
      }

      override def dataType(output: Unit): Unit = {
        ()
      }

      override def shape(output: Unit): Unit = {
        ()
      }

      override def outputs(output: Unit): Seq[Output[Any]] = {
        Seq.empty
      }

      override def dataTypes(dataType: Unit): Seq[DataType[Any]] = {
        Seq.empty
      }

      override def shapes(shape: Unit): Seq[Shape] = {
        Seq.empty
      }

      override def decodeOutputFromOutput(
          output: Unit,
          outputs: Seq[Output[Any]]
      ): (Unit, Seq[Output[Any]]) = {
        ((), outputs)
      }

      override def decodeOutputFromDataType(
          dataType: Unit,
          outputs: Seq[Output[Any]]
      ): (Unit, Seq[Output[Any]]) = {
        ((), outputs)
      }

      override def decodeDataTypeFromDataType(
          dataType: Unit,
          dataTypes: Seq[DataType[Any]]
      ): (Unit, Seq[DataType[Any]]) = {
        ((), dataTypes)
      }

      override def decodeShapeFromDataType(
          dataType: Unit,
          shapes: Seq[Shape]
      ): (Unit, Seq[Shape]) = {
        ((), shapes)
      }

      override def dataTypeToString(dataType: Unit): String = {
        ""
      }

      override def shapeToString(shape: Unit): String = {
        ""
      }
    }
  }

  implicit def fromOutput[T: TF]: Aux[Output[T], DataType[T], Shape] = {
    new OutputStructure[Output[T]] {
      override type D = DataType[T]
      override type S = Shape

      override def sizeFromOutput(output: Output[T]): Int = {
        1
      }

      override def sizeFromDataType(dataType: DataType[T]): Int = {
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

      override def decodeOutputFromOutput(
          output: Output[T],
          outputs: Seq[Output[Any]]
      ): (Output[T], Seq[Output[Any]]) = {
        (outputs.head.asInstanceOf[Output[T]], outputs.tail)
      }

      override def decodeOutputFromDataType(
          dataType: DataType[T],
          outputs: Seq[Output[Any]]
      ): (Output[T], Seq[Output[Any]]) = {
        (outputs.head.asInstanceOf[Output[T]], outputs.tail)
      }

      override def decodeDataTypeFromDataType(
          dataType: DataType[T],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[T], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[T]], dataTypes.tail)
      }

      override def decodeShapeFromDataType(
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

  // TODO: [FUNCTIONS] !!! Find a better way to deal with this for use in the reduce function of the "GroupByWindowDataset".

  case class VariantDataset[T] protected(
      handle: Output[Variant],
      private val _outputDataTypes: Any = null,
      private val _outputShapes: Any = null
  ) extends Dataset[T] {
    override val name: String = "VariantDataset"

    override def createHandle[D, S]()(implicit
        evT: Aux[T, D, S]
    ): Output[Variant] = {
      handle
    }

    override def outputDataTypes[D, S](implicit evT: Aux[T, D, S]): D = {
      _outputDataTypes.asInstanceOf[D]
    }

    override def outputShapes[D, S](implicit evT: Aux[T, D, S]): S = {
      _outputShapes.asInstanceOf[S]
    }
  }

  implicit def fromDataset[T, D, S](implicit
      evT: Aux[T, D, S]
  ): Aux[Dataset[T], DataType[Variant], Shape] = {
    new OutputStructure[Dataset[T]] {
      override type D = DataType[Variant]
      override type S = Shape

      override def sizeFromOutput(output: Dataset[T]): Int = {
        1
      }

      override def sizeFromDataType(dataType: DataType[Variant]): Int = {
        1
      }

      override def dataType(arg: Dataset[T]): DataType[Variant] = {
        VARIANT
      }

      override def shape(arg: Dataset[T]): Shape = {
        Shape()
      }

      override def outputs(arg: Dataset[T]): Seq[Output[Any]] = {
        Seq(arg.createHandle()(evT))
      }

      override def dataTypes(dataType: DataType[Variant]): Seq[DataType[Any]] = {
        Seq(VARIANT)
      }

      override def shapes(shape: Shape): Seq[Shape] = {
        Seq(shape)
      }

      override def decodeOutputFromOutput(
          output: Dataset[T],
          outputs: Seq[Output[Any]]
      ): (Dataset[T], Seq[Output[Any]]) = {
        (VariantDataset[T](
          handle = outputs.head.asInstanceOf[Output[Variant]],
          _outputDataTypes = output.outputDataTypes,
          _outputShapes = output.outputShapes
        ), outputs.drop(1))
      }

      override def decodeOutputFromDataType(
          dataType: DataType[Variant],
          outputs: Seq[Output[Any]]
      ): (Dataset[T], Seq[Output[Any]]) = {
        (VariantDataset[T](outputs.head.asInstanceOf[Output[Variant]]), outputs.drop(1))
      }

      override def decodeDataTypeFromDataType(
          dataType: DataType[Variant],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[Variant], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[Variant]], dataTypes.tail)
      }

      override def decodeShapeFromDataType(
          dataType: DataType[Variant],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes.tail)
      }

      override def dataTypeToString(dataType: DataType[Variant]): String = {
        dataType.toString
      }

      override def shapeToString(shape: Shape): String = {
        shape.toString
      }
    }
  }

  implicit def fromOption[T, DD, SS](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Option[T], Option[DD], Option[SS]] = {
    new OutputStructure[Option[T]] {
      override type D = Option[DD]
      override type S = Option[SS]

      override def sizeFromOutput(output: Option[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Option[DD]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def dataType(output: Option[T]): Option[DD] = {
        output.map(ev.dataType)
      }

      override def shape(output: Option[T]): Option[SS] = {
        output.map(ev.shape)
      }

      override def outputs(output: Option[T]): Seq[Output[Any]] = {
        output.toSeq.flatMap(ev.outputs)
      }

      override def dataTypes(dataType: Option[DD]): Seq[DataType[Any]] = {
        dataType.toSeq.flatMap(ev.dataTypes)
      }

      override def shapes(shape: Option[SS]): Seq[Shape] = {
        shape.toSeq.flatMap(ev.shapes)
      }

      override def decodeOutputFromOutput(
          output: Option[T],
          outputs: Seq[Output[Any]]
      ): (Option[T], Seq[Output[Any]]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeOutputFromOutput(o, outputs)
            (Some(result), remaining)
          case None => (None, outputs)
        }
      }

      override def decodeOutputFromDataType(
          dataType: Option[DD],
          outputs: Seq[Output[Any]]
      ): (Option[T], Seq[Output[Any]]) = {
        dataType match {
          case Some(d) =>
            val (result, remaining) = ev.decodeOutputFromDataType(d, outputs)
            (Some(result), remaining)
          case None => (None, outputs)
        }
      }

      override def decodeDataTypeFromDataType(
          dataType: Option[DD],
          dataTypes: Seq[DataType[Any]]
      ): (Option[DD], Seq[DataType[Any]]) = {
        dataType match {
          case Some(d) =>
            val (result, remaining) = ev.decodeDataTypeFromDataType(d, dataTypes)
            (Some(result), remaining)
          case None => (None, dataTypes)
        }
      }

      override def decodeShapeFromDataType(
          dataType: Option[DD],
          shapes: Seq[Shape]
      ): (Option[SS], Seq[Shape]) = {
        dataType match {
          case Some(d) =>
            val (result, remaining) = ev.decodeShapeFromDataType(d, shapes)
            (Some(result), remaining)
          case None => (None, shapes)
        }
      }

      override def dataTypeToString(dataType: Option[DD]): String = {
        s"{${dataType.map(ev.dataTypeToString).mkString(", ")}}"
      }

      override def shapeToString(shape: Option[SS]): String = {
        s"{${shape.map(ev.shapeToString).mkString(", ")}}"
      }
    }
  }

  implicit def fromArray[T: ClassTag, DD: ClassTag, SS: ClassTag](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Array[T], Array[DD], Array[SS]] = {
    new OutputStructure[Array[T]] {
      override type D = Array[DD]
      override type S = Array[SS]

      override def sizeFromOutput(output: Array[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Array[DD]): Int = {
        dataType.map(ev.sizeFromDataType).sum
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

      override def decodeOutputFromOutput(
          output: Array[T],
          outputs: Seq[Output[Any]]
      ): (Array[T], Seq[Output[Any]]) = {
        val n = sizeFromOutput(output)
        (output.zip(Collections.segment(outputs.take(n), output.map(ev.sizeFromOutput).toSeq))
            .map(f => ev.decodeOutputFromOutput(f._1, f._2)._1), outputs.drop(n))
      }

      override def decodeOutputFromDataType(
          dataType: Array[DD],
          outputs: Seq[Output[Any]]
      ): (Array[T], Seq[Output[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType.zip(Collections.segment(outputs.take(n), dataType.map(ev.sizeFromDataType).toSeq))
            .map(f => ev.decodeOutputFromDataType(f._1, f._2)._1), outputs.drop(n))
      }

      override def decodeDataTypeFromDataType(
          dataType: Array[DD],
          dataTypes: Seq[DataType[Any]]
      ): (Array[DD], Seq[DataType[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType.zip(Collections.segment(dataTypes.take(n), dataType.map(ev.sizeFromDataType).toSeq))
            .map(f => ev.decodeDataTypeFromDataType(f._1, f._2)._1), dataTypes.drop(n))
      }

      override def decodeShapeFromDataType(
          dataType: Array[DD],
          shapes: Seq[Shape]
      ): (Array[SS], Seq[Shape]) = {
        val n = sizeFromDataType(dataType)
        (dataType.zip(Collections.segment(shapes.take(n), dataType.map(ev.sizeFromDataType).toSeq))
            .map(f => ev.decodeShapeFromDataType(f._1, f._2)._1), shapes.drop(n))
      }

      override def dataTypeToString(dataType: Array[DD]): String = {
        s"{${dataType.map(ev.dataTypeToString).mkString(", ")}}"
      }

      override def shapeToString(shape: Array[SS]): String = {
        s"{${shape.map(ev.shapeToString).mkString(", ")}}"
      }
    }
  }

  implicit def fromSeq[T, DD, SS](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Seq[T], Seq[DD], Seq[SS]] = {
    new OutputStructure[Seq[T]] {
      override type D = Seq[DD]
      override type S = Seq[SS]

      override def sizeFromOutput(output: Seq[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Seq[DD]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def dataType(output: Seq[T]): Seq[DD] = {
        output.map(ev.dataType)
      }

      override def shape(output: Seq[T]): Seq[SS] = {
        output.map(ev.shape)
      }

      override def outputs(output: Seq[T]): Seq[Output[Any]] = {
        output.flatMap(ev.outputs)
      }

      override def dataTypes(dataType: Seq[DD]): Seq[DataType[Any]] = {
        dataType.flatMap(ev.dataTypes)
      }

      override def shapes(shape: Seq[SS]): Seq[Shape] = {
        shape.flatMap(ev.shapes)
      }

      override def decodeOutputFromOutput(
          output: Seq[T],
          outputs: Seq[Output[Any]]
      ): (Seq[T], Seq[Output[Any]]) = {
        val n = sizeFromOutput(output)
        (output
            .zip(Collections.segment(outputs.take(n), output.map(ev.sizeFromOutput)))
            .map(f => ev.decodeOutputFromOutput(f._1, f._2)._1), outputs.drop(n))
      }

      override def decodeOutputFromDataType(
          dataType: Seq[DD],
          outputs: Seq[Output[Any]]
      ): (Seq[T], Seq[Output[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType
            .zip(Collections.segment(outputs.take(n), dataType.map(ev.sizeFromDataType)))
            .map(f => ev.decodeOutputFromDataType(f._1, f._2)._1), outputs.drop(n))
      }

      override def decodeDataTypeFromDataType(
          dataType: Seq[DD],
          dataTypes: Seq[DataType[Any]]
      ): (Seq[DD], Seq[DataType[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType
            .zip(Collections.segment(dataTypes.take(n), dataType.map(ev.sizeFromDataType)))
            .map(f => ev.decodeDataTypeFromDataType(f._1, f._2)._1), dataTypes.drop(n))
      }

      override def decodeShapeFromDataType(
          dataType: Seq[DD],
          shapes: Seq[Shape]
      ): (Seq[SS], Seq[Shape]) = {
        val n = sizeFromDataType(dataType)
        (dataType
            .zip(Collections.segment(shapes.take(n), dataType.map(ev.sizeFromDataType)))
            .map(f => ev.decodeShapeFromDataType(f._1, f._2)._1), shapes.drop(n))
      }

      override def dataTypeToString(dataType: Seq[DD]): String = {
        s"{${dataType.map(ev.dataTypeToString).mkString(", ")}}"
      }

      override def shapeToString(shape: Seq[SS]): String = {
        s"{${shape.map(ev.shapeToString).mkString(", ")}}"
      }
    }
  }

  implicit def fromMap[K, T, DD, SS](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Map[K, T], Map[K, DD], Map[K, SS]] = {
    new OutputStructure[Map[K, T]] {
      override type D = Map[K, DD]
      override type S = Map[K, SS]

      override def sizeFromOutput(output: Map[K, T]): Int = {
        output.values.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Map[K, DD]): Int = {
        dataType.values.map(ev.sizeFromDataType).sum
      }

      override def dataType(output: Map[K, T]): Map[K, DD] = {
        output.mapValues(ev.dataType)
      }

      override def shape(output: Map[K, T]): Map[K, SS] = {
        output.mapValues(ev.shape)
      }

      override def outputs(output: Map[K, T]): Seq[Output[Any]] = {
        output.values.flatMap(ev.outputs).toSeq
      }

      override def dataTypes(dataType: Map[K, DD]): Seq[DataType[Any]] = {
        dataType.values.flatMap(ev.dataTypes).toSeq
      }

      override def shapes(shape: Map[K, SS]): Seq[Shape] = {
        shape.values.flatMap(ev.shapes).toSeq
      }

      override def decodeOutputFromOutput(
          output: Map[K, T],
          outputs: Seq[Output[Any]]
      ): (Map[K, T], Seq[Output[Any]]) = {
        val n = sizeFromOutput(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(outputs.take(n), output.values.map(ev.sizeFromOutput).toSeq))
              .map(f => ev.decodeOutputFromOutput(f._1, f._2)._1)).toMap, outputs.drop(n))
      }

      override def decodeOutputFromDataType(
          dataType: Map[K, DD],
          outputs: Seq[Output[Any]]
      ): (Map[K, T], Seq[Output[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(outputs.take(n), dataType.values.map(ev.sizeFromDataType).toSeq))
              .map(f => ev.decodeOutputFromDataType(f._1, f._2)._1)).toMap, outputs.drop(n))
      }

      override def decodeDataTypeFromDataType(
          dataType: Map[K, DD],
          dataTypes: Seq[DataType[Any]]
      ): (Map[K, DD], Seq[DataType[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(dataTypes.take(n), dataType.values.map(ev.sizeFromDataType).toSeq))
              .map(f => ev.decodeDataTypeFromDataType(f._1, f._2)._1)).toMap, dataTypes.drop(n))
      }

      override def decodeShapeFromDataType(
          dataType: Map[K, DD],
          shapes: Seq[Shape]
      ): (Map[K, SS], Seq[Shape]) = {
        val n = sizeFromDataType(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(shapes.take(n), dataType.values.map(ev.sizeFromDataType).toSeq))
              .map(f => ev.decodeShapeFromDataType(f._1, f._2)._1)).toMap, shapes.drop(n))
      }

      override def dataTypeToString(dataType: Map[K, DD]): String = {
        s"{${dataType.map(d => s"${d._1.toString} -> ${ev.dataTypeToString(d._2)}").mkString(", ")}}"
      }

      override def shapeToString(shape: Map[K, SS]): String = {
        s"{${shape.map(d => s"${d._1.toString} -> ${ev.shapeToString(d._2)}").mkString(", ")}}"
      }
    }
  }

  implicit val fromHNil: Aux[HNil, HNil, HNil] = {
    new OutputStructure[HNil] {
      override type D = HNil
      override type S = HNil

      override def sizeFromOutput(output: HNil): Int = {
        0
      }

      override def sizeFromDataType(dataType: HNil): Int = {
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

      override def decodeOutputFromOutput(
          output: HNil,
          outputs: Seq[Output[Any]]
      ): (HNil, Seq[Output[Any]]) = {
        (HNil, outputs)
      }

      override def decodeOutputFromDataType(
          dataType: HNil,
          outputs: Seq[Output[Any]]
      ): (HNil, Seq[Output[Any]]) = {
        (HNil, outputs)
      }

      override def decodeDataTypeFromDataType(
          dataType: HNil,
          dataTypes: Seq[DataType[Any]]
      ): (HNil, Seq[DataType[Any]]) = {
        (HNil, dataTypes)
      }

      override def decodeShapeFromDataType(
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

  implicit def fromHList[HT, HD, HS, TT <: HList, TD <: HList, TS <: HList](implicit
      evH: Strict[Aux[HT, HD, HS]],
      evT: Aux[TT, TD, TS]
  ): Aux[HT :: TT, HD :: TD, HS :: TS] = {
    new OutputStructure[HT :: TT] {
      override type D = HD :: TD
      override type S = HS :: TS

      override def sizeFromOutput(output: HT :: TT): Int = {
        evH.value.sizeFromOutput(output.head) +
            evT.sizeFromOutput(output.tail)
      }

      override def sizeFromDataType(dataType: HD :: TD): Int = {
        evH.value.sizeFromDataType(dataType.head) +
            evT.sizeFromDataType(dataType.tail)
      }

      override def dataType(output: HT :: TT): HD :: TD = {
        evH.value.dataType(output.head) ::
            evT.dataType(output.tail)
      }

      override def shape(output: HT :: TT): HS :: TS = {
        evH.value.shape(output.head) ::
            evT.shape(output.tail)
      }

      override def outputs(output: HT :: TT): Seq[Output[Any]] = {
        evH.value.outputs(output.head) ++
            evT.outputs(output.tail)
      }

      override def dataTypes(dataType: HD :: TD): Seq[DataType[Any]] = {
        evH.value.dataTypes(dataType.head) ++
            evT.dataTypes(dataType.tail)
      }

      override def shapes(shape: HS :: TS): Seq[Shape] = {
        evH.value.shapes(shape.head) ++
            evT.shapes(shape.tail)
      }

      override def decodeOutputFromOutput(
          output: HT :: TT,
          outputs: Seq[Output[Any]]
      ): (HT :: TT, Seq[Output[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeOutputFromOutput(output.head, outputs)
        val (tailOut, tailRemaining) = evT.decodeOutputFromOutput(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def decodeOutputFromDataType(
          dataType: HD :: TD,
          outputs: Seq[Output[Any]]
      ): (HT :: TT, Seq[Output[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeOutputFromDataType(dataType.head, outputs)
        val (tailOut, tailRemaining) = evT.decodeOutputFromDataType(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def decodeDataTypeFromDataType(
          dataType: HD :: TD,
          dataTypes: Seq[DataType[Any]]
      ): (HD :: TD, Seq[DataType[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeDataTypeFromDataType(dataType.head, dataTypes)
        val (tailOut, tailRemaining) = evT.decodeDataTypeFromDataType(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def decodeShapeFromDataType(
          dataType: HD :: TD,
          shapes: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = evH.value.decodeShapeFromDataType(dataType.head, shapes)
        val (tailOut, tailRemaining) = evT.decodeShapeFromDataType(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def dataTypeToString(dataType: HD :: TD): String = {
        val headPart = evH.value.dataTypeToString(dataType.head)
        val tailPart = evT.dataTypeToString(dataType.tail)
        if (headPart == "")
          tailPart
        else if (tailPart == "")
          headPart
        else
          s"$headPart, $tailPart"
      }

      override def shapeToString(shape: HS :: TS): String = {
        val headPart = evH.value.shapeToString(shape.head)
        val tailPart = evT.shapeToString(shape.tail)
        if (headPart == "")
          tailPart
        else if (tailPart == "")
          headPart
        else
          s"$headPart, $tailPart"
      }
    }
  }

  implicit def fromCoproduct[HT, HD, HS, TT <: Coproduct, TD <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[Aux[HT, HD, HS]],
      evT: Aux[TT, TD, TS]
  ): Aux[HT :+: TT, HD :+: TD, HS :+: TS] = {
    new OutputStructure[HT :+: TT] {
      override type D = HD :+: TD
      override type S = HS :+: TS

      override def sizeFromOutput(output: HT :+: TT): Int = {
        output match {
          case Inl(h) => evH.value.sizeFromOutput(h)
          case Inr(t) => evT.sizeFromOutput(t)
        }
      }

      override def sizeFromDataType(dataType: HD :+: TD): Int = {
        dataType match {
          case Inl(h) => evH.value.sizeFromDataType(h)
          case Inr(t) => evT.sizeFromDataType(t)
        }
      }

      override def dataType(output: HT :+: TT): HD :+: TD = {
        output match {
          case Inl(h) => Inl(evH.value.dataType(h))
          case Inr(t) => Inr(evT.dataType(t))
        }
      }

      override def shape(output: HT :+: TT): HS :+: TS = {
        output match {
          case Inl(h) => Inl(evH.value.shape(h))
          case Inr(t) => Inr(evT.shape(t))
        }
      }

      override def outputs(output: HT :+: TT): Seq[Output[Any]] = {
        output match {
          case Inl(h) => evH.value.outputs(h)
          case Inr(t) => evT.outputs(t)
        }
      }

      override def dataTypes(dataType: HD :+: TD): Seq[DataType[Any]] = {
        dataType match {
          case Inl(h) => evH.value.dataTypes(h)
          case Inr(t) => evT.dataTypes(t)
        }
      }

      override def shapes(shape: HS :+: TS): Seq[Shape] = {
        shape match {
          case Inl(h) => evH.value.shapes(h)
          case Inr(t) => evT.shapes(t)
        }
      }

      override def decodeOutputFromOutput(
          output: HT :+: TT,
          outputs: Seq[Output[Any]]
      ): (HT :+: TT, Seq[Output[Any]]) = {
        output match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeOutputFromOutput(h, outputs)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeOutputFromOutput(t, outputs)
            (Inr(result), remaining)
        }
      }

      override def decodeOutputFromDataType(
          dataType: HD :+: TD,
          outputs: Seq[Output[Any]]
      ): (HT :+: TT, Seq[Output[Any]]) = {
        dataType match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeOutputFromDataType(h, outputs)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeOutputFromDataType(t, outputs)
            (Inr(result), remaining)
        }
      }

      override def decodeDataTypeFromDataType(
          dataType: HD :+: TD,
          dataTypes: Seq[DataType[Any]]
      ): (HD :+: TD, Seq[DataType[Any]]) = {
        dataType match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeDataTypeFromDataType(h, dataTypes)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeDataTypeFromDataType(t, dataTypes)
            (Inr(result), remaining)
        }
      }

      override def decodeShapeFromDataType(
          dataType: HD :+: TD,
          shapes: Seq[Shape]
      ): (HS :+: TS, Seq[Shape]) = {
        dataType match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeShapeFromDataType(h, shapes)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeShapeFromDataType(t, shapes)
            (Inr(result), remaining)
        }
      }

      override def dataTypeToString(dataType: HD :+: TD): String = {
        dataType match {
          case Inl(h) => evH.value.dataTypeToString(h)
          case Inr(t) => evT.dataTypeToString(t)
        }
      }

      override def shapeToString(shape: HS :+: TS): String = {
        shape match {
          case Inl(h) => evH.value.shapeToString(h)
          case Inr(t) => evT.shapeToString(t)
        }
      }
    }
  }

  implicit def fromProduct[PT <: Product, PD <: Product, PS <: Product, HT <: HList, HD <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[Aux[HT, HD, HS]],
      tuplerD: Tupler.Aux[HD, PD],
      tuplerS: Tupler.Aux[HS, PS],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS]
  ): Aux[PT, PD, PS] = {
    new OutputStructure[PT] {
      override type D = PD
      override type S = PS

      override def sizeFromOutput(output: PT): Int = {
        evT.value.sizeFromOutput(genT.to(output))
      }

      override def sizeFromDataType(dataType: PD): Int = {
        evT.value.sizeFromDataType(genD.to(dataType))
      }

      override def dataType(output: PT): PD = {
        tuplerD(evT.value.dataType(genT.to(output)))
      }

      override def shape(output: PT): PS = {
        tuplerS(evT.value.shape(genT.to(output)))
      }

      override def outputs(output: PT): Seq[Output[Any]] = {
        evT.value.outputs(genT.to(output))
      }

      override def dataTypes(dataType: PD): Seq[DataType[Any]] = {
        evT.value.dataTypes(genD.to(dataType))
      }

      override def shapes(shape: PS): Seq[Shape] = {
        evT.value.shapes(genS.to(shape))
      }

      override def decodeOutputFromOutput(
          output: PT,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.value.decodeOutputFromOutput(genT.to(output), outputs)
        (genT.from(out), remaining)
      }

      override def decodeOutputFromDataType(
          dataType: PD,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.value.decodeOutputFromDataType(genD.to(dataType), outputs)
        (genT.from(out), remaining)
      }

      override def decodeDataTypeFromDataType(
          dataType: PD,
          dataTypes: Seq[DataType[Any]]
      ): (PD, Seq[DataType[Any]]) = {
        val (out, remaining) = evT.value.decodeDataTypeFromDataType(genD.to(dataType), dataTypes)
        (tuplerD(out), remaining)
      }

      override def decodeShapeFromDataType(
          dataType: PD,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evT.value.decodeShapeFromDataType(genD.to(dataType), shapes)
        (tuplerS(out), remaining)
      }

      override def dataTypeToString(dataType: PD): String = {
        evT.value.dataTypeToString(genD.to(dataType))
      }

      override def shapeToString(shape: PS): String = {
        evT.value.shapeToString(genS.to(shape))
      }
    }
  }
}
