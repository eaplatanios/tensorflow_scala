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
import org.platanios.tensorflow.api.core.types.{DataType, FLOAT32, INT64, VARIANT, Variant}
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput, TensorArray}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.language.higherKinds
import scala.reflect.ClassTag

/** Data that can be emitted by [[Dataset]]s (i.e., the element types of all [[Dataset]]s are [[NestedStructure]]).
  *
  * Currently supported data types are:
  *   - Single [[Tensor]].
  *   - Sequences of other [[NestedStructure]] (e.g., `Seq`s, `List`s, etc.).
  *     - Sequences that are not homogeneous are not supported (e.g., `Seq(data1, Seq(data1, data2))`).
  *     - Note that, for that reason, even though `Seq(List(data1), List(data1, data2))` is supported,
  *       `Seq(Seq(data1), List(data1, data2))` is not.
  *     - A sequence containing both [[Output]]s and [[SparseOutput]]s, for example, is considered heterogeneous.
  *       For such cases, it is advisable to use tuples.
  *   - Arrays of other [[NestedStructure]].
  *   - Maps with arbitrary key types and [[NestedStructure]] value types.
  *   - Products of other [[NestedStructure]] (e.g., tuples).
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
trait NestedStructure[T] {
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
  def decodeDataTypeFromOutput(output: T, dataTypes: Seq[DataType[Any]]): (D, Seq[DataType[Any]])
  def decodeShapeFromOutput(output: T, shapes: Seq[Shape]): (S, Seq[Shape])

  def decodeOutputFromDataType(dataType: D, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])
  def decodeDataTypeFromDataType(dataType: D, dataTypes: Seq[DataType[Any]]): (D, Seq[DataType[Any]])
  def decodeShapeFromDataType(dataType: D, shapes: Seq[Shape]): (S, Seq[Shape])

  def map(
      value: T,
      shape: Option[S],
      converter: NestedStructure.Converter
  ): T
}

object NestedStructure {
  type SparseDataType[T] = (DataType[Long], DataType[T], DataType[Long])
  type SparseShape = (Shape, Shape, Shape)

  trait Converter {
    def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = value
    def apply[T](value: OutputIndexedSlices[T], shape: Option[SparseShape]): OutputIndexedSlices[T] = value
    def apply[T](value: SparseOutput[T], shape: Option[SparseShape]): SparseOutput[T] = value
    def apply[T](value: TensorArray[T], shape: Option[Shape]): TensorArray[T] = value
    def apply[T](value: Dataset[T], shape: Option[Shape]): Dataset[T] = value
  }

  type Aux[T, DD, SS] = NestedStructure[T] {
    type D = DD
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit, Unit] = {
    new NestedStructure[Unit] {
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

      override def decodeDataTypeFromOutput(
          output: Unit,
          dataTypes: Seq[DataType[Any]]
      ): (Unit, Seq[DataType[Any]]) = {
        ((), dataTypes)
      }

      override def decodeShapeFromOutput(
          output: Unit,
          shapes: Seq[Shape]
      ): (Unit, Seq[Shape]) = {
        ((), shapes)
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

      def map(
          value: Unit,
          shape: Option[Unit],
          converter: NestedStructure.Converter
      ): Unit = {
        ()
      }
    }
  }

  implicit def fromOutput[T]: Aux[Output[T], DataType[T], Shape] = {
    new NestedStructure[Output[T]] {
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

      override def decodeDataTypeFromOutput(
          output: Output[T],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[T], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[T]], dataTypes)
      }

      override def decodeShapeFromOutput(
          output: Output[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes)
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

      override def map(
          value: Output[T],
          shape: Option[S],
          converter: Converter
      ): Output[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: Aux[OutputIndexedSlices[T], SparseDataType[T], SparseShape] = {
    new NestedStructure[OutputIndexedSlices[T]] {
      override type D = SparseDataType[T]
      override type S = SparseShape

      override def sizeFromOutput(output: OutputIndexedSlices[T]): Int = {
        3
      }

      override def sizeFromDataType(dataType: SparseDataType[T]): Int = {
        3
      }

      override def dataType(output: OutputIndexedSlices[T]): SparseDataType[T] = {
        (INT64, output.dataType, INT64)
      }

      override def shape(output: OutputIndexedSlices[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }

      override def outputs(output: OutputIndexedSlices[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def dataTypes(dataType: SparseDataType[T]): Seq[DataType[Any]] = {
        Seq(dataType._1, dataType._2, dataType._3)
      }

      override def shapes(shape: SparseShape): Seq[Shape] = {
        Seq(shape._1, shape._2, shape._3)
      }

      override def decodeOutputFromOutput(
          output: OutputIndexedSlices[T],
          outputs: Seq[Output[Any]]
      ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
        (OutputIndexedSlices[T](
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]],
        ), outputs.drop(3))
      }

      override def decodeDataTypeFromOutput(
          output: OutputIndexedSlices[T],
          dataTypes: Seq[DataType[Any]]
      ): (SparseDataType[T], Seq[DataType[Any]]) = {
        ((dataTypes(0).asInstanceOf[DataType[Long]],
            dataTypes(1).asInstanceOf[DataType[T]],
            dataTypes(2).asInstanceOf[DataType[Long]]
        ), dataTypes.drop(3))
      }

      override def decodeShapeFromOutput(
          output: OutputIndexedSlices[T],
          shapes: Seq[Shape]
      ): (SparseShape, Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def decodeOutputFromDataType(
          dataType: SparseDataType[T],
          outputs: Seq[Output[Any]]
      ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
        (OutputIndexedSlices[T](
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]],
        ), outputs.drop(3))
      }

      override def decodeDataTypeFromDataType(
          dataType: SparseDataType[T],
          dataTypes: Seq[DataType[Any]]
      ): (SparseDataType[T], Seq[DataType[Any]]) = {
        ((dataTypes(0).asInstanceOf[DataType[Long]],
            dataTypes(1).asInstanceOf[DataType[T]],
            dataTypes(2).asInstanceOf[DataType[Long]]
        ), dataTypes.drop(3))
      }

      override def decodeShapeFromDataType(
          dataType: SparseDataType[T],
          shapes: Seq[Shape]
      ): (SparseShape, Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def map(
          value: OutputIndexedSlices[T],
          shape: Option[SparseShape],
          converter: Converter
      ): OutputIndexedSlices[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromSparseOutput[T]: Aux[SparseOutput[T], SparseDataType[T], SparseShape] = {
    new NestedStructure[SparseOutput[T]] {
      override type D = SparseDataType[T]
      override type S = SparseShape

      override def sizeFromOutput(output: SparseOutput[T]): Int = {
        3
      }

      override def sizeFromDataType(dataType: SparseDataType[T]): Int = {
        3
      }

      override def dataType(output: SparseOutput[T]): SparseDataType[T] = {
        (INT64, output.dataType, INT64)
      }

      override def shape(output: SparseOutput[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }

      override def outputs(output: SparseOutput[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def dataTypes(dataType: SparseDataType[T]): Seq[DataType[Any]] = {
        Seq(dataType._1, dataType._2, dataType._3)
      }

      override def shapes(shape: SparseShape): Seq[Shape] = {
        Seq(shape._1, shape._2, shape._3)
      }

      override def decodeOutputFromOutput(
          output: SparseOutput[T],
          outputs: Seq[Output[Any]]
      ): (SparseOutput[T], Seq[Output[Any]]) = {
        (SparseOutput[T](
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]],
        ), outputs.drop(3))
      }

      override def decodeDataTypeFromOutput(
          output: SparseOutput[T],
          dataTypes: Seq[DataType[Any]]
      ): (SparseDataType[T], Seq[DataType[Any]]) = {
        ((dataTypes(0).asInstanceOf[DataType[Long]],
            dataTypes(1).asInstanceOf[DataType[T]],
            dataTypes(2).asInstanceOf[DataType[Long]]
        ), dataTypes.drop(3))
      }

      override def decodeShapeFromOutput(
          output: SparseOutput[T],
          shapes: Seq[Shape]
      ): (SparseShape, Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def decodeOutputFromDataType(
          dataType: SparseDataType[T],
          outputs: Seq[Output[Any]]
      ): (SparseOutput[T], Seq[Output[Any]]) = {
        (SparseOutput[T](
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]],
        ), outputs.drop(3))
      }

      override def decodeDataTypeFromDataType(
          dataType: SparseDataType[T],
          dataTypes: Seq[DataType[Any]]
      ): (SparseDataType[T], Seq[DataType[Any]]) = {
        ((dataTypes(0).asInstanceOf[DataType[Long]],
            dataTypes(1).asInstanceOf[DataType[T]],
            dataTypes(2).asInstanceOf[DataType[Long]]
        ), dataTypes.drop(3))
      }

      override def decodeShapeFromDataType(
          dataType: SparseDataType[T],
          shapes: Seq[Shape]
      ): (SparseShape, Seq[Shape]) = {
        ((shapes(0), shapes(1), shapes(2)), shapes.drop(3))
      }

      override def map(
          value: SparseOutput[T],
          shape: Option[SparseShape],
          converter: Converter
      ): SparseOutput[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], DataType[Float], Shape] = {
    new NestedStructure[TensorArray[T]] {
      override type D = DataType[Float]
      override type S = Shape

      override def sizeFromOutput(output: TensorArray[T]): Int = {
        1
      }

      override def sizeFromDataType(dataType: DataType[Float]): Int = {
        1
      }

      override def dataType(output: TensorArray[T]): DataType[Float] = {
        FLOAT32
      }

      override def shape(output: TensorArray[T]): Shape = {
        ???
      }

      override def outputs(output: TensorArray[T]): Seq[Output[Any]] = {
        Seq(output.flow)
      }

      override def dataTypes(dataType: DataType[Float]): Seq[DataType[Any]] = {
        Seq(dataType)
      }

      override def shapes(shape: Shape): Seq[Shape] = {
        Seq(shape)
      }

      override def decodeOutputFromOutput(
          output: TensorArray[T],
          outputs: Seq[Output[Any]]
      ): (TensorArray[T], Seq[Output[Any]]) = {
        val newTensorArray = output.copy(
          flow = outputs.head.asInstanceOf[Output[Float]]
        )(output.evTTF)
        // TODO: !!! [TENSOR_ARRAY] What about colocate with?
        (newTensorArray, outputs.tail)
      }

      override def decodeDataTypeFromOutput(
          output: TensorArray[T],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[Float], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[Float]], dataTypes)
      }

      override def decodeShapeFromOutput(
          output: TensorArray[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes)
      }

      override def decodeOutputFromDataType(
          dataType: DataType[Float],
          outputs: Seq[Output[Any]]
      ): (TensorArray[T], Seq[Output[Any]]) = {
        ???
      }

      override def decodeDataTypeFromDataType(
          dataType: DataType[Float],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[Float], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[Float]], dataTypes.tail)
      }

      override def decodeShapeFromDataType(
          dataType: DataType[Float],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes.tail)
      }

      override def map(
          value: TensorArray[T],
          shape: Option[S],
          converter: Converter
      ): TensorArray[T] = {
        converter[T](value, shape)
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
    new NestedStructure[Dataset[T]] {
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

      override def decodeDataTypeFromOutput(
          output: Dataset[T],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[Variant], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[Variant]], dataTypes.tail)
      }

      override def decodeShapeFromOutput(
          output: Dataset[T],
          shapes: Seq[Shape]
      ): (Shape, Seq[Shape]) = {
        (shapes.head, shapes.tail)
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

      override def map(
          value: Dataset[T],
          shape: Option[S],
          converter: Converter
      ): Dataset[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromOption[T, DD, SS](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Option[T], Option[DD], Option[SS]] = {
    new NestedStructure[Option[T]] {
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

      override def decodeDataTypeFromOutput(
          output: Option[T],
          dataTypes: Seq[DataType[Any]]
      ): (Option[DD], Seq[DataType[Any]]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeDataTypeFromOutput(o, dataTypes)
            (Some(result), remaining)
          case None => (None, dataTypes)
        }
      }

      override def decodeShapeFromOutput(
          output: Option[T],
          shapes: Seq[Shape]
      ): (Option[SS], Seq[Shape]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeShapeFromOutput(o, shapes)
            (Some(result), remaining)
          case None => (None, shapes)
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

      override def map(
          value: Option[T],
          shape: Option[Option[SS]],
          converter: Converter
      ): Option[T] = {
        (value, shape) match {
          case (Some(v), Some(s)) => Some(ev.map(v, s, converter))
          case _ => None
        }
      }
    }
  }

  implicit def fromArray[T: ClassTag, DD: ClassTag, SS: ClassTag](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Array[T], Array[DD], Array[SS]] = {
    new NestedStructure[Array[T]] {
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

      override def decodeDataTypeFromOutput(
          output: Array[T],
          dataTypes: Seq[DataType[Any]]
      ): (Array[DD], Seq[DataType[Any]]) = {
        val n = sizeFromOutput(output)
        (output.zip(Collections.segment(dataTypes.take(n), output.map(ev.sizeFromOutput).toSeq))
            .map(f => ev.decodeDataTypeFromOutput(f._1, f._2)._1), dataTypes.drop(n))
      }

      override def decodeShapeFromOutput(
          output: Array[T],
          shapes: Seq[Shape]
      ): (Array[SS], Seq[Shape]) = {
        val n = sizeFromOutput(output)
        (output.zip(Collections.segment(shapes.take(n), output.map(ev.sizeFromOutput).toSeq))
            .map(f => ev.decodeShapeFromOutput(f._1, f._2)._1), shapes.drop(n))
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

      override def map(
          value: Array[T],
          shape: Option[Array[SS]],
          converter: Converter
      ): Array[T] = {
        val shapes = shape.map(_.map(Option(_))).getOrElse(value.map(_ => None))
        value.zip(shapes).map(p => ev.map(p._1, p._2, converter))
      }
    }
  }

  implicit def fromSeq[T, DD, SS](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Seq[T], Seq[DD], Seq[SS]] = {
    new NestedStructure[Seq[T]] {
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

      override def decodeDataTypeFromOutput(
          output: Seq[T],
          dataTypes: Seq[DataType[Any]]
      ): (Seq[DD], Seq[DataType[Any]]) = {
        val n = sizeFromOutput(output)
        (output
            .zip(Collections.segment(dataTypes.take(n), output.map(ev.sizeFromOutput)))
            .map(f => ev.decodeDataTypeFromOutput(f._1, f._2)._1), dataTypes.drop(n))
      }

      override def decodeShapeFromOutput(
          output: Seq[T],
          shapes: Seq[Shape]
      ): (Seq[SS], Seq[Shape]) = {
        val n = sizeFromOutput(output)
        (output
            .zip(Collections.segment(shapes.take(n), output.map(ev.sizeFromOutput)))
            .map(f => ev.decodeShapeFromOutput(f._1, f._2)._1), shapes.drop(n))
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

      override def map(
          value: Seq[T],
          shape: Option[Seq[SS]],
          converter: Converter
      ): Seq[T] = {
        val shapes = shape.map(_.map(Option(_))).getOrElse(value.map(_ => None))
        value.zip(shapes).map(p => ev.map(p._1, p._2, converter))
      }
    }
  }

  implicit def fromMap[K, T, DD, SS](implicit
      ev: Aux[T, DD, SS]
  ): Aux[Map[K, T], Map[K, DD], Map[K, SS]] = {
    new NestedStructure[Map[K, T]] {
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

      override def decodeDataTypeFromOutput(
          output: Map[K, T],
          dataTypes: Seq[DataType[Any]]
      ): (Map[K, DD], Seq[DataType[Any]]) = {
        val n = sizeFromOutput(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(dataTypes.take(n), output.values.map(ev.sizeFromOutput).toSeq))
              .map(f => ev.decodeDataTypeFromOutput(f._1, f._2)._1)).toMap, dataTypes.drop(n))
      }

      override def decodeShapeFromOutput(
          output: Map[K, T],
          shapes: Seq[Shape]
      ): (Map[K, SS], Seq[Shape]) = {
        val n = sizeFromOutput(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(shapes.take(n), output.values.map(ev.sizeFromOutput).toSeq))
              .map(f => ev.decodeShapeFromOutput(f._1, f._2)._1)).toMap, shapes.drop(n))
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

      override def map(
          value: Map[K, T],
          shape: Option[Map[K, SS]],
          converter: Converter
      ): Map[K, T] = {
        val shapes = shape.map(_.mapValues(Option(_))).getOrElse(value.mapValues(_ => None))
        (value.keys ++ shapes.keys).map(k => k -> ev.map(value(k), shapes(k), converter)).toMap
      }
    }
  }

  implicit val fromHNil: Aux[HNil, HNil, HNil] = {
    new NestedStructure[HNil] {
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

      override def decodeDataTypeFromOutput(
          output: HNil,
          dataTypes: Seq[DataType[Any]]
      ): (HNil, Seq[DataType[Any]]) = {
        (HNil, dataTypes)
      }

      override def decodeShapeFromOutput(
          output: HNil,
          shapes: Seq[Shape]
      ): (HNil, Seq[Shape]) = {
        (HNil, shapes)
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

      override def map(
          value: HNil,
          shape: Option[HNil],
          converter: Converter
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HD, HS, TT <: HList, TD <: HList, TS <: HList](implicit
      evH: Strict[Aux[HT, HD, HS]],
      evT: Aux[TT, TD, TS]
  ): Aux[HT :: TT, HD :: TD, HS :: TS] = {
    new NestedStructure[HT :: TT] {
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

      override def decodeDataTypeFromOutput(
          output: HT :: TT,
          dataTypes: Seq[DataType[Any]]
      ): (HD :: TD, Seq[DataType[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeDataTypeFromOutput(output.head, dataTypes)
        val (tailOut, tailRemaining) = evT.decodeDataTypeFromOutput(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def decodeShapeFromOutput(
          output: HT :: TT,
          shapes: Seq[Shape]
      ): (HS :: TS, Seq[Shape]) = {
        val (headOut, headRemaining) = evH.value.decodeShapeFromOutput(output.head, shapes)
        val (tailOut, tailRemaining) = evT.decodeShapeFromOutput(output.tail, headRemaining)
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

      override def map(
          value: HT :: TT,
          shape: Option[HS :: TS],
          converter: Converter
      ): HT :: TT = {
        evH.value.map(value.head, shape.map(_.head), converter) ::
            evT.map(value.tail, shape.map(_.tail), converter)
      }
    }
  }

  implicit def fromCoproduct[HT, HD, HS, TT <: Coproduct, TD <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[Aux[HT, HD, HS]],
      evT: Aux[TT, TD, TS]
  ): Aux[HT :+: TT, HD :+: TD, HS :+: TS] = {
    new NestedStructure[HT :+: TT] {
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

      override def decodeDataTypeFromOutput(
          output: HT :+: TT,
          dataTypes: Seq[DataType[Any]]
      ): (HD :+: TD, Seq[DataType[Any]]) = {
        output match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeDataTypeFromOutput(h, dataTypes)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeDataTypeFromOutput(t, dataTypes)
            (Inr(result), remaining)
        }
      }

      override def decodeShapeFromOutput(
          output: HT :+: TT,
          shapes: Seq[Shape]
      ): (HS :+: TS, Seq[Shape]) = {
        output match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeShapeFromOutput(h, shapes)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeShapeFromOutput(t, shapes)
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

      override def map(
          value: HT :+: TT,
          shape: Option[HS :+: TS],
          converter: Converter
      ): HT :+: TT = {
        value match {
          case Inl(hv) => Inl(evH.value.map(hv, shape.map(_.asInstanceOf[Inl[HS, TS]].head), converter))
          case Inr(tv) => Inr(evT.map(tv, shape.map(_.asInstanceOf[Inr[HS, TS]].tail), converter))
          case _ => throw new IllegalStateException("Something went wrong while deriving implicit evidence.")
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
    new NestedStructure[PT] {
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

      override def decodeDataTypeFromOutput(
          output: PT,
          dataTypes: Seq[DataType[Any]]
      ): (PD, Seq[DataType[Any]]) = {
        val (out, remaining) = evT.value.decodeDataTypeFromOutput(genT.to(output), dataTypes)
        (genD.from(out), remaining)
      }

      override def decodeShapeFromOutput(
          output: PT,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evT.value.decodeShapeFromOutput(genT.to(output), shapes)
        (genS.from(out), remaining)
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

      override def map(
          value: PT,
          shape: Option[PS],
          converter: Converter
      ): PT = {
        genT.from(evT.value.map(genT.to(value), shape.map(genS.to), converter))
      }
    }
  }
}
