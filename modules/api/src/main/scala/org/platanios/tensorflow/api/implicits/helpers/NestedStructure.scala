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
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.language.higherKinds

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
sealed trait NestedStructure[T] {
  type V // Tensor value
  type D // Data type
  type S // Shape

  def asAux(): NestedStructure.Aux[T, V, D, S] = {
    this.asInstanceOf[NestedStructure.Aux[T, V, D, S]]
  }

  def sizeFromOutput(output: T): Int
  def sizeFromDataType(dataType: D): Int

  def dataTypeFromOutput(output: T): D
  def shapeFromOutput(output: T): S

  def outputFromTensor(tensor: V): T
  def dataTypeFromTensor(tensor: V): D
  def shapeFromTensor(tensor: V): S

  def outputs(output: T): Seq[Output[Any]]
  def tensors(tensor: V): Seq[Tensor[Any]]
  def dataTypes(dataType: D): Seq[DataType[Any]]
  def shapes(shape: S): Seq[Shape]

  def decodeOutputFromOutput(output: T, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])
  def decodeTensorFromOutput(output: T, tensors: Seq[Tensor[Any]]): (V, Seq[Tensor[Any]])
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

  type Aux[T, VV, DD, SS] = NestedStructure[T] {
    type V = VV
    type D = DD
    type S = SS
  }

  implicit val evStructureString: Aux[Output[String], Tensor[String], DataType[String], Shape] = {
    fromOutput[String]
  }

  implicit val evStructureLong: Aux[Output[Long], Tensor[Long], DataType[Long], Shape] = {
    fromOutput[Long]
  }

  implicit val evStructureFloat: Aux[Output[Float], Tensor[Float], DataType[Float], Shape] = {
    fromOutput[Float]
  }

  implicit val evStructureUntyped: Aux[Output[Any], Tensor[Any], DataType[Any], Shape] = {
    fromOutput[Any]
  }

  implicit val evStructureSeqUntyped: Aux[Seq[Output[Any]], Seq[Tensor[Any]], Seq[DataType[Any]], Seq[Shape]] = {
    fromSeq[Output[Any], Tensor[Any], DataType[Any], Shape]
  }

  implicit val evStructureOptionSeqUntyped: Aux[Option[Seq[Output[Any]]], Option[Seq[Tensor[Any]]], Option[Seq[DataType[Any]]], Option[Seq[Shape]]] = {
    fromOption[Seq[Output[Any]], Seq[Tensor[Any]], Seq[DataType[Any]], Seq[Shape]]
  }

  def apply[T](implicit ev: NestedStructure[T]): NestedStructure.Aux[T, ev.V, ev.D, ev.S] = {
    ev.asAux()
  }

  implicit val fromUnit: NestedStructure.Aux[Unit, Unit, Unit, Unit] = {
    new NestedStructure[Unit] {
      override type V = Unit
      override type D = Unit
      override type S = Unit

      override def sizeFromOutput(output: Unit): Int = {
        0
      }

      override def sizeFromDataType(dataType: Unit): Int = {
        0
      }

      override def dataTypeFromOutput(output: Unit): Unit = {
        ()
      }

      override def shapeFromOutput(output: Unit): Unit = {
        ()
      }

      override def outputFromTensor(tensor: Unit): Unit = {
        ()
      }

      override def dataTypeFromTensor(output: Unit): Unit = {
        ()
      }

      override def shapeFromTensor(output: Unit): Unit = {
        ()
      }

      override def outputs(output: Unit): Seq[Output[Any]] = {
        Seq.empty
      }

      override def tensors(tensor: Unit): Seq[Tensor[Any]] = {
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

      override def decodeTensorFromOutput(
          output: Unit,
          tensors: Seq[Tensor[Any]]
      ): (Unit, Seq[Tensor[Any]]) = {
        ((), tensors)
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

  implicit def fromOutput[T]: NestedStructure.Aux[Output[T], Tensor[T], DataType[T], Shape] = {
    new NestedStructure[Output[T]] {
      override type V = Tensor[T]
      override type D = DataType[T]
      override type S = Shape

      override def sizeFromOutput(output: Output[T]): Int = {
        1
      }

      override def sizeFromDataType(dataType: DataType[T]): Int = {
        1
      }

      override def dataTypeFromOutput(output: Output[T]): DataType[T] = {
        output.dataType
      }

      override def shapeFromOutput(output: Output[T]): Shape = {
        output.shape
      }

      override def outputFromTensor(tensor: Tensor[T]): Output[T] = {
        tensor.toOutput
      }

      override def dataTypeFromTensor(tensor: Tensor[T]): DataType[T] = {
        tensor.dataType
      }

      override def shapeFromTensor(tensor: Tensor[T]): Shape = {
        tensor.shape
      }

      override def outputs(output: Output[T]): Seq[Output[Any]] = {
        Seq(output)
      }

      override def tensors(tensor: Tensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor)
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

      override def decodeTensorFromOutput(
          output: Output[T],
          tensors: Seq[Tensor[Any]]
      ): (Tensor[T], Seq[Tensor[Any]]) = {
        (tensors.head.asInstanceOf[Tensor[T]], tensors.tail)
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
          converter: NestedStructure.Converter
      ): Output[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: NestedStructure.Aux[OutputIndexedSlices[T], TensorIndexedSlices[T], SparseDataType[T], SparseShape] = {
    new NestedStructure[OutputIndexedSlices[T]] {
      override type V = TensorIndexedSlices[T]
      override type D = SparseDataType[T]
      override type S = SparseShape

      override def sizeFromOutput(output: OutputIndexedSlices[T]): Int = {
        3
      }

      override def sizeFromDataType(dataType: SparseDataType[T]): Int = {
        3
      }

      override def dataTypeFromOutput(output: OutputIndexedSlices[T]): SparseDataType[T] = {
        (INT64, output.dataType, INT64)
      }

      override def shapeFromOutput(output: OutputIndexedSlices[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }

      override def outputFromTensor(tensor: TensorIndexedSlices[T]): OutputIndexedSlices[T] = {
        OutputIndexedSlices(
          indices = tensor.indices,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape)
      }

      override def dataTypeFromTensor(tensor: TensorIndexedSlices[T]): SparseDataType[T] = {
        (INT64, tensor.dataType, INT64)
      }

      override def shapeFromTensor(tensor: TensorIndexedSlices[T]): SparseShape = {
        (tensor.indices.shape, tensor.values.shape, tensor.denseShape.shape)
      }

      override def outputs(output: OutputIndexedSlices[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def tensors(tensor: TensorIndexedSlices[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices, tensor.values, tensor.denseShape)
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
          denseShape = outputs(2).asInstanceOf[Output[Long]]
        ), outputs.drop(3))
      }

      override def decodeTensorFromOutput(
          output: OutputIndexedSlices[T],
          tensors: Seq[Tensor[Any]]
      ): (TensorIndexedSlices[T], Seq[Tensor[Any]]) = {
        (TensorIndexedSlices[T](
          indices = tensors(0).asInstanceOf[Tensor[Long]],
          values = tensors(1).asInstanceOf[Tensor[T]],
          denseShape = tensors(2).asInstanceOf[Tensor[Long]]
        ), tensors.drop(3))
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
          denseShape = outputs(2).asInstanceOf[Output[Long]]
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
          converter: NestedStructure.Converter
      ): OutputIndexedSlices[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromSparseOutput[T]: NestedStructure.Aux[SparseOutput[T], SparseTensor[T], SparseDataType[T], SparseShape] = {
    new NestedStructure[SparseOutput[T]] {
      override type V = SparseTensor[T]
      override type D = SparseDataType[T]
      override type S = SparseShape

      override def sizeFromOutput(output: SparseOutput[T]): Int = {
        3
      }

      override def sizeFromDataType(dataType: SparseDataType[T]): Int = {
        3
      }

      override def dataTypeFromOutput(output: SparseOutput[T]): SparseDataType[T] = {
        (INT64, output.dataType, INT64)
      }

      override def shapeFromOutput(output: SparseOutput[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }

      override def outputFromTensor(tensor: SparseTensor[T]): SparseOutput[T] = {
        SparseOutput(
          indices = tensor.indices,
          values = tensor.values.toOutput,
          denseShape = tensor.denseShape)
      }

      override def dataTypeFromTensor(tensor: SparseTensor[T]): SparseDataType[T] = {
        (INT64, tensor.dataType, INT64)
      }

      override def shapeFromTensor(tensor: SparseTensor[T]): SparseShape = {
        (tensor.indices.shape, tensor.values.shape, tensor.denseShape.shape)
      }

      override def outputs(output: SparseOutput[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def tensors(tensor: SparseTensor[T]): Seq[Tensor[Any]] = {
        Seq(tensor.indices, tensor.values, tensor.denseShape)
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
          denseShape = outputs(2).asInstanceOf[Output[Long]]
        ), outputs.drop(3))
      }

      override def decodeTensorFromOutput(
          output: SparseOutput[T],
          tensors: Seq[Tensor[Any]]
      ): (SparseTensor[T], Seq[Tensor[Any]]) = {
        (SparseTensor[T](
          indices = tensors(0).asInstanceOf[Tensor[Long]],
          values = tensors(1).asInstanceOf[Tensor[T]],
          denseShape = tensors(2).asInstanceOf[Tensor[Long]]
        ), tensors.drop(3))
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
          denseShape = outputs(2).asInstanceOf[Output[Long]]
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
          converter: NestedStructure.Converter
      ): SparseOutput[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromTensorArray[T]: NestedStructure.Aux[TensorArray[T], Tensor[Float], DataType[Float], Shape] = {
    new NestedStructure[TensorArray[T]] {
      override type V = Tensor[Float]
      override type D = DataType[Float]
      override type S = Shape

      override def sizeFromOutput(output: TensorArray[T]): Int = {
        1
      }

      override def sizeFromDataType(dataType: DataType[Float]): Int = {
        1
      }

      override def dataTypeFromOutput(output: TensorArray[T]): DataType[Float] = {
        FLOAT32
      }

      override def shapeFromOutput(output: TensorArray[T]): Shape = {
        Shape()
      }

      override def outputFromTensor(tensor: Tensor[Float]): TensorArray[T] = {
        ???
      }

      override def dataTypeFromTensor(tensor: Tensor[Float]): DataType[Float] = {
        FLOAT32
      }

      override def shapeFromTensor(tensor: Tensor[Float]): Shape = {
        Shape()
      }

      override def outputs(output: TensorArray[T]): Seq[Output[Any]] = {
        Seq(output.flow)
      }

      override def tensors(tensor: Tensor[Float]): Seq[Tensor[Any]] = {
        Seq(tensor)
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

      override def decodeTensorFromOutput(
          output: TensorArray[T],
          tensors: Seq[Tensor[Any]]
      ): (Tensor[Float], Seq[Tensor[Any]]) = {
        (tensors.head.asInstanceOf[Tensor[Float]], tensors.tail)
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
          converter: NestedStructure.Converter
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

    override def createHandle[V, D, S]()(implicit
        evT: NestedStructure.Aux[T, V, D, S]
    ): Output[Variant] = {
      handle
    }

    override def outputDataTypes[V, D, S](implicit evT: NestedStructure.Aux[T, V, D, S]): D = {
      _outputDataTypes.asInstanceOf[D]
    }

    override def outputShapes[V, D, S](implicit evT: NestedStructure.Aux[T, V, D, S]): S = {
      _outputShapes.asInstanceOf[S]
    }
  }

  implicit def fromDataset[T, V, D, S](implicit
      evT: NestedStructure.Aux[T, V, D, S]
  ): NestedStructure.Aux[Dataset[T], Unit, DataType[Variant], Shape] = {
    new NestedStructure[Dataset[T]] {
      override type V = Unit
      override type D = DataType[Variant]
      override type S = Shape

      override def sizeFromOutput(output: Dataset[T]): Int = {
        1
      }

      override def sizeFromDataType(dataType: DataType[Variant]): Int = {
        1
      }

      override def dataTypeFromOutput(arg: Dataset[T]): DataType[Variant] = {
        VARIANT
      }

      override def shapeFromOutput(arg: Dataset[T]): Shape = {
        Shape()
      }

      override def outputFromTensor(tensor: Unit): Dataset[T] = {
        ???
      }

      override def dataTypeFromTensor(tensor: Unit): DataType[Variant] = {
        VARIANT
      }

      override def shapeFromTensor(tensor: Unit): Shape = {
        Shape()
      }

      override def outputs(arg: Dataset[T]): Seq[Output[Any]] = {
        Seq(arg.createHandle()(evT))
      }

      override def tensors(tensor: Unit): Seq[Tensor[Any]] = {
        Seq.empty
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
          _outputDataTypes = output.outputDataTypes(evT),
          _outputShapes = output.outputShapes(evT)
        ), outputs.drop(1))
      }

      override def decodeTensorFromOutput(
          output: Dataset[T],
          tensors: Seq[Tensor[Any]]
      ): (Unit, Seq[Tensor[Any]]) = {
        ???
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
          converter: NestedStructure.Converter
      ): Dataset[T] = {
        converter[T](value, shape)
      }
    }
  }

  implicit def fromOption[T, VV, DD, SS](implicit
      ev: NestedStructure.Aux[T, VV, DD, SS]
  ): NestedStructure.Aux[Option[T], Option[VV], Option[DD], Option[SS]] = {
    new NestedStructure[Option[T]] {
      override type V = Option[VV]
      override type D = Option[DD]
      override type S = Option[SS]

      override def sizeFromOutput(output: Option[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Option[DD]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def dataTypeFromOutput(output: Option[T]): Option[DD] = {
        output.map(ev.dataTypeFromOutput)
      }

      override def shapeFromOutput(output: Option[T]): Option[SS] = {
        output.map(ev.shapeFromOutput)
      }

      override def outputFromTensor(tensor: Option[VV]): Option[T] = {
        tensor.map(ev.outputFromTensor)
      }

      override def dataTypeFromTensor(tensor: Option[VV]): Option[DD] = {
        tensor.map(ev.dataTypeFromTensor)
      }

      override def shapeFromTensor(tensor: Option[VV]): Option[SS] = {
        tensor.map(ev.shapeFromTensor)
      }

      override def outputs(output: Option[T]): Seq[Output[Any]] = {
        output.toSeq.flatMap(ev.outputs)
      }

      override def tensors(tensor: Option[VV]): Seq[Tensor[Any]] = {
        tensor.toSeq.flatMap(ev.tensors)
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

      override def decodeTensorFromOutput(
          output: Option[T],
          tensors: Seq[Tensor[Any]]
      ): (Option[VV], Seq[Tensor[Any]]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeTensorFromOutput(o, tensors)
            (Some(result), remaining)
          case None => (None, tensors)
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
          converter: NestedStructure.Converter
      ): Option[T] = {
        (value, shape) match {
          case (Some(v), Some(s)) => Some(ev.map(v, s, converter))
          case _ => None
        }
      }
    }
  }

  implicit def fromSeq[T, VV, DD, SS](implicit
      ev: NestedStructure.Aux[T, VV, DD, SS]
  ): NestedStructure.Aux[Seq[T], Seq[VV], Seq[DD], Seq[SS]] = {
    new NestedStructure[Seq[T]] {
      override type V = Seq[VV]
      override type D = Seq[DD]
      override type S = Seq[SS]

      override def sizeFromOutput(output: Seq[T]): Int = {
        output.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Seq[DD]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def dataTypeFromOutput(output: Seq[T]): Seq[DD] = {
        output.map(ev.dataTypeFromOutput)
      }

      override def shapeFromOutput(output: Seq[T]): Seq[SS] = {
        output.map(ev.shapeFromOutput)
      }

      override def outputFromTensor(tensor: Seq[VV]): Seq[T] = {
        tensor.map(ev.outputFromTensor)
      }

      override def dataTypeFromTensor(tensor: Seq[VV]): Seq[DD] = {
        tensor.map(ev.dataTypeFromTensor)
      }

      override def shapeFromTensor(tensor: Seq[VV]): Seq[SS] = {
        tensor.map(ev.shapeFromTensor)
      }

      override def outputs(output: Seq[T]): Seq[Output[Any]] = {
        output.flatMap(ev.outputs)
      }

      override def tensors(tensor: Seq[VV]): Seq[Tensor[Any]] = {
        tensor.flatMap(ev.tensors)
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

      override def decodeTensorFromOutput(
          output: Seq[T],
          tensors: Seq[Tensor[Any]]
      ): (Seq[VV], Seq[Tensor[Any]]) = {
        val n = sizeFromOutput(output)
        (output
            .zip(Collections.segment(tensors.take(n), output.map(ev.sizeFromOutput)))
            .map(f => ev.decodeTensorFromOutput(f._1, f._2)._1), tensors.drop(n))
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
          converter: NestedStructure.Converter
      ): Seq[T] = {
        val shapes = shape.map(_.map(Option(_))).getOrElse(value.map(_ => None))
        value.zip(shapes).map(p => ev.map(p._1, p._2, converter))
      }
    }
  }

  implicit def fromMap[K, T, VV, DD, SS](implicit
      ev: NestedStructure.Aux[T, VV, DD, SS]
  ): NestedStructure.Aux[Map[K, T], Map[K, VV], Map[K, DD], Map[K, SS]] = {
    new NestedStructure[Map[K, T]] {
      override type V = Map[K, VV]
      override type D = Map[K, DD]
      override type S = Map[K, SS]

      override def sizeFromOutput(output: Map[K, T]): Int = {
        output.values.map(ev.sizeFromOutput).sum
      }

      override def sizeFromDataType(dataType: Map[K, DD]): Int = {
        dataType.values.map(ev.sizeFromDataType).sum
      }

      override def dataTypeFromOutput(output: Map[K, T]): Map[K, DD] = {
        output.mapValues(ev.dataTypeFromOutput)
      }

      override def shapeFromOutput(output: Map[K, T]): Map[K, SS] = {
        output.mapValues(ev.shapeFromOutput)
      }

      override def outputFromTensor(tensor: Map[K, VV]): Map[K, T] = {
        tensor.mapValues(ev.outputFromTensor)
      }

      override def dataTypeFromTensor(tensor: Map[K, VV]): Map[K, DD] = {
        tensor.mapValues(ev.dataTypeFromTensor)
      }

      override def shapeFromTensor(tensor: Map[K, VV]): Map[K, SS] = {
        tensor.mapValues(ev.shapeFromTensor)
      }

      override def outputs(output: Map[K, T]): Seq[Output[Any]] = {
        output.values.flatMap(ev.outputs).toSeq
      }

      override def tensors(tensor: Map[K, VV]): Seq[Tensor[Any]] = {
        tensor.values.flatMap(ev.tensors).toSeq
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

      override def decodeTensorFromOutput(
          output: Map[K, T],
          tensors: Seq[Tensor[Any]]
      ): (Map[K, VV], Seq[Tensor[Any]]) = {
        val n = sizeFromOutput(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(tensors.take(n), output.values.map(ev.sizeFromOutput).toSeq))
              .map(f => ev.decodeTensorFromOutput(f._1, f._2)._1)).toMap, tensors.drop(n))
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
          converter: NestedStructure.Converter
      ): Map[K, T] = {
        val shapes = shape.map(_.mapValues(Option(_))).getOrElse(value.mapValues(_ => None))
        (value.keys ++ shapes.keys).map(k => k -> ev.map(value(k), shapes(k), converter)).toMap
      }
    }
  }

  implicit val fromHNil: NestedStructure.Aux[HNil, HNil, HNil, HNil] = {
    new NestedStructure[HNil] {
      override type V = HNil
      override type D = HNil
      override type S = HNil

      override def sizeFromOutput(output: HNil): Int = {
        0
      }

      override def sizeFromDataType(dataType: HNil): Int = {
        0
      }

      override def dataTypeFromOutput(output: HNil): HNil = {
        HNil
      }

      override def shapeFromOutput(output: HNil): HNil = {
        HNil
      }

      override def outputFromTensor(tensor: HNil): HNil = {
        HNil
      }

      override def dataTypeFromTensor(output: HNil): HNil = {
        HNil
      }

      override def shapeFromTensor(output: HNil): HNil = {
        HNil
      }

      override def outputs(output: HNil): Seq[Output[Any]] = {
        Seq.empty
      }

      override def tensors(tensor: HNil): Seq[Tensor[Any]] = {
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

      override def decodeTensorFromOutput(
          output: HNil,
          tensors: Seq[Tensor[Any]]
      ): (HNil, Seq[Tensor[Any]]) = {
        (HNil, tensors)
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
          converter: NestedStructure.Converter
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HV, HD, HS, TT <: HList, TV <: HList, TD <: HList, TS <: HList](implicit
      evH: Strict[NestedStructure.Aux[HT, HV, HD, HS]],
      evT: NestedStructure.Aux[TT, TV, TD, TS]
  ): NestedStructure.Aux[HT :: TT, HV :: TV, HD :: TD, HS :: TS] = {
    new NestedStructure[HT :: TT] {
      override type V = HV :: TV
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

      override def dataTypeFromOutput(output: HT :: TT): HD :: TD = {
        evH.value.dataTypeFromOutput(output.head) ::
            evT.dataTypeFromOutput(output.tail)
      }

      override def shapeFromOutput(output: HT :: TT): HS :: TS = {
        evH.value.shapeFromOutput(output.head) ::
            evT.shapeFromOutput(output.tail)
      }

      override def outputFromTensor(tensor: HV :: TV): HT :: TT = {
        evH.value.outputFromTensor(tensor.head) ::
            evT.outputFromTensor(tensor.tail)
      }

      override def dataTypeFromTensor(tensor: HV :: TV): HD :: TD = {
        evH.value.dataTypeFromTensor(tensor.head) ::
            evT.dataTypeFromTensor(tensor.tail)
      }

      override def shapeFromTensor(tensor: HV :: TV): HS :: TS = {
        evH.value.shapeFromTensor(tensor.head) ::
            evT.shapeFromTensor(tensor.tail)
      }

      override def outputs(output: HT :: TT): Seq[Output[Any]] = {
        evH.value.outputs(output.head) ++
            evT.outputs(output.tail)
      }

      override def tensors(tensor: HV :: TV): Seq[Tensor[Any]] = {
        evH.value.tensors(tensor.head) ++
            evT.tensors(tensor.tail)
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

      override def decodeTensorFromOutput(
          output: HT :: TT,
          tensors: Seq[Tensor[Any]]
      ): (HV :: TV, Seq[Tensor[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeTensorFromOutput(output.head, tensors)
        val (tailOut, tailRemaining) = evT.decodeTensorFromOutput(output.tail, headRemaining)
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
          converter: NestedStructure.Converter
      ): HT :: TT = {
        evH.value.map(value.head, shape.map(_.head), converter) ::
            evT.map(value.tail, shape.map(_.tail), converter)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PV <: Product, PD <: Product, PS <: Product, HT <: HList, HV <: HList, HD <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: NestedStructure.Aux[HT, HV, HD, HS],
      tuplerV: Tupler.Aux[HV, PV],
      tuplerD: Tupler.Aux[HD, PD],
      tuplerS: Tupler.Aux[HS, PS],
      genV: Generic.Aux[PV, HV],
      genD: Generic.Aux[PD, HD],
      genS: Generic.Aux[PS, HS]
  ): NestedStructure.Aux[PT, PV, PD, PS] = {
    new NestedStructure[PT] {
      override type V = PV
      override type D = PD
      override type S = PS

      override def sizeFromOutput(output: PT): Int = {
        evT.sizeFromOutput(genT.to(output))
      }

      override def sizeFromDataType(dataType: PD): Int = {
        evT.sizeFromDataType(genD.to(dataType))
      }

      override def dataTypeFromOutput(output: PT): PD = {
        tuplerD(evT.dataTypeFromOutput(genT.to(output)))
      }

      override def shapeFromOutput(output: PT): PS = {
        tuplerS(evT.shapeFromOutput(genT.to(output)))
      }

      override def outputFromTensor(tensor: PV): PT = {
        genT.from(evT.outputFromTensor(genV.to(tensor)))
      }

      override def dataTypeFromTensor(tensor: PV): PD = {
        tuplerD(evT.dataTypeFromTensor(genV.to(tensor)))
      }

      override def shapeFromTensor(tensor: PV): PS = {
        tuplerS(evT.shapeFromTensor(genV.to(tensor)))
      }

      override def outputs(output: PT): Seq[Output[Any]] = {
        evT.outputs(genT.to(output))
      }

      override def tensors(tensor: PV): Seq[Tensor[Any]] = {
        evT.tensors(genV.to(tensor))
      }

      override def dataTypes(dataType: PD): Seq[DataType[Any]] = {
        evT.dataTypes(genD.to(dataType))
      }

      override def shapes(shape: PS): Seq[Shape] = {
        evT.shapes(genS.to(shape))
      }

      override def decodeOutputFromOutput(
          output: PT,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.decodeOutputFromOutput(genT.to(output), outputs)
        (genT.from(out), remaining)
      }

      override def decodeTensorFromOutput(
          output: PT,
          tensors: Seq[Tensor[Any]]
      ): (PV, Seq[Tensor[Any]]) = {
        val (out, remaining) = evT.decodeTensorFromOutput(genT.to(output), tensors)
        (tuplerV(out), remaining)
      }

      override def decodeDataTypeFromOutput(
          output: PT,
          dataTypes: Seq[DataType[Any]]
      ): (PD, Seq[DataType[Any]]) = {
        val (out, remaining) = evT.decodeDataTypeFromOutput(genT.to(output), dataTypes)
        (genD.from(out), remaining)
      }

      override def decodeShapeFromOutput(
          output: PT,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evT.decodeShapeFromOutput(genT.to(output), shapes)
        (genS.from(out), remaining)
      }

      override def decodeOutputFromDataType(
          dataType: PD,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.decodeOutputFromDataType(genD.to(dataType), outputs)
        (genT.from(out), remaining)
      }

      override def decodeDataTypeFromDataType(
          dataType: PD,
          dataTypes: Seq[DataType[Any]]
      ): (PD, Seq[DataType[Any]]) = {
        val (out, remaining) = evT.decodeDataTypeFromDataType(genD.to(dataType), dataTypes)
        (tuplerD(out), remaining)
      }

      override def decodeShapeFromDataType(
          dataType: PD,
          shapes: Seq[Shape]
      ): (PS, Seq[Shape]) = {
        val (out, remaining) = evT.decodeShapeFromDataType(genD.to(dataType), shapes)
        (tuplerS(out), remaining)
      }

      override def map(
          value: PT,
          shape: Option[PS],
          converter: NestedStructure.Converter
      ): PT = {
        genT.from(evT.map(genT.to(value), shape.map(genS.to), converter))
      }
    }
  }

  implicit def fromCoproduct[HT, HV, HD, HS, TT <: Coproduct, TV <: Coproduct, TD <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[NestedStructure.Aux[HT, HV, HD, HS]],
      evT: NestedStructure.Aux[TT, TV, TD, TS]
  ): NestedStructure.Aux[HT :+: TT, HV :+: TV, HD :+: TD, HS :+: TS] = {
    new NestedStructure[HT :+: TT] {
      override type V = HV :+: TV
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

      override def dataTypeFromOutput(output: HT :+: TT): HD :+: TD = {
        output match {
          case Inl(h) => Inl(evH.value.dataTypeFromOutput(h))
          case Inr(t) => Inr(evT.dataTypeFromOutput(t))
        }
      }

      override def shapeFromOutput(output: HT :+: TT): HS :+: TS = {
        output match {
          case Inl(h) => Inl(evH.value.shapeFromOutput(h))
          case Inr(t) => Inr(evT.shapeFromOutput(t))
        }
      }

      override def outputFromTensor(tensor: HV :+: TV): HT :+: TT = {
        tensor match {
          case Inl(h) => Inl(evH.value.outputFromTensor(h))
          case Inr(t) => Inr(evT.outputFromTensor(t))
        }
      }

      override def dataTypeFromTensor(tensor: HV :+: TV): HD :+: TD = {
        tensor match {
          case Inl(h) => Inl(evH.value.dataTypeFromTensor(h))
          case Inr(t) => Inr(evT.dataTypeFromTensor(t))
        }
      }

      override def shapeFromTensor(tensor: HV :+: TV): HS :+: TS = {
        tensor match {
          case Inl(h) => Inl(evH.value.shapeFromTensor(h))
          case Inr(t) => Inr(evT.shapeFromTensor(t))
        }
      }

      override def outputs(output: HT :+: TT): Seq[Output[Any]] = {
        output match {
          case Inl(h) => evH.value.outputs(h)
          case Inr(t) => evT.outputs(t)
        }
      }

      override def tensors(tensor: HV :+: TV): Seq[Tensor[Any]] = {
        tensor match {
          case Inl(h) => evH.value.tensors(h)
          case Inr(t) => evT.tensors(t)
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

      override def decodeTensorFromOutput(
          output: HT :+: TT,
          tensors: Seq[Tensor[Any]]
      ): (HV :+: TV, Seq[Tensor[Any]]) = {
        output match {
          case Inl(h) =>
            val (result, remaining) = evH.value.decodeTensorFromOutput(h, tensors)
            (Inl(result), remaining)
          case Inr(t) =>
            val (result, remaining) = evT.decodeTensorFromOutput(t, tensors)
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
          converter: NestedStructure.Converter
      ): HT :+: TT = {
        value match {
          case Inl(hv) => Inl(evH.value.map(hv, shape.map(_.asInstanceOf[Inl[HS, TS]].head), converter))
          case Inr(tv) => Inr(evT.map(tv, shape.map(_.asInstanceOf[Inr[HS, TS]].tail), converter))
          case _ => throw new IllegalStateException("Something went wrong while deriving implicit evidence.")
        }
      }
    }
  }

  implicit def fromZero[T, V, D, S](implicit
      ev: Zero.Aux[T, V, D, S]
  ): NestedStructure.Aux[T, V, D, S] = {
    ev.structure
  }
}
