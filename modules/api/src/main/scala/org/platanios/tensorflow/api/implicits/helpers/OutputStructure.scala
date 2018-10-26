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
trait OutputStructure[T] {
  def size(output: T): Int
  def outputs(output: T): Seq[Output[Any]]
  def decodeOutput(output: T, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])
  def map(value: T, converter: OutputStructure.Converter): T
}

object OutputStructure {
  trait Converter {
    def apply[T](value: Output[T], shape: Option[Shape]): Output[T] = value
    def apply[T](value: OutputIndexedSlices[T], shape: Option[SparseShape]): OutputIndexedSlices[T] = value
    def apply[T](value: SparseOutput[T], shape: Option[SparseShape]): SparseOutput[T] = value
    def apply[T](value: TensorArray[T], shape: Option[Shape]): TensorArray[T] = value
    def apply[T](value: Dataset[T], shape: Option[Shape]): Dataset[T] = value
  }

  def apply[T](implicit ev: OutputStructure[T]): OutputStructure[T] = {
    ev
  }

  implicit val fromUnit: OutputStructure[Unit] = {
    new OutputStructure[Unit] {
      override def size(output: Unit): Int = {
        0
      }

      override def outputs(output: Unit): Seq[Output[Any]] = {
        Seq.empty
      }

      override def decodeOutput(
          output: Unit,
          outputs: Seq[Output[Any]]
      ): (Unit, Seq[Output[Any]]) = {
        ((), outputs)
      }

      def map(
          value: Unit,
          converter: OutputStructure.Converter
      ): Unit = {
        ()
      }
    }
  }

  implicit def fromOutput[T]: OutputStructure[Output[T]] = {
    new OutputStructure[Output[T]] {
      override def size(output: Output[T]): Int = {
        1
      }

      override def outputs(output: Output[T]): Seq[Output[Any]] = {
        Seq(output)
      }

      override def decodeOutput(
          output: Output[T],
          outputs: Seq[Output[Any]]
      ): (Output[T], Seq[Output[Any]]) = {
        (outputs.head.asInstanceOf[Output[T]], outputs.tail)
      }

      override def map(
          value: Output[T],
          converter: OutputStructure.Converter
      ): Output[T] = {
        converter[T](value, shape = None)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: OutputStructure[OutputIndexedSlices[T]] = {
    new OutputStructure[OutputIndexedSlices[T]] {
      override def size(output: OutputIndexedSlices[T]): Int = {
        3
      }

      override def outputs(output: OutputIndexedSlices[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def decodeOutput(
          output: OutputIndexedSlices[T],
          outputs: Seq[Output[Any]]
      ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
        (OutputIndexedSlices[T](
          indices = outputs(0).asInstanceOf[Output[Int]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Int]]
        ), outputs.drop(3))
      }

      override def map(
          value: OutputIndexedSlices[T],
          converter: OutputStructure.Converter
      ): OutputIndexedSlices[T] = {
        converter[T](value, shape = None)
      }
    }
  }

  implicit def fromSparseOutput[T]: OutputStructure[SparseOutput[T]] = {
    new OutputStructure[SparseOutput[T]] {
      override def size(output: SparseOutput[T]): Int = {
        3
      }

      override def outputs(output: SparseOutput[T]): Seq[Output[Any]] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def decodeOutput(
          output: SparseOutput[T],
          outputs: Seq[Output[Any]]
      ): (SparseOutput[T], Seq[Output[Any]]) = {
        (SparseOutput[T](
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]]
        ), outputs.drop(3))
      }

      override def map(
          value: SparseOutput[T],
          converter: OutputStructure.Converter
      ): SparseOutput[T] = {
        converter[T](value, shape = None)
      }
    }
  }

  implicit def fromTensorArray[T]: OutputStructure[TensorArray[T]] = {
    new OutputStructure[TensorArray[T]] {
      override def size(output: TensorArray[T]): Int = {
        1
      }

      override def outputs(output: TensorArray[T]): Seq[Output[Any]] = {
        Seq(output.flow)
      }

      override def decodeOutput(
          output: TensorArray[T],
          outputs: Seq[Output[Any]]
      ): (TensorArray[T], Seq[Output[Any]]) = {
        val newTensorArray = output.copy(
          flow = outputs.head.asInstanceOf[Output[Float]]
        )(output.evTTF)
        // TODO: !!! [TENSOR_ARRAY] What about colocate with?
        (newTensorArray, outputs.tail)
      }

      override def map(
          value: TensorArray[T],
          converter: OutputStructure.Converter
      ): TensorArray[T] = {
        converter[T](value, shape = None)
      }
    }
  }

  implicit def fromDataset[T: OutputStructure, D, S](implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): OutputStructure[Dataset[T]] = {
    new OutputStructure[Dataset[T]] {
      override def size(output: Dataset[T]): Int = {
        1
      }

      override def outputs(arg: Dataset[T]): Seq[Output[Any]] = {
        Seq(arg.createHandle())
      }

      override def decodeOutput(
          output: Dataset[T],
          outputs: Seq[Output[Any]]
      ): (Dataset[T], Seq[Output[Any]]) = {
        (VariantDataset[T](
          handle = outputs.head.asInstanceOf[Output[Variant]],
          _outputDataTypes = output.outputDataTypes,
          _outputShapes = output.outputShapes
        ), outputs.drop(1))
      }

      override def map(
          value: Dataset[T],
          converter: OutputStructure.Converter
      ): Dataset[T] = {
        converter[T](value, shape = None)
      }
    }
  }

  implicit def fromOption[T](implicit ev: OutputStructure[T]): OutputStructure[Option[T]] = {
    new OutputStructure[Option[T]] {
      override def size(output: Option[T]): Int = {
        output.map(ev.size).sum
      }

      override def outputs(output: Option[T]): Seq[Output[Any]] = {
        output.toSeq.flatMap(ev.outputs)
      }

      override def decodeOutput(
          output: Option[T],
          outputs: Seq[Output[Any]]
      ): (Option[T], Seq[Output[Any]]) = {
        output match {
          case Some(o) =>
            val (result, remaining) = ev.decodeOutput(o, outputs)
            (Some(result), remaining)
          case None => (None, outputs)
        }
      }

      override def map(
          value: Option[T],
          converter: OutputStructure.Converter
      ): Option[T] = {
        value.map(ev.map(_, converter))
      }
    }
  }

  implicit def fromSeq[T](implicit ev: OutputStructure[T]): OutputStructure[Seq[T]] = {
    new OutputStructure[Seq[T]] {
      override def size(output: Seq[T]): Int = {
        output.map(ev.size).sum
      }

      override def outputs(output: Seq[T]): Seq[Output[Any]] = {
        output.flatMap(ev.outputs)
      }

      override def decodeOutput(
          output: Seq[T],
          outputs: Seq[Output[Any]]
      ): (Seq[T], Seq[Output[Any]]) = {
        val n = size(output)
        (output
            .zip(Collections.segment(outputs.take(n), output.map(ev.size)))
            .map(f => ev.decodeOutput(f._1, f._2)._1), outputs.drop(n))
      }

      override def map(
          value: Seq[T],
          converter: OutputStructure.Converter
      ): Seq[T] = {
        value.map(ev.map(_, converter))
      }
    }
  }

  implicit def fromMap[K, T](implicit ev: OutputStructure[T]): OutputStructure[Map[K, T]] = {
    new OutputStructure[Map[K, T]] {
      override def size(output: Map[K, T]): Int = {
        output.values.map(ev.size).sum
      }

      override def outputs(output: Map[K, T]): Seq[Output[Any]] = {
        output.values.flatMap(ev.outputs).toSeq
      }

      override def decodeOutput(
          output: Map[K, T],
          outputs: Seq[Output[Any]]
      ): (Map[K, T], Seq[Output[Any]]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(outputs.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.decodeOutput(f._1, f._2)._1)).toMap, outputs.drop(n))
      }

      override def map(
          value: Map[K, T],
          converter: OutputStructure.Converter
      ): Map[K, T] = {
        value.mapValues(ev.map(_, converter))
      }
    }
  }

  implicit val fromHNil: OutputStructure[HNil] = {
    new OutputStructure[HNil] {
      override def size(output: HNil): Int = {
        0
      }

      override def outputs(output: HNil): Seq[Output[Any]] = {
        Seq.empty
      }

      override def decodeOutput(
          output: HNil,
          outputs: Seq[Output[Any]]
      ): (HNil, Seq[Output[Any]]) = {
        (HNil, outputs)
      }

      override def map(
          value: HNil,
          converter: OutputStructure.Converter
      ): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, TT <: HList](implicit
      evH: Strict[OutputStructure[HT]],
      evT: OutputStructure[TT]
  ): OutputStructure[HT :: TT] = {
    new OutputStructure[HT :: TT] {
      override def size(output: HT :: TT): Int = {
        evH.value.size(output.head) +
            evT.size(output.tail)
      }

      override def outputs(output: HT :: TT): Seq[Output[Any]] = {
        evH.value.outputs(output.head) ++
            evT.outputs(output.tail)
      }

      override def decodeOutput(
          output: HT :: TT,
          outputs: Seq[Output[Any]]
      ): (HT :: TT, Seq[Output[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeOutput(output.head, outputs)
        val (tailOut, tailRemaining) = evT.decodeOutput(output.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }

      override def map(
          value: HT :: TT,
          converter: OutputStructure.Converter
      ): HT :: TT = {
        evH.value.map(value.head, converter) ::
            evT.map(value.tail, converter)
      }
    }
  }

  implicit def fromProduct[PT <: Product, HT <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[OutputStructure[HT]]
  ): OutputStructure[PT] = {
    new OutputStructure[PT] {
      override def size(output: PT): Int = {
        evT.value.size(genT.to(output))
      }

      override def outputs(output: PT): Seq[Output[Any]] = {
        evT.value.outputs(genT.to(output))
      }

      override def decodeOutput(
          output: PT,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.value.decodeOutput(genT.to(output), outputs)
        (genT.from(out), remaining)
      }

      override def map(
          value: PT,
          converter: OutputStructure.Converter
      ): PT = {
        genT.from(evT.value.map(genT.to(value), converter))
      }
    }
  }
}
