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

import org.platanios.tensorflow.api.core.types.{DataType, FLOAT32, INT32, INT64, VARIANT, Variant}
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput, TensorArray}
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of outputs to structures of data types.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait OutputToDataType[T] {
  type D

  def dataTypeStructure: DataTypeStructure[D]
  def sizeFromDataType(dataType: D): Int
  def dataType(output: T): D
  def decodeOutput(dataType: D, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])
}

object OutputToDataType {
  def apply[T](implicit ev: OutputToDataType[T]): OutputToDataType.Aux[T, ev.D] = {
    ev.asInstanceOf[OutputToDataType.Aux[T, ev.D]]
  }

  type Aux[T, DD] = OutputToDataType[T] {
    type D = DD
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new OutputToDataType[Unit] {
      override type D = Unit

      override def dataTypeStructure: DataTypeStructure[Unit] = {
        DataTypeStructure.fromUnit
      }

      override def sizeFromDataType(dataType: Unit): Int = {
        0
      }

      override def dataType(output: Unit): Unit = {
        ()
      }

      override def decodeOutput(
          dataType: Unit,
          outputs: Seq[Output[Any]]
      ): (Unit, Seq[Output[Any]]) = {
        ((), outputs)
      }
    }
  }

  implicit def fromOutput[T]: Aux[Output[T], DataType[T]] = {
    new OutputToDataType[Output[T]] {
      override type D = DataType[T]

      override def dataTypeStructure: DataTypeStructure[DataType[T]] = {
        DataTypeStructure.fromOutput[T]
      }

      override def sizeFromDataType(dataType: DataType[T]): Int = {
        3
      }

      override def dataType(output: Output[T]): DataType[T] = {
        output.dataType
      }

      override def decodeOutput(
          dataType: DataType[T],
          outputs: Seq[Output[Any]]
      ): (Output[T], Seq[Output[Any]]) = {
        (outputs.head.asInstanceOf[Output[T]], outputs.tail)
      }
    }
  }

  implicit def fromOutputIndexedSlices[T]: Aux[OutputIndexedSlices[T], IndexedSlicesDataType[T]] = {
    new OutputToDataType[OutputIndexedSlices[T]] {
      override type D = IndexedSlicesDataType[T]

      override def dataTypeStructure: DataTypeStructure[IndexedSlicesDataType[T]] = {
        implicitly[DataTypeStructure[IndexedSlicesDataType[T]]]
      }

      override def sizeFromDataType(dataType: IndexedSlicesDataType[T]): Int = {
        3
      }

      override def dataType(output: OutputIndexedSlices[T]): IndexedSlicesDataType[T] = {
        (INT32, output.dataType, INT32)
      }

      override def decodeOutput(
          dataType: IndexedSlicesDataType[T],
          outputs: Seq[Output[Any]]
      ): (OutputIndexedSlices[T], Seq[Output[Any]]) = {
        (OutputIndexedSlices[T](
          indices = outputs(0).asInstanceOf[Output[Int]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Int]]
        ), outputs.drop(3))
      }
    }
  }

  implicit def fromSparseOutput[T]: Aux[SparseOutput[T], SparseDataType[T]] = {
    new OutputToDataType[SparseOutput[T]] {
      override type D = SparseDataType[T]

      override def dataTypeStructure: DataTypeStructure[SparseDataType[T]] = {
        implicitly[DataTypeStructure[SparseDataType[T]]]
      }

      override def sizeFromDataType(dataType: SparseDataType[T]): Int = {
        3
      }

      override def dataType(output: SparseOutput[T]): SparseDataType[T] = {
        (INT64, output.dataType, INT64)
      }

      override def decodeOutput(
          dataType: SparseDataType[T],
          outputs: Seq[Output[Any]]
      ): (SparseOutput[T], Seq[Output[Any]]) = {
        (SparseOutput[T](
          indices = outputs(0).asInstanceOf[Output[Long]],
          values = outputs(1).asInstanceOf[Output[T]],
          denseShape = outputs(2).asInstanceOf[Output[Long]]
        ), outputs.drop(3))
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], DataType[Float]] = {
    new OutputToDataType[TensorArray[T]] {
      override type D = DataType[Float]

      override def dataTypeStructure: DataTypeStructure[DataType[Float]] = {
        DataTypeStructure.fromOutput[Float]
      }

      override def sizeFromDataType(dataType: DataType[Float]): Int = {
        1
      }

      override def dataType(output: TensorArray[T]): DataType[Float] = {
        FLOAT32
      }

      override def decodeOutput(
          dataType: DataType[Float],
          outputs: Seq[Output[Any]]
      ): (TensorArray[T], Seq[Output[Any]]) = {
        ???
      }
    }
  }

  implicit def fromDataset[T: OutputStructure, DD, SS](implicit
      evOutputToDataType: OutputToDataType.Aux[T, DD],
      evOutputToShape: OutputToShape.Aux[T, SS]
  ): Aux[Dataset[T], DataType[Variant]] = {
    new OutputToDataType[Dataset[T]] {
      override type D = DataType[Variant]

      override def dataTypeStructure: DataTypeStructure[DataType[Variant]] = {
        DataTypeStructure.fromOutput[Variant]
      }

      override def sizeFromDataType(dataType: DataType[Variant]): Int = {
        1
      }

      override def dataType(output: Dataset[T]): DataType[Variant] = {
        VARIANT
      }

      override def decodeOutput(
          dataType: DataType[Variant],
          outputs: Seq[Output[Any]]
      ): (Dataset[T], Seq[Output[Any]]) = {
        (VariantDataset[T](outputs.head.asInstanceOf[Output[Variant]]), outputs.drop(1))
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: OutputToDataType[T]
  ): OutputToDataType.Aux[Option[T], Option[ev.D]] = {
    new OutputToDataType[Option[T]] {
      override type D = Option[ev.D]

      override def dataTypeStructure: DataTypeStructure[Option[ev.D]] = {
        DataTypeStructure.fromOption[ev.D](ev.dataTypeStructure)
      }

      override def sizeFromDataType(dataType: Option[ev.D]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def dataType(output: Option[T]): Option[ev.D] = {
        output.map(o => ev.dataType(o))
      }

      override def decodeOutput(
          dataType: Option[ev.D],
          outputs: Seq[Output[Any]]
      ): (Option[T], Seq[Output[Any]]) = {
        dataType match {
          case Some(d) =>
            val (result, remaining) = ev.decodeOutput(d, outputs)
            (Some(result), remaining)
          case None => (None, outputs)
        }
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: OutputToDataType[T]
  ): OutputToDataType.Aux[Seq[T], Seq[ev.D]] = {
    new OutputToDataType[Seq[T]] {
      override type D = Seq[ev.D]

      override def dataTypeStructure: DataTypeStructure[Seq[ev.D]] = {
        DataTypeStructure.fromSeq[ev.D](ev.dataTypeStructure)
      }

      override def sizeFromDataType(dataType: Seq[ev.D]): Int = {
        dataType.map(ev.sizeFromDataType).sum
      }

      override def dataType(output: Seq[T]): Seq[ev.D] = {
        output.map(o => ev.dataType(o))
      }

      override def decodeOutput(
          dataType: Seq[ev.D],
          outputs: Seq[Output[Any]]
      ): (Seq[T], Seq[Output[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType
            .zip(Collections.segment(outputs.take(n), dataType.map(ev.sizeFromDataType)))
            .map(f => ev.decodeOutput(f._1, f._2)._1), outputs.drop(n))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: OutputToDataType[T]
  ): OutputToDataType.Aux[Map[K, T], Map[K, ev.D]] = {
    new OutputToDataType[Map[K, T]] {
      override type D = Map[K, ev.D]

      override def dataTypeStructure: DataTypeStructure[Map[K, ev.D]] = {
        DataTypeStructure.fromMap[K, ev.D](ev.dataTypeStructure)
      }

      override def sizeFromDataType(dataType: Map[K, ev.D]): Int = {
        dataType.values.map(ev.sizeFromDataType).sum
      }

      override def dataType(output: Map[K, T]): Map[K, ev.D] = {
        output.mapValues(o => ev.dataType(o))
      }

      override def decodeOutput(
          dataType: Map[K, ev.D],
          outputs: Seq[Output[Any]]
      ): (Map[K, T], Seq[Output[Any]]) = {
        val n = sizeFromDataType(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(outputs.take(n), dataType.values.map(ev.sizeFromDataType).toSeq))
              .map(f => ev.decodeOutput(f._1, f._2)._1)).toMap, outputs.drop(n))
      }
    }
  }

  implicit val fromHNil: OutputToDataType.Aux[HNil, HNil] = {
    new OutputToDataType[HNil] {
      override type D = HNil

      override def dataTypeStructure: DataTypeStructure[HNil] = {
        DataTypeStructure.fromHNil
      }

      override def sizeFromDataType(dataType: HNil): Int = {
        0
      }

      override def dataType(output: HNil): HNil = {
        HNil
      }

      override def decodeOutput(
          dataType: HNil,
          outputs: Seq[Output[Any]]
      ): (HNil, Seq[Output[Any]]) = {
        (HNil, outputs)
      }
    }
  }

  implicit def fromHList[HT, HD, TT <: HList, TD <: HList](implicit
      evH: Strict[OutputToDataType.Aux[HT, HD]],
      evT: Strict[OutputToDataType.Aux[TT, TD]]
  ): OutputToDataType.Aux[HT :: TT, HD :: TD] = {
    new OutputToDataType[HT :: TT] {
      override type D = HD :: TD

      override def dataTypeStructure: DataTypeStructure[HD :: TD] = {
        DataTypeStructure.fromHList[HD, TD](evH.value.dataTypeStructure, evT.value.dataTypeStructure)
      }

      override def sizeFromDataType(dataType: HD :: TD): Int = {
        evH.value.sizeFromDataType(dataType.head) + evT.value.sizeFromDataType(dataType.tail)
      }

      override def dataType(output: HT :: TT): HD :: TD = {
        evH.value.dataType(output.head) :: evT.value.dataType(output.tail)
      }

      override def decodeOutput(
          dataType: HD :: TD,
          outputs: Seq[Output[Any]]
      ): (HT :: TT, Seq[Output[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeOutput(dataType.head, outputs)
        val (tailOut, tailRemaining) = evT.value.decodeOutput(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PD <: Product, HT <: HList, HD <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: Strict[OutputToDataType.Aux[HT, HD]],
      tuplerD: Tupler.Aux[HD, PD],
      genD: Generic.Aux[PD, HD]
  ): OutputToDataType.Aux[PT, PD] = {
    new OutputToDataType[PT] {
      override type D = PD

      override def dataTypeStructure: DataTypeStructure[PD] = {
        DataTypeStructure.fromProduct[PD, HD](genD, evT.value.dataTypeStructure)
      }

      override def sizeFromDataType(dataType: PD): Int = {
        evT.value.sizeFromDataType(genD.to(dataType))
      }

      override def dataType(output: PT): PD = {
        tuplerD(evT.value.dataType(genT.to(output)))
      }

      override def decodeOutput(
          dataType: PD,
          outputs: Seq[Output[Any]]
      ): (PT, Seq[Output[Any]]) = {
        val (out, remaining) = evT.value.decodeOutput(genD.to(dataType), outputs)
        (genT.from(out), remaining)
      }
    }
  }
}
