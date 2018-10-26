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

import org.platanios.tensorflow.api.core.types.DataType
import org.platanios.tensorflow.api.utilities.Collections

import shapeless._

/** Type trait used to represent nested structures over data types.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataTypeStructure[D] {
  def size(dataType: D): Int
  def dataTypes(dataType: D): Seq[DataType[Any]]
  def decodeDataType(dataType: D, dataTypes: Seq[DataType[Any]]): (D, Seq[DataType[Any]])
}

object DataTypeStructure {
  def apply[D](implicit ev: DataTypeStructure[D]): DataTypeStructure[D] = {
    ev
  }

  implicit val fromUnit: DataTypeStructure[Unit] = {
    new DataTypeStructure[Unit] {
      override def size(dataType: Unit): Int = {
        0
      }

      override def dataTypes(dataType: Unit): Seq[DataType[Any]] = {
        Seq.empty
      }

      override def decodeDataType(
          dataType: Unit,
          dataTypes: Seq[DataType[Any]]
      ): (Unit, Seq[DataType[Any]]) = {
        ((), dataTypes)
      }
    }
  }

  implicit def fromOutput[T]: DataTypeStructure[DataType[T]] = {
    new DataTypeStructure[DataType[T]] {
      override def size(dataType: DataType[T]): Int = {
        1
      }

      override def dataTypes(dataType: DataType[T]): Seq[DataType[Any]] = {
        Seq(dataType)
      }

      override def decodeDataType(
          dataType: DataType[T],
          dataTypes: Seq[DataType[Any]]
      ): (DataType[T], Seq[DataType[Any]]) = {
        (dataTypes.head.asInstanceOf[DataType[T]], dataTypes.tail)
      }
    }
  }

  implicit def fromOption[D](implicit ev: DataTypeStructure[D]): DataTypeStructure[Option[D]] = {
    new DataTypeStructure[Option[D]] {
      override def size(dataType: Option[D]): Int = {
        dataType.map(ev.size).sum
      }

      override def dataTypes(dataType: Option[D]): Seq[DataType[Any]] = {
        dataType.toSeq.flatMap(ev.dataTypes)
      }

      override def decodeDataType(
          dataType: Option[D],
          dataTypes: Seq[DataType[Any]]
      ): (Option[D], Seq[DataType[Any]]) = {
        dataType match {
          case Some(d) =>
            val (result, remaining) = ev.decodeDataType(d, dataTypes)
            (Some(result), remaining)
          case None => (None, dataTypes)
        }
      }
    }
  }

  implicit def fromSeq[D](implicit
      ev: DataTypeStructure[D]
  ): DataTypeStructure[Seq[D]] = {
    new DataTypeStructure[Seq[D]] {
      override def size(dataType: Seq[D]): Int = {
        dataType.map(ev.size).sum
      }

      override def dataTypes(dataType: Seq[D]): Seq[DataType[Any]] = {
        dataType.flatMap(ev.dataTypes)
      }

      override def decodeDataType(
          dataType: Seq[D],
          dataTypes: Seq[DataType[Any]]
      ): (Seq[D], Seq[DataType[Any]]) = {
        val n = size(dataType)
        (dataType
            .zip(Collections.segment(dataTypes.take(n), dataType.map(ev.size)))
            .map(f => ev.decodeDataType(f._1, f._2)._1), dataTypes.drop(n))
      }
    }
  }

  implicit def fromMap[K, D](implicit
      ev: DataTypeStructure[D]
  ): DataTypeStructure[Map[K, D]] = {
    new DataTypeStructure[Map[K, D]] {
      override def size(dataType: Map[K, D]): Int = {
        dataType.values.map(ev.size).sum
      }

      override def dataTypes(dataType: Map[K, D]): Seq[DataType[Any]] = {
        dataType.values.flatMap(ev.dataTypes).toSeq
      }

      override def decodeDataType(
          dataType: Map[K, D],
          dataTypes: Seq[DataType[Any]]
      ): (Map[K, D], Seq[DataType[Any]]) = {
        val n = size(dataType)
        (dataType.keys.zip(
          dataType.values
              .zip(Collections.segment(dataTypes.take(n), dataType.values.map(ev.size).toSeq))
              .map(f => ev.decodeDataType(f._1, f._2)._1)).toMap, dataTypes.drop(n))
      }
    }
  }

  implicit val fromHNil: DataTypeStructure[HNil] = {
    new DataTypeStructure[HNil] {
      override def size(dataType: HNil): Int = {
        0
      }

      override def dataTypes(dataType: HNil): Seq[DataType[Any]] = {
        Seq.empty
      }

      override def decodeDataType(
          dataType: HNil,
          dataTypes: Seq[DataType[Any]]
      ): (HNil, Seq[DataType[Any]]) = {
        (HNil, dataTypes)
      }
    }
  }

  implicit def fromHList[HD, TD <: HList](implicit
      evH: Strict[DataTypeStructure[HD]],
      evT: DataTypeStructure[TD]
  ): DataTypeStructure[HD :: TD] = {
    new DataTypeStructure[HD :: TD] {
      override def size(dataType: HD :: TD): Int = {
        evH.value.size(dataType.head) + evT.size(dataType.tail)
      }

      override def dataTypes(dataType: HD :: TD): Seq[DataType[Any]] = {
        evH.value.dataTypes(dataType.head) ++ evT.dataTypes(dataType.tail)
      }

      override def decodeDataType(
          dataType: HD :: TD,
          dataTypes: Seq[DataType[Any]]
      ): (HD :: TD, Seq[DataType[Any]]) = {
        val (headOut, headRemaining) = evH.value.decodeDataType(dataType.head, dataTypes)
        val (tailOut, tailRemaining) = evT.decodeDataType(dataType.tail, headRemaining)
        (headOut :: tailOut, tailRemaining)
      }
    }
  }

  implicit def fromProduct[PD <: Product, HD <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evD: Strict[DataTypeStructure[HD]]
  ): DataTypeStructure[PD] = {
    new DataTypeStructure[PD] {
      override def size(dataType: PD): Int = {
        evD.value.size(genD.to(dataType))
      }

      override def dataTypes(dataType: PD): Seq[DataType[Any]] = {
        evD.value.dataTypes(genD.to(dataType))
      }

      override def decodeDataType(
          dataType: PD,
          dataTypes: Seq[DataType[Any]]
      ): (PD, Seq[DataType[Any]]) = {
        val (out, remaining) = evD.value.decodeDataType(genD.to(dataType), dataTypes)
        (genD.from(out), remaining)
      }
    }
  }
}
