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

import org.platanios.tensorflow.api.types.DataType

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike}
import scala.collection.generic.CanBuildFrom
import scala.reflect.ClassTag

/** Type trait used to map structures of data type aux types to structures of data types.
  *
  * @author Emmanouil Antonios Platanios
  */
trait DataTypeAuxToDataType[DA] {
  type DataTypeType
  def castDataType(dataType: DA): DataTypeType
}

object DataTypeAuxToDataType {
  type Aux[DA, D] = DataTypeAuxToDataType[DA] {
    type DataTypeType = D
  }

  implicit val dataTypeToDataType: Aux[DataType, DataType] = {
    new DataTypeAuxToDataType[DataType] {
      override type DataTypeType = DataType
      override def castDataType(dataType: DataType): DataType = dataType
    }
  }

  implicit def dataTypeAuxToDataType[DA]: Aux[DataType.Aux[DA], DataType] = {
    new DataTypeAuxToDataType[DataType.Aux[DA]] {
      override type DataTypeType = DataType
      override def castDataType(dataType: DataType.Aux[DA]): DataType = dataType
    }
  }

  implicit def arrayDataTypeAuxToDataType[DA, D: ClassTag](implicit ev: Aux[DA, D]): Aux[Array[DA], Array[D]] = {
    new DataTypeAuxToDataType[Array[DA]] {
      override type DataTypeType = Array[D]
      override def castDataType(dataType: Array[DA]): Array[D] = dataType.map(ev.castDataType)
    }
  }

  implicit def seqDataTypeAuxToDataType[DA, D, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[DA, D],
      cbfDAD: CanBuildFrom[CC[DA], D, CC[D]]
  ): Aux[CC[DA], CC[D]] = {
    new DataTypeAuxToDataType[CC[DA]] {
      override type DataTypeType = CC[D]
      override def castDataType(dataType: CC[DA]): CC[D] = dataType.map(ev.castDataType).to[CC](cbfDAD)
    }
  }

  implicit def mapDataTypeAuxToDataType[K, DA, D, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[DA, D]
  ): Aux[CC[K, DA], Map[K, D]] = new DataTypeAuxToDataType[CC[K, DA]] {
    // TODO: Return CC type instead of Map.
    override type DataTypeType = Map[K, D]
    override def castDataType(dataType: CC[K, DA]): Map[K, D] = dataType.mapValues(ev.castDataType)
  }

  implicit val hnilDataTypeAuxToDataType: Aux[HNil, HNil] = new DataTypeAuxToDataType[HNil] {
    override type DataTypeType = HNil
    override def castDataType(dataType: HNil): HNil = HNil
  }

  implicit def recursiveDataTypeAuxToDataTypeConstructor[HDA, HD, TDA <: HList, TD <: HList](implicit
      evHead: Lazy[Aux[HDA, HD]],
      evTail: Aux[TDA, TD]
  ): Aux[HDA :: TDA, HD :: TD] = new DataTypeAuxToDataType[HDA :: TDA] {
    override type DataTypeType = HD :: TD
    override def castDataType(dataType: HDA :: TDA): HD :: TD = {
      evHead.value.castDataType(dataType.head) :: evTail.castDataType(dataType.tail)
    }
  }

  implicit def productDataTypeAuxToDataTypeConstructor[
  PDA <: Product, PD <: Product, HDA <: HList, HD <: HList](implicit
      genO: Generic.Aux[PDA, HDA],
      evH: Aux[HDA, HD],
      tuplerT: Tupler.Aux[HD, PD]
  ): Aux[PDA, PD] = new DataTypeAuxToDataType[PDA] {
    override type DataTypeType = PD
    override def castDataType(dataType: PDA): PD = {
      tuplerT(evH.castDataType(genO.to(dataType)))
    }
  }
}
