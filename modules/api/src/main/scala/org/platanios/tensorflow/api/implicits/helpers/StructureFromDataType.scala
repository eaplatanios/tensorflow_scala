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
import org.platanios.tensorflow.api.core.types.{DataType, TF}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.tensors.Tensor

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike}

/** Type trait used to map structures of symbolic tensors to structures of tensors, data types, and shapes.
  *
  * @author Emmanouil Antonios Platanios
  */
trait StructureFromDataType[D]

object StructureFromDataType {
  private type SparseDataType[D] = (DataType[Long], D, DataType[Long])
  private type SparseShape = (Shape, Shape, Shape)

  type Aux[T, O, D, S] = StructureFromDataType[D]

  implicit val fromUnit: Aux[Unit, Unit, Unit, Unit] = {
    new StructureFromDataType[Unit] {}
  }

  implicit def fromOutput[T: TF]: Aux[Tensor[T], Output[T], DataType[T], Shape] = {
    new StructureFromDataType[DataType[T]] {}
  }

  implicit def fromOption[T, O, D, S](implicit
      ev: Aux[T, O, D, S]
  ): Aux[Option[T], Option[O], Option[D], Option[S]] = {
    new StructureFromDataType[Option[D]] {}
  }

  implicit def fromArray[T, O, D, S](implicit
      ev: Aux[T, O, D, S]
  ): Aux[Array[T], Array[O], Array[D], Array[S]] = {
    new StructureFromDataType[Array[D]] {}
  }

  implicit def fromSeq[T, O, D, S, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, O, D, S]
  ): Aux[CC[T], CC[O], CC[D], CC[S]] = {
    new StructureFromDataType[CC[D]] {}
  }

  implicit def fromMap[T, O, D, S, K, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[T, O, D, S]
  ): Aux[CC[K, T], CC[K, O], CC[K, D], CC[K, S]] = {
    new StructureFromDataType[CC[K, D]] {}
  }

  implicit val fromHNil: Aux[HNil, HNil, HNil, HNil] = {
    new StructureFromDataType[HNil] {}
  }

  implicit def fromHList[HT, HO, HD, HS, TT <: HList, TO <: HList, TD <: HList, TS <: HList](implicit
      evHead: Strict[Aux[HT, HO, HD, HS]],
      evTail: Aux[TT, TO, TD, TS]
  ): Aux[HT :: TT, HO :: TO, HD :: TD, HS :: TS] = {
    new StructureFromDataType[HD :: TD] {}
  }

  implicit def fromCoproduct[HT, HO, HD, HS, TT <: Coproduct, TO <: Coproduct, TD <: Coproduct, TS <: Coproduct](implicit
      evHead: Strict[Aux[HT, HO, HD, HS]],
      evTail: Aux[TT, TO, TD, TS]
  ): Aux[HT :+: TT, HO :+: TO, HD :+: TD, HS :+: TS] = {
    new StructureFromDataType[HD :+: TD] {}
  }

  implicit def fromProduct[PT <: Product, PO <: Product, PD <: Product, PS <: Product, HT <: HList, HO <: HList, HD <: HList, HS <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evH: Strict[Aux[HT, HO, HD, HS]],
      tuplerT: Tupler.Aux[HT, PT],
      tuplerO: Tupler.Aux[HO, PO],
      tuplerS: Tupler.Aux[HS, PS]
  ): Aux[PT, PO, PD, PS] = {
    new StructureFromDataType[PD] {}
  }
}
