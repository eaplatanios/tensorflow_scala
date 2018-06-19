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
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}
import org.platanios.tensorflow.api.types.{DataType, INT64}

import shapeless._

import scala.collection.{MapLike, SeqLike}

/** Type trait used to map structures of symbolic tensors to structures of tensors, data types, and shapes.
  *
  * @author Emmanouil Antonios Platanios
  */
trait StructureFromDataType[-D]

object StructureFromDataType {
  private type DataTypes3[D] = (INT64, D, INT64)
  private type Shapes3 = (Shape, Shape, Shape)

  type Aux[-T, -O, -D, -S] = StructureFromDataType[D]

  implicit def fromOutput[D <: DataType]: Aux[Tensor, Output, D, Shape] = {
    new StructureFromDataType[D] {}
  }

  //  implicit def fromOutputIndexedSlices[D <: DataType]: Aux[TensorIndexedSlices, OutputIndexedSlices, DataTypes3[D], Shapes3] = {
  //    new StructureFromDataType[DataTypes3[D]] {}
  //  }
  //
  //  implicit def fromSparseOutput[D <: DataType]: Aux[SparseTensor, SparseOutput, DataTypes3[D], Shapes3] = {
  //    new StructureFromDataType[DataTypes3[D]] {}
  //  }

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

  implicit def fromRecursiveStructure[HT, HO, HD, HS, TT <: HList, TO <: HList, TD <: HList, TS <: HList](implicit
      evHead: Lazy[Aux[HT, HO, HD, HS]],
      evTail: Aux[TT, TO, TD, TS]
  ): Aux[HT :: TT, HO :: TO, HD :: TD, HS :: TS] = {
    new StructureFromDataType[HD :: TD] {}
  }

  implicit def fromProduct[PT, PO, PD, PS, HT, HO, HD, HS](implicit
      genD: Generic.Aux[PD, HD],
      evH: Aux[HT, HO, HD, HS],
      genT: Generic.Aux[PT, HT],
      genO: Generic.Aux[PO, HO],
      genS: Generic.Aux[PS, HS]
  ): Aux[PT, PO, PD, PS] = {
    new StructureFromDataType[PD] {}
  }
}
