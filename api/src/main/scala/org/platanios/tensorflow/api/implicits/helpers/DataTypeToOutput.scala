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

import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.types.DataType

import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike}

/** Type trait used to map structures of data types to structures of symbolic tensors (i.e., outputs).
  *
  * @author Emmanouil Antonios Platanios
  */
trait DataTypeToOutput[D] {
  type OutputType
}

object DataTypeToOutput {
  type Aux[D, O] = DataTypeToOutput[D] {
    type OutputType = O
  }

  implicit val dataTypeToOutput: Aux[DataType, Output] = new DataTypeToOutput[DataType] {
    override type OutputType = Output
  }

  implicit def arrayDataTypeToOutput[D, O](implicit ev: Aux[D, O]): Aux[Array[D], Array[O]] = {
    new DataTypeToOutput[Array[D]] {
      override type OutputType = Array[O]
    }
  }

  implicit def seqDataTypeToOutput[D, O, CC[A] <: SeqLike[A, CC[A]]](implicit ev: Aux[D, O]): Aux[CC[D], CC[O]] = {
    new DataTypeToOutput[CC[D]] {
      override type OutputType = CC[O]
    }
  }

  implicit def mapDataTypeToOutput[K, D, O, CC[CK, CV] <: MapLike[CK, CV, CC[CK, CV]] with Map[CK, CV]](implicit
      ev: Aux[D, O]
  ): Aux[CC[K, D], CC[K, O]] = new DataTypeToOutput[CC[K, D]] {
    override type OutputType = CC[K, O]
  }

  implicit val hnilDataTypeToOutput: Aux[HNil, HNil] = new DataTypeToOutput[HNil] {
    override type OutputType = HNil
  }

  implicit def recursiveDataTypeToOutputConstructor[HD, HO, TD <: HList, TO <: HList](implicit
      evHead: Lazy[Aux[HD, HO]],
      evTail: Aux[TD, TO]
  ): Aux[HD :: TD, HO :: TO] = new DataTypeToOutput[HD :: TD] {
    override type OutputType = HO :: TO
  }

  implicit def productDataTypeToOutputConstructor[PD <: Product, PO <: Product, HD <: HList, HO <: HList](implicit
      genO: Generic.Aux[PD, HD],
      evH: Aux[HD, HO],
      tuplerT: Tupler.Aux[HO, PO]
  ): Aux[PD, PO] = new DataTypeToOutput[PD] {
    override type OutputType = PO
  }
}
