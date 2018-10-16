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

import org.platanios.tensorflow.api.core.types.{DataType, TF}
import org.platanios.tensorflow.api.ops.Output

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataTypeToOutput[D] {
  type O
}

object DataTypeToOutput {
  type Aux[D, OO] = DataTypeToOutput[D] {
    type O = OO
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new DataTypeToOutput[Unit] {
      override type O = Unit
    }
  }

  implicit def fromDataType[T: TF]: Aux[DataType[T], Output[T]] = {
    new DataTypeToOutput[DataType[T]] {
      override type O = Output[T]
    }
  }

  implicit def fromOption[D, OO](implicit
      ev: DataTypeToOutput.Aux[D, OO]
  ): DataTypeToOutput.Aux[Option[D], Option[OO]] = {
    new DataTypeToOutput[Option[D]] {
      override type O = Option[OO]
    }
  }

  implicit def fromSeq[D, OO](implicit
      ev: DataTypeToOutput.Aux[D, OO]
  ): DataTypeToOutput.Aux[Seq[D], Seq[OO]] = {
    new DataTypeToOutput[Seq[D]] {
      override type O = Seq[OO]
    }
  }

  implicit def fromMap[K, D, OO](implicit
      ev: DataTypeToOutput.Aux[D, OO]
  ): DataTypeToOutput.Aux[Map[K, D], Map[K, OO]] = {
    new DataTypeToOutput[Map[K, D]] {
      override type O = Map[K, OO]
    }
  }

  implicit def fromNestedStructure[T, V, D, S](implicit
      evStructure: NestedStructure.Aux[T, V, D, S]
  ): DataTypeToOutput.Aux[D, T] = {
    new DataTypeToOutput[D] {
      override type O = T
    }
  }

  implicit val fromHNil: DataTypeToOutput.Aux[HNil, HNil] = {
    new DataTypeToOutput[HNil] {
      override type O = HNil
    }
  }

  implicit def fromHList[HD, HO, TD <: HList, TO <: HList](implicit
      evH: Strict[DataTypeToOutput.Aux[HD, HO]],
      evT: DataTypeToOutput.Aux[TD, TO]
  ): DataTypeToOutput.Aux[HD :: TD, HO :: TO] = {
    new DataTypeToOutput[HD :: TD] {
      override type O = HO :: TO
    }
  }

  implicit def fromProduct[PD <: Product, PO <: Product, HD <: HList, HO <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evD: DataTypeToOutput.Aux[HD, HO],
      tuplerO: Tupler.Aux[HO, PO],
      genO: Generic.Aux[PO, HO]
  ): DataTypeToOutput.Aux[PD, PO] = {
    new DataTypeToOutput[PD] {
      override type O = PO
    }
  }

  implicit def fromCoproduct[HD, HO, TD <: Coproduct, TO <: Coproduct](implicit
      evH: Strict[DataTypeToOutput.Aux[HD, HO]],
      evT: DataTypeToOutput.Aux[TD, TO]
  ): DataTypeToOutput.Aux[HD :+: TD, HO :+: TO] = {
    new DataTypeToOutput[HD :+: TD] {
      override type O = HO :+: TO
    }
  }
}
