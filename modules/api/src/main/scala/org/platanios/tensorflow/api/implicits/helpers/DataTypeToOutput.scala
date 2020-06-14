/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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
import org.platanios.tensorflow.api.ops.Output

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataTypeToOutput[D] {
  type O

  def dataTypeStructure: DataTypeStructure[D]
}

object DataTypeToOutput extends DataTypeToOutputLowPriorityImplicits {
  def apply[D](implicit ev: DataTypeToOutput[D]): Aux[D, ev.O] = {
    ev.asInstanceOf[Aux[D, ev.O]]
  }

  type Aux[D, OO] = DataTypeToOutput[D] {
    type O = OO
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new DataTypeToOutput[Unit] {
      override type O = Unit

      override def dataTypeStructure: DataTypeStructure[Unit] = {
        DataTypeStructure.fromUnit
      }
    }
  }

  implicit def fromDataType[T]: Aux[DataType[T], Output[T]] = {
    new DataTypeToOutput[DataType[T]] {
      override type O = Output[T]

      override def dataTypeStructure: DataTypeStructure[DataType[T]] = {
        DataTypeStructure.fromOutput[T]
      }
    }
  }

  implicit def fromOption[D](implicit
      ev: DataTypeToOutput[D]
  ): DataTypeToOutput.Aux[Option[D], Option[ev.O]] = {
    new DataTypeToOutput[Option[D]] {
      override type O = Option[ev.O]

      override def dataTypeStructure: DataTypeStructure[Option[D]] = {
        DataTypeStructure.fromOption[D](ev.dataTypeStructure)
      }
    }
  }

  implicit def fromSeq[D](implicit
      ev: DataTypeToOutput[D]
  ): DataTypeToOutput.Aux[Seq[D], Seq[ev.O]] = {
    new DataTypeToOutput[Seq[D]] {
      override type O = Seq[ev.O]

      override def dataTypeStructure: DataTypeStructure[Seq[D]] = {
        DataTypeStructure.fromSeq[D](ev.dataTypeStructure)
      }
    }
  }

  implicit def fromMap[K, D](implicit
      ev: DataTypeToOutput[D]
  ): DataTypeToOutput.Aux[Map[K, D], Map[K, ev.O]] = {
    new DataTypeToOutput[Map[K, D]] {
      override type O = Map[K, ev.O]

      override def dataTypeStructure: DataTypeStructure[Map[K, D]] = {
        DataTypeStructure.fromMap[K, D](ev.dataTypeStructure)
      }
    }
  }

  implicit val fromHNil: DataTypeToOutput.Aux[HNil, HNil] = {
    new DataTypeToOutput[HNil] {
      override type O = HNil

      override def dataTypeStructure: DataTypeStructure[HNil] = {
        DataTypeStructure.fromHNil
      }
    }
  }

  implicit def fromHList[HD, HO, TD <: HList, TO <: HList](implicit
      evH: Strict[DataTypeToOutput.Aux[HD, HO]],
      evT: Strict[DataTypeToOutput.Aux[TD, TO]]
  ): DataTypeToOutput.Aux[HD :: TD, HO :: TO] = {
    new DataTypeToOutput[HD :: TD] {
      override type O = HO :: TO

      override def dataTypeStructure: DataTypeStructure[HD :: TD] = {
        DataTypeStructure.fromHList[HD, TD](evH.value.dataTypeStructure, evT.value.dataTypeStructure)
      }
    }
  }

  implicit def fromKnownProduct[PD <: Product, PO <: Product, HD <: HList, HO <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evD: Strict[DataTypeToOutput.Aux[HD, HO]],
      genO: Generic.Aux[PO, HO]
  ): DataTypeToOutput.Aux[PD, PO] = {
    new DataTypeToOutput[PD] {
      override type O = PO

      override def dataTypeStructure: DataTypeStructure[PD] = {
        DataTypeStructure.fromProduct[PD, HD](genD, evD.value.dataTypeStructure)
      }
    }
  }
}

trait DataTypeToOutputLowPriorityImplicits {
  implicit def fromProduct[PD <: Product, PO <: Product, HD <: HList, HO <: HList](implicit
      genD: Generic.Aux[PD, HD],
      evD: Strict[DataTypeToOutput.Aux[HD, HO]],
      tuplerO: Tupler.Aux[HO, PO],
      genO: Generic.Aux[PO, HO]
  ): DataTypeToOutput.Aux[PD, PO] = {
    DataTypeToOutput.fromKnownProduct
  }
}
