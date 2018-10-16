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

import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops.{Output, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorToOutput[T] {
  type O
}

object TensorToOutput {
  type Aux[T, OO] = TensorToOutput[T] {
    type O = OO
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new TensorToOutput[Unit] {
      override type O = Unit
    }
  }

  implicit def fromTensor[T: TF]: Aux[Tensor[T], Output[T]] = {
    new TensorToOutput[Tensor[T]] {
      override type O = Output[T]
    }
  }

  implicit def fromTensorIndexedSlices[T]: Aux[TensorIndexedSlices[T], OutputIndexedSlices[T]] = {
    new TensorToOutput[TensorIndexedSlices[T]] {
      override type O = OutputIndexedSlices[T]
    }
  }

  implicit def fromSparseTensor[T]: Aux[SparseTensor[T], SparseOutput[T]] = {
    new TensorToOutput[SparseTensor[T]] {
      override type O = SparseOutput[T]
    }
  }

  implicit def fromOption[T, OO](implicit
      ev: TensorToOutput.Aux[T, OO]
  ): TensorToOutput.Aux[Option[T], Option[OO]] = {
    new TensorToOutput[Option[T]] {
      override type O = Option[OO]
    }
  }

  implicit def fromSeq[T, OO](implicit
      ev: TensorToOutput.Aux[T, OO]
  ): TensorToOutput.Aux[Seq[T], Seq[OO]] = {
    new TensorToOutput[Seq[T]] {
      override type O = Seq[OO]
    }
  }

  implicit def fromMap[K, T, OO](implicit
      ev: TensorToOutput.Aux[T, OO]
  ): TensorToOutput.Aux[Map[K, T], Map[K, OO]] = {
    new TensorToOutput[Map[K, T]] {
      override type O = Map[K, OO]
    }
  }

  implicit def fromNestedStructure[T, V, D, S](implicit
      evStructure: NestedStructure.Aux[T, V, D, S]
  ): TensorToOutput.Aux[V, T] = {
    new TensorToOutput[V] {
      override type O = T
    }
  }

  implicit val fromHNil: TensorToOutput.Aux[HNil, HNil] = {
    new TensorToOutput[HNil] {
      override type O = HNil
    }
  }

  implicit def fromHList[HT, HO, TT <: HList, TO <: HList](implicit
      evH: Strict[TensorToOutput.Aux[HT, HO]],
      evT: TensorToOutput.Aux[TT, TO]
  ): TensorToOutput.Aux[HT :: TT, HO :: TO] = {
    new TensorToOutput[HT :: TT] {
      override type O = HO :: TO
    }
  }

  implicit def fromProduct[PT <: Product, PO <: Product, HT <: HList, HO <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: TensorToOutput.Aux[HT, HO],
      tuplerO: Tupler.Aux[HO, PO],
      genO: Generic.Aux[PO, HO]
  ): TensorToOutput.Aux[PT, PO] = {
    new TensorToOutput[PT] {
      override type O = PO
    }
  }

  implicit def fromCoproduct[HT, HO, TT <: Coproduct, TO <: Coproduct](implicit
      evH: Strict[TensorToOutput.Aux[HT, HO]],
      evT: TensorToOutput.Aux[TT, TO]
  ): TensorToOutput.Aux[HT :+: TT, HO :+: TO] = {
    new TensorToOutput[HT :+: TT] {
      override type O = HO :+: TO
    }
  }
}
