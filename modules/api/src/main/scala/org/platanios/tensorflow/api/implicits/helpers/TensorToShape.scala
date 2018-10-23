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
import org.platanios.tensorflow.api.ops.TensorArray
import org.platanios.tensorflow.api.ops.data.Dataset
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import shapeless._
import shapeless.ops.hlist.Tupler

/** Type trait used to map structures of tensors to structures of symbolic tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait TensorToShape[T] {
  type S

  def shape(tensor: T): S
}

object TensorToShape {
  type Aux[T, SS] = TensorToShape[T] {
    type S = SS
  }

  implicit val fromUnit: Aux[Unit, Unit] = {
    new TensorToShape[Unit] {
      override type S = Unit

      override def shape(output: Unit): Unit = {
        ()
      }
    }
  }

  implicit def fromTensor[T]: Aux[Tensor[T], Shape] = {
    new TensorToShape[Tensor[T]] {
      override type S = Shape

      override def shape(output: Tensor[T]): Shape = {
        output.shape
      }
    }
  }

  implicit def fromTensorIndexedSlices[T]: Aux[TensorIndexedSlices[T], SparseShape] = {
    new TensorToShape[TensorIndexedSlices[T]] {
      override type S = SparseShape

      override def shape(output: TensorIndexedSlices[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }
    }
  }

  implicit def fromSparseTensor[T]: Aux[SparseTensor[T], SparseShape] = {
    new TensorToShape[SparseTensor[T]] {
      override type S = SparseShape

      override def shape(output: SparseTensor[T]): SparseShape = {
        (output.indices.shape, output.values.shape, output.denseShape.shape)
      }
    }
  }

  implicit def fromTensorArray[T]: Aux[TensorArray[T], Shape] = {
    new TensorToShape[TensorArray[T]] {
      override type S = Shape

      override def shape(output: TensorArray[T]): Shape = {
        Shape()
      }
    }
  }

  implicit def fromDataset[T]: Aux[Dataset[T], Shape] = {
    new TensorToShape[Dataset[T]] {
      override type S = Shape

      override def shape(output: Dataset[T]): Shape = {
        Shape()
      }
    }
  }

  implicit def fromOption[T](implicit
      ev: TensorToShape[T]
  ): TensorToShape.Aux[Option[T], Option[ev.S]] = {
    new TensorToShape[Option[T]] {
      override type S = Option[ev.S]

      override def shape(output: Option[T]): Option[ev.S] = {
        output.map(o => ev.shape(o))
      }
    }
  }

  implicit def fromSeq[T](implicit
      ev: TensorToShape[T]
  ): TensorToShape.Aux[Seq[T], Seq[ev.S]] = {
    new TensorToShape[Seq[T]] {
      override type S = Seq[ev.S]

      override def shape(output: Seq[T]): Seq[ev.S] = {
        output.map(o => ev.shape(o))
      }
    }
  }

  implicit def fromMap[K, T](implicit
      ev: TensorToShape[T]
  ): TensorToShape.Aux[Map[K, T], Map[K, ev.S]] = {
    new TensorToShape[Map[K, T]] {
      override type S = Map[K, ev.S]

      override def shape(output: Map[K, T]): Map[K, ev.S] = {
        output.mapValues(o => ev.shape(o))
      }
    }
  }

  implicit val fromHNil: TensorToShape.Aux[HNil, HNil] = {
    new TensorToShape[HNil] {
      override type S = HNil

      override def shape(output: HNil): HNil = {
        HNil
      }
    }
  }

  implicit def fromHList[HT, HS, TT <: HList, TS <: HList](implicit
      evH: Strict[TensorToShape.Aux[HT, HS]],
      evT: TensorToShape.Aux[TT, TS]
  ): TensorToShape.Aux[HT :: TT, HS :: TS] = {
    new TensorToShape[HT :: TT] {
      override type S = HS :: TS

      override def shape(output: HT :: TT): HS :: TS = {
        evH.value.shape(output.head) :: evT.shape(output.tail)
      }
    }
  }

  implicit def fromProduct[PT <: Product, PS <: Product, HT <: HList, HS <: HList](implicit
      genT: Generic.Aux[PT, HT],
      evT: TensorToShape.Aux[HT, HS],
      tuplerS: Tupler.Aux[HS, PS],
      genS: Generic.Aux[PS, HS]
  ): TensorToShape.Aux[PT, PS] = {
    new TensorToShape[PT] {
      override type S = PS

      override def shape(output: PT): PS = {
        tuplerS(evT.shape(genT.to(output)))
      }
    }
  }

  implicit def fromCoproduct[HT, HS, TT <: Coproduct, TS <: Coproduct](implicit
      evH: Strict[TensorToShape.Aux[HT, HS]],
      evT: TensorToShape.Aux[TT, TS]
  ): TensorToShape.Aux[HT :+: TT, HS :+: TS] = {
    new TensorToShape[HT :+: TT] {
      override type S = HS :+: TS

      override def shape(output: HT :+: TT): HS :+: TS = {
        output match {
          case Inl(h) => Inl(evH.value.shape(h))
          case Inr(t) => Inr(evT.shape(t))
        }
      }
    }
  }
}
