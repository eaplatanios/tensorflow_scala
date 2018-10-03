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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tensors.ops.{Basic, Cast, Math, NN}
import org.platanios.tensorflow.api.types._

import scala.collection.{TraversableLike, breakOut}

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends Priority3Implicits
        with Basic.Implicits
        with Cast.Implicits
        with Math.Implicits
        with NN.Implicits {
  implicit def tensorFromSupportedType[T](value: T)(implicit
      evSupported: SupportedType[T]
  ): Tensor[T] = {
    Tensor.fill(evSupported.dataType, Shape())(value)
  }

  implicit def tensorFromTensorLike[T](value: TensorLike[T]): Tensor[T] = {
    value.toTensor
  }

  implicit def tensorFromShape(shape: Shape): Tensor[Long] = {
    shape.toTensor
  }

  implicit def tensorFromRange(range: Range): Tensor[Int] = {
    Basic.stack(range.map(Tensor.fill(INT32, Shape())))
  }

  implicit def tensorFromArray[T](value: Array[Tensor[T]]): Tensor[T] = {
    Basic.stack(value.toSeq)
  }

  implicit def tensorFromTraversable[T, CC[A] <: TraversableLike[A, CC[A]]](value: CC[Tensor[T]]): Tensor[T] = {
    Basic.stack(value.toSeq)
  }
}

private[tensors] trait Priority3Implicits
    extends Priority2Implicits {
  implicit def tensorFromConvertibleArray[T, V](value: Array[V])(implicit
      f: V => Tensor[T]
  ): Tensor[T] = {
    Basic.stack(value.toSeq.map(f))
  }

  implicit def tensorFromConvertibleTraversable[T, V, CC[A] <: TraversableLike[A, CC[A]]](value: CC[V])(implicit
      f: V => Tensor[T]
  ): Tensor[T] = {
    Basic.stack(value.map(f)(breakOut))
  }
}

private[tensors] trait Priority2Implicits
    extends Priority1Implicits {
  implicit def tIntToLong[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Long] = {
    f(value).cast(INT64)
  }
}

private[tensors] trait Priority1Implicits
    extends Priority0Implicits {
  implicit def tIntToFloat[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Float] = {
    f(value).cast(FLOAT32)
  }
}

private[tensors] trait Priority0Implicits {
  implicit def tIntToDouble[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Double] = {
    f(value).cast(FLOAT64)
  }
}
