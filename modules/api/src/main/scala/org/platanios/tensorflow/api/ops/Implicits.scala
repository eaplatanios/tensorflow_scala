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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

import scala.collection.{TraversableLike, breakOut}

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends Priority3Implicits
        with control_flow.Implicits
        with Basic.Implicits
        with Cast.Implicits
        with Clip.Implicits
        with Embedding.Implicits
        with Math.Implicits
        with NN.Implicits
        with Sparse.Implicits
        with Statistics.Implicits
        with Text.Implicits {
  implicit def outputFromSupportedType[T](value: T)(implicit
      evSupported: SupportedType[T]
  ): Output[T] = {
    Basic.constant(Tensor.fill(evSupported.dataType, Shape())(value))
  }

  implicit def outputFromTensor[T](value: Tensor[T]): Output[T] = {
    Basic.constant(value)
  }

  implicit def outputFromShape(shape: Shape): Output[Long] = {
    Basic.constant(shape.toTensor)
  }

  implicit def outputFromRange(range: Range): Output[Int] = {
    Basic.constant(tensors.ops.Basic.stack(range.map(Tensor.fill(INT32, Shape()))))
  }

  implicit def outputFromOutputLike[T](value: OutputLike[T]): Output[T] = {
    value.toOutput
  }

  implicit def outputFromTensorArray[T](value: TensorArray[T]): Output[T] = {
    value.toOutput
  }

  implicit def outputFromVariable[T](value: Variable[T]): Output[T] = {
    value.toOutput
  }

  implicit def outputFromArray[T](value: Array[Output[T]]): Output[T] = {
    Basic.stack(value.toSeq)
  }

  implicit def outputFromTraversable[T, CC[A] <: TraversableLike[A, CC[A]]](value: CC[Output[T]]): Output[T] = {
    Basic.stack(value.toSeq)
  }
}

private[ops] trait Priority3Implicits
    extends Priority2Implicits {
  implicit def outputFromTensorConvertible[T, TC](value: TC)(implicit
      f: TC => Tensor[T]
  ): Output[T] = {
    Basic.constant(f(value))
  }

  implicit def outputFromConvertibleArray[T, V](value: Array[V])(implicit
      f: V => Output[T]
  ): Output[T] = {
    Basic.stack(value.toSeq.map(f))
  }

  implicit def outputFromConvertibleTraversable[T, V, CC[A] <: TraversableLike[A, CC[A]]](value: CC[V])(implicit
      f: V => Output[T]
  ): Output[T] = {
    Basic.stack(value.map(f)(breakOut))
  }
}

private[ops] trait Priority2Implicits
    extends Priority1Implicits {
  implicit def oIntToLong[V](value: V)(implicit f: V => Output[Int]): Output[Long] = {
    f(value).cast(INT64)
  }
}

private[ops] trait Priority1Implicits
    extends Priority0Implicits {
  implicit def oIntToFloat[V](value: V)(implicit f: V => Output[Int]): Output[Float] = {
    f(value).cast(FLOAT32)
  }
}

private[ops] trait Priority0Implicits {
  implicit def oIntToDouble[V](value: V)(implicit f: V => Output[Int]): Output[Double] = {
    f(value).cast(FLOAT64)
  }
}
