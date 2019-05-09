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

package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.tensors.ops.{Basic, Math, NN}

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends Priority4Implicits
        with Basic.Implicits
        with Math.Implicits
        with NN.Implicits {
  implicit def tensorFromSupportedType[T: TF](value: T): Tensor[T] = {
    if (value == null) null else Tensor.fill[T](Shape())(value)
  }

  implicit def tensorFromTensorLike[T: TF](
      value: TensorLike[T]
  ): Tensor[T] = {
    if (value == null) null else value.toTensor
  }

  implicit def tensorFromShape(shape: Shape): Tensor[Int] = {
    if (shape == null) null else shape.toTensor
  }

  implicit def tensorFromRange(range: Range): Tensor[Int] = {
    if (range == null) null else Basic.stack(range.map(Tensor.fill[Int](Shape())))
  }
}

private[tensors] trait Priority4Implicits extends Priority3Implicits {
  implicit def tensorFromArray[T: TF](
      value: Array[Tensor[T]]
  ): Tensor[T] = {
    if (value == null) null else Basic.stack(value.toSeq)
  }

  implicit def tensorFromSeq[T: TF](
      value: Seq[Tensor[T]]
  ): Tensor[T] = {
    if (value == null) null else Basic.stack(value)
  }
}

private[tensors] trait Priority3Implicits extends Priority2Implicits {
  implicit def tensorFromConvertibleArray[T, V](
      value: Array[V]
  )(implicit
      f: V => Tensor[T],
      evTF: TF[T]
  ): Tensor[T] = {
    if (value == null) null else Basic.stack(value.toSeq.map(f))
  }

  implicit def tensorFromConvertibleSeq[T, V](
      value: Seq[V]
  )(implicit
      f: V => Tensor[T],
      evTF: TF[T]
  ): Tensor[T] = {
    if (value == null) null else Basic.stack(value.map(f))
  }
}

private[tensors] trait Priority2Implicits extends Priority1Implicits {
  implicit def tInt2Long[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Long] = f(value).toLong
  implicit def tLong2Float[V](value: V)(implicit f: V => Tensor[Long]): Tensor[Float] = f(value).toFloat
}

private[tensors] trait Priority1Implicits extends Priority0Implicits {
  implicit def tInt2Float[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Float] = f(value).toFloat
  implicit def tLong2Double[V](value: V)(implicit f: V => Tensor[Long]): Tensor[Double] = f(value).toDouble
}

private[tensors] trait Priority0Implicits {
  implicit def tInt2Double[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Double] = f(value).toDouble
}

// TODO: [TYPES] The following are disabled for now because they slow compilation down significantly.

//private[tensors] trait Priority7Implicits extends Priority6Implicits {
//  implicit def tUByte2Float[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[Float] = f(value).toFloat
//}
//
//private[tensors] trait Priority6Implicits extends Priority5Implicits {
//  implicit def tUByte2Double[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[Double] = f(value).toDouble
//}
//
//private[tensors] trait Priority5Implicits extends Priority4Implicits {
//  implicit def tUByte2Short[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[Short] = f(value).castTo[Short]
//  implicit def tUShort2Float[V](value: V)(implicit f: V => Tensor[UShort]): Tensor[Float] = f(value).toFloat
//}
//
//private[tensors] trait Priority4Implicits extends Priority3Implicits {
//  implicit def tByte2Float[V](value: V)(implicit f: V => Tensor[Byte]): Tensor[Float] = f(value).toFloat
//  implicit def tUByte2Int[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[Int] = f(value).toInt
//  implicit def tUShort2Double[V](value: V)(implicit f: V => Tensor[UShort]): Tensor[Double] = f(value).toDouble
//}
//
//private[tensors] trait Priority3Implicits extends Priority2Implicits {
//  implicit def tByte2Double[V](value: V)(implicit f: V => Tensor[Byte]): Tensor[Double] = f(value).toDouble
//  implicit def tShort2Float[V](value: V)(implicit f: V => Tensor[Short]): Tensor[Float] = f(value).toFloat
//  implicit def tUByte2Long[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[Long] = f(value).toLong
//  implicit def tUShort2Int[V](value: V)(implicit f: V => Tensor[UShort]): Tensor[Int] = f(value).toInt
//  implicit def tUInt2Float[V](value: V)(implicit f: V => Tensor[UInt]): Tensor[Float] = f(value).toFloat
//}
//
//private[tensors] trait Priority2Implicits extends Priority1Implicits {
//  implicit def tByte2Short[V](value: V)(implicit f: V => Tensor[Byte]): Tensor[Short] = f(value).castTo[Short]
//  implicit def tShort2Double[V](value: V)(implicit f: V => Tensor[Short]): Tensor[Double] = f(value).toDouble
//  implicit def tInt2Float[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Float] = f(value).toFloat
//  implicit def tUByte2UShort[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[UShort] = f(value).castTo[UShort]
//  implicit def tUShort2Long[V](value: V)(implicit f: V => Tensor[UShort]): Tensor[Long] = f(value).toLong
//  implicit def tUInt2Double[V](value: V)(implicit f: V => Tensor[UInt]): Tensor[Double] = f(value).toDouble
//}
//
//private[tensors] trait Priority1Implicits extends Priority0Implicits {
//  implicit def tByte2Int[V](value: V)(implicit f: V => Tensor[Byte]): Tensor[Int] = f(value).toInt
//  implicit def tShort2Int[V](value: V)(implicit f: V => Tensor[Short]): Tensor[Int] = f(value).toInt
//  implicit def tInt2Double[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Double] = f(value).toDouble
//  implicit def tLong2Float[V](value: V)(implicit f: V => Tensor[Long]): Tensor[Float] = f(value).toFloat
//  implicit def tUByte2UInt[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[UInt] = f(value).castTo[UInt]
//  implicit def tUShort2UInt[V](value: V)(implicit f: V => Tensor[UShort]): Tensor[UInt] = f(value).castTo[UInt]
//  implicit def tUInt2Long[V](value: V)(implicit f: V => Tensor[UInt]): Tensor[Long] = f(value).toLong
//  implicit def tULong2Float[V](value: V)(implicit f: V => Tensor[ULong]): Tensor[Float] = f(value).toFloat
//}
//
//private[tensors] trait Priority0Implicits {
//  implicit def tByte2Long[V](value: V)(implicit f: V => Tensor[Byte]): Tensor[Long] = f(value).toLong
//  implicit def tShort2Long[V](value: V)(implicit f: V => Tensor[Short]): Tensor[Long] = f(value).toLong
//  implicit def tInt2Long[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Long] = f(value).toLong
//  implicit def tLong2Double[V](value: V)(implicit f: V => Tensor[Long]): Tensor[Double] = f(value).toDouble
//  implicit def tUByte2ULong[V](value: V)(implicit f: V => Tensor[UByte]): Tensor[ULong] = f(value).castTo[ULong]
//  implicit def tUShort2ULong[V](value: V)(implicit f: V => Tensor[UShort]): Tensor[ULong] = f(value).castTo[ULong]
//  implicit def tUInt2ULong[V](value: V)(implicit f: V => Tensor[UInt]): Tensor[ULong] = f(value).castTo[ULong]
//  implicit def tULong2Double[V](value: V)(implicit f: V => Tensor[ULong]): Tensor[Double] = f(value).toDouble
//}
