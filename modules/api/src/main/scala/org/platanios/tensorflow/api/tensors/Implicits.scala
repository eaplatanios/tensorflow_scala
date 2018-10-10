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
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.tensors.ops.{Basic, Cast, Math, NN}

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
  implicit def tensorFromSupportedType[T: TF](value: T): Tensor[T] = {
    Tensor.fill[T](Shape())(value)
  }

  implicit def tensorFromTensorLike[T: TF](
      value: TensorLike[T]
  ): Tensor[T] = {
    value.toTensor
  }

  implicit def tensorFromShape(shape: Shape): Tensor[Long] = {
    shape.toTensor
  }

  implicit def tensorFromRange(range: Range): Tensor[Int] = {
    Basic.stack(range.map(Tensor.fill[Int](Shape())))
  }

  implicit def tensorFromArray[T: TF](
      value: Array[Tensor[T]]
  ): Tensor[T] = {
    Basic.stack(value.toSeq)
  }

  implicit def tensorFromTraversable[T: TF, CC[A] <: TraversableLike[A, CC[A]]](
      value: CC[Tensor[T]]
  ): Tensor[T] = {
    Basic.stack(value.toSeq)
  }
}

private[tensors] trait Priority3Implicits extends Priority2Implicits {
  implicit def tensorFromConvertibleArray[T, V](
      value: Array[V]
  )(implicit
      f: V => Tensor[T],
      evTF: TF[T]
  ): Tensor[T] = {
    Basic.stack(value.toSeq.map(f))
  }

  implicit def tensorFromConvertibleTraversable[T, V, CC[A] <: TraversableLike[A, CC[A]]](
      value: CC[V]
  )(implicit
      f: V => Tensor[T],
      evTF: TF[T]
  ): Tensor[T] = {
    Basic.stack(value.map(f)(breakOut))
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

private[tensors] trait Priority0Implicits extends UntypedPriority1Implicits {
  implicit def tInt2Double[V](value: V)(implicit f: V => Tensor[Int]): Tensor[Double] = f(value).toDouble
}

private[tensors] trait UntypedPriority1Implicits extends UntypedPriority0Implicits {
  implicit def tensorAsUntyped[T](tensor: Tensor[T]): Tensor[Any] = {
    tensor.asInstanceOf[Tensor[Any]]
  }

  implicit def tensorIndexedSlicesAsUntyped[T](tensor: TensorIndexedSlices[T]): TensorIndexedSlices[Any] = {
    tensor.asInstanceOf[TensorIndexedSlices[Any]]
  }

  implicit def sparseTensorAsUntyped[T](tensor: SparseTensor[T]): SparseTensor[Any] = {
    tensor.asInstanceOf[SparseTensor[Any]]
  }

  implicit def tensorLikeAsUntyped[T](tensor: TensorLike[T]): TensorLike[Any] = {
    tensor.asInstanceOf[TensorLike[Any]]
  }

  implicit def tensorArrayAsUntyped(
      tensors: Array[Tensor[_]]
  ): Array[Tensor[Any]] = {
    tensors.asInstanceOf[Array[Tensor[Any]]]
  }

  implicit def tensorIndexedSlicesArrayAsUntyped(
      tensors: Array[TensorIndexedSlices[_]]
  ): Array[TensorIndexedSlices[Any]] = {
    tensors.asInstanceOf[Array[TensorIndexedSlices[Any]]]
  }

  implicit def sparseTensorArrayAsUntyped(
      tensors: Array[SparseTensor[_]]
  ): Array[SparseTensor[Any]] = {
    tensors.asInstanceOf[Array[SparseTensor[Any]]]
  }

  implicit def tensorLikeArrayAsUntyped(
      tensors: Array[TensorLike[_]]
  ): Array[TensorLike[Any]] = {
    tensors.asInstanceOf[Array[TensorLike[Any]]]
  }

  implicit def tensorSeqAsUntyped(
      tensors: Seq[Tensor[_]]
  ): Seq[Tensor[Any]] = {
    tensors.asInstanceOf[Seq[Tensor[Any]]]
  }

  implicit def tensorIndexedSlicesSeqAsUntyped(
      tensors: Seq[TensorIndexedSlices[_]]
  ): Seq[TensorIndexedSlices[Any]] = {
    tensors.asInstanceOf[Seq[TensorIndexedSlices[Any]]]
  }

  implicit def sparseTensorSeqAsUntyped(
      tensors: Seq[SparseTensor[_]]
  ): Seq[SparseTensor[Any]] = {
    tensors.asInstanceOf[Seq[SparseTensor[Any]]]
  }

  implicit def tensorLikeSeqAsUntyped(
      tensors: Seq[TensorLike[_]]
  ): Seq[TensorLike[Any]] = {
    tensors.asInstanceOf[Seq[TensorLike[Any]]]
  }

  implicit def tensorSetAsUntyped(
      tensors: Set[Tensor[_]]
  ): Set[Tensor[Any]] = {
    tensors.asInstanceOf[Set[Tensor[Any]]]
  }

  implicit def tensorIndexedSlicesSetAsUntyped(
      tensors: Set[TensorIndexedSlices[_]]
  ): Set[TensorIndexedSlices[Any]] = {
    tensors.asInstanceOf[Set[TensorIndexedSlices[Any]]]
  }

  implicit def sparseTensorSetAsUntyped(
      tensors: Set[SparseTensor[_]]
  ): Set[SparseTensor[Any]] = {
    tensors.asInstanceOf[Set[SparseTensor[Any]]]
  }

  implicit def tensorLikeSetAsUntyped(
      tensors: Set[TensorLike[_]]
  ): Set[TensorLike[Any]] = {
    tensors.asInstanceOf[Set[TensorLike[Any]]]
  }
}

private[tensors] trait UntypedPriority0Implicits {
  implicit def tensorConvertibleAsUntyped[V, T](
      value: V
  )(implicit f: V => Tensor[T]): Tensor[Any] = {
    f(value).asInstanceOf[Tensor[Any]]
  }

  implicit def tensorIndexedSlicesConvertibleAsUntyped[V, T](
      value: V
  )(implicit f: V => TensorIndexedSlices[T]): TensorIndexedSlices[Any] = {
    f(value).asInstanceOf[TensorIndexedSlices[Any]]
  }

  implicit def sparseTensorConvertibleAsUntyped[V, T](
      value: V
  )(implicit f: V => SparseTensor[T]): SparseTensor[Any] = {
    f(value).asInstanceOf[SparseTensor[Any]]
  }

  implicit def tensorLikeConvertibleAsUntyped[V, T](
      value: V
  )(implicit f: V => TensorLike[T]): TensorLike[Any] = {
    f(value).asInstanceOf[TensorLike[Any]]
  }
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
