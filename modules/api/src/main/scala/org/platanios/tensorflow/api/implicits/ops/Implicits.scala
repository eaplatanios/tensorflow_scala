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

package org.platanios.tensorflow.api.implicits.ops

import org.platanios.tensorflow.api.UntypedOp
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.ops.{Output, OutputLike, TensorArray}
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

/** Groups together all implicits related to tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Implicits
    extends Priority5Implicits
        with ControlFlowImplicits
        with BasicImplicits
        with ClipImplicits
        with EmbeddingImplicits
        with MathImplicits
        with NNImplicits
        with SparseImplicits
        with StatisticsImplicits
        with TextImplicits {
  implicit def opFromOutputLike[T: TF](value: OutputLike[T]): UntypedOp = {
    if (value == null) null else value.op
  }

  implicit def outputFromSupportedType[T: TF](value: T): Output[T] = {
    if (value == null) null else Basic.constant(Tensor.fill[T](Shape())(value))
  }

  implicit def outputFromTensor[T: TF](value: Tensor[T]): Output[T] = {
    if (value == null) null else value.toOutput
  }

  implicit def outputFromOutputLike[T: TF](value: OutputLike[T]): Output[T] = {
    if (value == null) null else value.toOutput
  }
}

private[ops] trait Priority5Implicits extends Priority4Implicits {
  implicit def outputFromTensorArray[T: TF](value: TensorArray[T]): Output[T] = {
    if (value == null) null else value.toOutput
  }

  implicit def outputFromVariable[T: TF](value: Variable[T]): Output[T] = {
    if (value == null) null else value.toOutput
  }
}

private[ops] trait Priority4Implicits extends Priority3Implicits {
  implicit def outputFromArray[T: TF](
      value: Array[Output[T]]
  ): Output[T] = {
    if (value == null) null else Basic.stack(value.toSeq)
  }

  implicit def outputFromSeq[T: TF](
      value: Seq[Output[T]]
  ): Output[T] = {
    if (value == null) null else Basic.stack(value)
  }
}

private[ops] trait Priority3Implicits extends Priority2Implicits {
  implicit def outputFromTensorConvertible[T, TC](value: TC)(implicit
      f: TC => Tensor[T],
      evTF: TF[T]
  ): Output[T] = {
    if (value == null) null else Basic.constant(f(value))
  }

  implicit def outputFromConvertibleArray[T, V](value: Array[V])(implicit
      f: V => Output[T],
      evTF: TF[T]
  ): Output[T] = {
    if (value == null) null else Basic.stack(value.toSeq.map(f))
  }

  implicit def outputFromConvertibleSeq[T, V](value: Seq[V])(implicit
      f: V => Output[T],
      evTF: TF[T]
  ): Output[T] = {
    if (value == null) null else Basic.stack(value.map(f))
  }
}

private[ops] trait Priority2Implicits extends Priority1Implicits {
  implicit def oInt2Long[V](value: V)(implicit f: V => Output[Int]): Output[Long] = f(value).castTo[Long]
  implicit def oLong2Float[V](value: V)(implicit f: V => Output[Long]): Output[Float] = f(value).castTo[Float]
}

private[ops] trait Priority1Implicits extends Priority0Implicits {
  implicit def oInt2Float[V](value: V)(implicit f: V => Output[Int]): Output[Float] = f(value).castTo[Float]
  implicit def oLong2Double[V](value: V)(implicit f: V => Output[Long]): Output[Double] = f(value).castTo[Double]
}

private[ops] trait Priority0Implicits {
  implicit def oInt2Double[V](value: V)(implicit f: V => Output[Int]): Output[Double] = f(value).castTo[Double]
}

// TODO: [TYPES] The following are diasbled for now because they slow compilation down significantly.

//private[ops] trait Priority7Implicits extends Priority6Implicits {
//  implicit def oUByte2Float[V](value: V)(implicit f: V => Output[UByte]): Output[Float] = f(value).castTo[Float]
//}
//
//private[ops] trait Priority6Implicits extends Priority5Implicits {
//  implicit def oUByte2Double[V](value: V)(implicit f: V => Output[UByte]): Output[Double] = f(value).castTo[Double]
//}
//
//private[ops] trait Priority5Implicits extends Priority4Implicits {
//  implicit def oUByte2Short[V](value: V)(implicit f: V => Output[UByte]): Output[Short] = f(value).castTo[Short]
//  implicit def oUShort2Float[V](value: V)(implicit f: V => Output[UShort]): Output[Float] = f(value).castTo[Float]
//}
//
//private[ops] trait Priority4Implicits extends Priority3Implicits {
//  implicit def oByte2Float[V](value: V)(implicit f: V => Output[Byte]): Output[Float] = f(value).castTo[Float]
//  implicit def oUByte2Int[V](value: V)(implicit f: V => Output[UByte]): Output[Int] = f(value).castTo[Int]
//  implicit def oUShort2Double[V](value: V)(implicit f: V => Output[UShort]): Output[Double] = f(value).castTo[Double]
//}
//
//private[ops] trait Priority3Implicits extends Priority2Implicits {
//  implicit def oByte2Double[V](value: V)(implicit f: V => Output[Byte]): Output[Double] = f(value).castTo[Double]
//  implicit def oShort2Float[V](value: V)(implicit f: V => Output[Short]): Output[Float] = f(value).castTo[Float]
//  implicit def oUByte2Long[V](value: V)(implicit f: V => Output[UByte]): Output[Long] = f(value).castTo[Long]
//  implicit def oUShort2Int[V](value: V)(implicit f: V => Output[UShort]): Output[Int] = f(value).castTo[Int]
//  implicit def oUInt2Float[V](value: V)(implicit f: V => Output[UInt]): Output[Float] = f(value).castTo[Float]
//}
//
//private[ops] trait Priority2Implicits extends Priority1Implicits {
//  implicit def oByte2Short[V](value: V)(implicit f: V => Output[Byte]): Output[Short] = f(value).castTo[Short]
//  implicit def oShort2Double[V](value: V)(implicit f: V => Output[Short]): Output[Double] = f(value).castTo[Double]
//  implicit def oInt2Float[V](value: V)(implicit f: V => Output[Int]): Output[Float] = f(value).castTo[Float]
//  implicit def oUByte2UShort[V](value: V)(implicit f: V => Output[UByte]): Output[UShort] = f(value).castTo[UShort]
//  implicit def oUShort2Long[V](value: V)(implicit f: V => Output[UShort]): Output[Long] = f(value).castTo[Long]
//  implicit def oUInt2Double[V](value: V)(implicit f: V => Output[UInt]): Output[Double] = f(value).castTo[Double]
//}
//
//private[ops] trait Priority1Implicits extends Priority0Implicits {
//  implicit def oByte2Int[V](value: V)(implicit f: V => Output[Byte]): Output[Int] = f(value).castTo[Int]
//  implicit def oShort2Int[V](value: V)(implicit f: V => Output[Short]): Output[Int] = f(value).castTo[Int]
//  implicit def oInt2Double[V](value: V)(implicit f: V => Output[Int]): Output[Double] = f(value).castTo[Double]
//  implicit def oLong2Float[V](value: V)(implicit f: V => Output[Long]): Output[Float] = f(value).castTo[Float]
//  implicit def oUByte2UInt[V](value: V)(implicit f: V => Output[UByte]): Output[UInt] = f(value).castTo[UInt]
//  implicit def oUShort2UInt[V](value: V)(implicit f: V => Output[UShort]): Output[UInt] = f(value).castTo[UInt]
//  implicit def oUInt2Long[V](value: V)(implicit f: V => Output[UInt]): Output[Long] = f(value).castTo[Long]
//  implicit def oULong2Float[V](value: V)(implicit f: V => Output[ULong]): Output[Float] = f(value).castTo[Float]
//}
//
//private[ops] trait Priority0Implicits {
//  implicit def oByte2Long[V](value: V)(implicit f: V => Output[Byte]): Output[Long] = f(value).castTo[Long]
//  implicit def oShort2Long[V](value: V)(implicit f: V => Output[Short]): Output[Long] = f(value).castTo[Long]
//  implicit def oInt2Long[V](value: V)(implicit f: V => Output[Int]): Output[Long] = f(value).castTo[Long]
//  implicit def oLong2Double[V](value: V)(implicit f: V => Output[Long]): Output[Double] = f(value).castTo[Double]
//  implicit def oUByte2ULong[V](value: V)(implicit f: V => Output[UByte]): Output[ULong] = f(value).castTo[ULong]
//  implicit def oUShort2ULong[V](value: V)(implicit f: V => Output[UShort]): Output[ULong] = f(value).castTo[ULong]
//  implicit def oUInt2ULong[V](value: V)(implicit f: V => Output[UInt]): Output[ULong] = f(value).castTo[ULong]
//  implicit def oULong2Double[V](value: V)(implicit f: V => Output[ULong]): Output[Double] = f(value).castTo[Double]
//}
