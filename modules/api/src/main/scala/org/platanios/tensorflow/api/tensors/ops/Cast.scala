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

package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

/** Contains functions for executing cast-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Cast {
  /** $OpDocCastCast
    *
    * @group CastOps
    *
    * @param  x        Tensor to cast.
    * @param  dataType Target data type.
    * @return Result as a new tensor.
    */
  def cast[T, R, TL[TT] <: TensorLike[TT]](
      x: TL[T],
      dataType: DataType[R],
      truncate: Boolean = false
  )(implicit ev: TensorOps.Aux[TL, T]): TL[R] = {
    if (x.dataType == dataType) {
      x.asInstanceOf[TL[R]]
    } else {
      ev.applyUnary(x, t => {
        Tensor.fromNativeHandle(NativeTensorOpsMath.cast(
          executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue, truncate))
      })
    }
  }

  // TODO: [OPS] saturateCast

  /** $OpDocCastBitcast
    *
    * @group CastOps
    *
    * @param  input    Input tensor.
    * @param  dataType Target data type.
    * @return Result as a new tensor.
    */
  def bitcast[T: IsNumeric, R](
      input: Tensor[T],
      dataType: DataType[R]
  ): Tensor[R] = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.bitcast(
      executionContext.value.nativeHandle, input.nativeHandle, dataType.cValue))
  }
}

object Cast extends Cast {
  private[tensors] trait Implicits {
    implicit class CastOps[T](val tensor: Tensor[T]) {
      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R: SupportedType]: Tensor[R] = {
        Cast.cast(tensor, implicitly[SupportedType[R]].dataType)
      }

      /** $OpDocCastCast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def castTo[R](dataType: DataType[R]): Tensor[R] = {
        Cast.cast(tensor, dataType)
      }
    }

    implicit class NumericCastOps[T: IsNumeric](val tensor: Tensor[T]) {
      /** $OpDocCastBitcast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R: SupportedType]: Tensor[R] = {
        Cast.bitcast(tensor, implicitly[SupportedType[R]].dataType)
      }

      /** $OpDocCastBitcast
        *
        * @group CastOps
        * @tparam R Target data type.
        * @return Result as a new tensor.
        */
      def bitcastTo[R](dataType: DataType[R]): Tensor[R] = {
        Cast.bitcast(tensor, dataType)
      }
    }

    implicit def tensorConvertibleToCastOps[T, TC](
        value: TC
    )(implicit f: TC => Tensor[T]): CastOps[T] = {
      new CastOps(f(value))
    }

    implicit def tensorConvertibleToNumericCastOps[T: IsNumeric, TC](
        value: TC
    )(implicit f: TC => Tensor[T]): NumericCastOps[T] = {
      new NumericCastOps(f(value))
    }
  }
}
