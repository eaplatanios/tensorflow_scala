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

import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.tensors._
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
    * @param  input Tensor to cast.
    * @tparam R Target data type.
    * @return Result as a new tensor.
    */
  private[tensors] def cast[T, R: TF, TL[TT] <: TensorLike[TT]](
      input: TL[T],
      truncate: Boolean = false
  )(implicit ev: TensorOps.Aux[TL, T]): TL[R] = {
    val dataType = implicitly[TF[R]].dataType
    if (input.dataType == dataType) {
      input.asInstanceOf[TL[R]]
    } else {
      ev.applyUnary(input, t => {
        Tensor.fromNativeHandle[R](NativeTensorOpsMath.cast(
          executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue, truncate))
      })
    }
  }

  // TODO: [OPS] saturateCast

  /** $OpDocCastBitcast
    *
    * @group CastOps
    *
    * @param  input Input tensor.
    * @tparam R Target data type.
    * @return Result as a new tensor.
    */
  private[tensors] def bitcast[T: IsNumeric, R: TF, TL[TT] <: TensorLike[TT]](
      input: TL[T]
  )(implicit ev: TensorOps.Aux[TL, T]): TL[R] = {
    val dataType = implicitly[TF[R]].dataType
    ev.applyUnary(input, t => {
      Tensor.fromNativeHandle[R](NativeTensorOpsMath.bitcast(
        executionContext.value.nativeHandle, t.nativeHandle, dataType.cValue))
    })
  }
}

object Cast extends Cast
