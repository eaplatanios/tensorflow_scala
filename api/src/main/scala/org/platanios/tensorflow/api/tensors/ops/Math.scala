/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.tensors.{Context, Tensor, TensorConvertible}
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.jni.generated.tensors.{Math => NativeTensorOpsMath}

import scala.util.DynamicVariable

/** Contains functions for executing general math-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Math {
  def cast(x: Tensor, dataType: DataType)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.cast(context.value.nativeHandle, x.nativeHandle, dataType.cValue))
  }

  def add(x: Tensor, y: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.add(context.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }
}

private[api] object Math extends Math {
  private[ops] trait Implicits {
    implicit def tensorToMathTensorOps(tensor: Tensor): TensorOps = TensorOps(tensor)
  }

  case class TensorOps private[ops](tensor: Tensor) {
    def +[T: TensorConvertible](other: T): Tensor = add(other)

    def cast(dataType: DataType): Tensor = Math.cast(tensor, dataType)

    private[this] def binaryOperatorHelper(other: Tensor, operator: (Tensor, Tensor) => Tensor): Tensor = {
      if (tensor.dataType.priority >= other.dataType.priority)
        operator(tensor, Math.cast(other, tensor.dataType))
      else
        operator(Math.cast(tensor, other.dataType), other)
    }

    def add[T: TensorConvertible](other: T): Tensor = binaryOperatorHelper(other, Math.add)
  }
}
