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
import org.platanios.tensorflow.api.types._
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

  def range(
      start: Tensor, limit: Tensor, delta: Tensor = 1, dataType: DataType = null)(
      implicit context: DynamicVariable[Context]): Tensor = {
    require(start.rank == 0, s"'start' (rank = ${start.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(limit.rank == 0, s"'limit' (rank = ${limit.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    require(delta.rank == 0, s"'delta' (rank = ${delta.rank}) must have rank 0 (i.e., must be a scalar tensor).")
    var castedStart: Tensor = start
    var castedLimit: Tensor = limit
    var castedDelta: Tensor = delta
    val supportedDataTypes = Set[DataType](FLOAT32, FLOAT64, INT32, INT64)
    require(supportedDataTypes.contains(start.dataType), s"Unsupported data type '${start.dataType}'.")
    require(supportedDataTypes.contains(limit.dataType), s"Unsupported data type '${limit.dataType}'.")
    require(supportedDataTypes.contains(delta.dataType), s"Unsupported data type '${delta.dataType}'.")
    val inferredDataType = {
      if (dataType != null)
        dataType
      else
        DataType.mostPrecise(start.dataType, limit.dataType, delta.dataType)
    }
    if (start.dataType != inferredDataType)
      castedStart = cast(start, inferredDataType)
    if (limit.dataType != inferredDataType)
      castedLimit = cast(limit, inferredDataType)
    if (delta.dataType != inferredDataType)
      castedDelta = cast(delta, inferredDataType)
    Tensor.fromNativeHandle(
      NativeTensorOpsMath.range(
        context.value.nativeHandle, castedStart.nativeHandle, castedLimit.nativeHandle, castedDelta.nativeHandle))
  }

  def add(x: Tensor, y: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.add(context.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  def sub(x: Tensor, y: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.sub(context.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  def mod(x: Tensor, y: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.mod(context.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  def less(x: Tensor, y: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.less(context.value.nativeHandle, x.nativeHandle, y.nativeHandle))
  }

  def prod(input: Tensor, axes: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(
      NativeTensorOpsMath.prod(context.value.nativeHandle, input.nativeHandle, axes.nativeHandle, false))
  }

  def max(input: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor.fromNativeHandle(NativeTensorOpsMath.max(context.value.nativeHandle, input.nativeHandle, 0, false))
  }
}

private[api] object Math extends Math {
  private[ops] trait Implicits {
    implicit def tensorToMathTensorOps(tensor: Tensor): TensorOps = TensorOps(tensor)
  }

  case class TensorOps private[ops](tensor: Tensor) {
    def +[T: TensorConvertible](other: T): Tensor = add(other)
    def -[T: TensorConvertible](other: T): Tensor = subtract(other)
    def %[T: TensorConvertible](other: T): Tensor = mod(other)

    def cast(dataType: DataType): Tensor = Math.cast(tensor, dataType)

    private[this] def binaryOperatorHelper(other: Tensor, operator: (Tensor, Tensor) => Tensor): Tensor = {
      if (tensor.dataType.priority >= other.dataType.priority)
        operator(tensor, Math.cast(other, tensor.dataType))
      else
        operator(Math.cast(tensor, other.dataType), other)
    }

    def add[T: TensorConvertible](other: T): Tensor = binaryOperatorHelper(other, Math.add)

    def subtract[T: TensorConvertible](other: T): Tensor = binaryOperatorHelper(other, Math.sub)

    def mod[T: TensorConvertible](other: T): Tensor = binaryOperatorHelper(other, Math.mod)
  }
}
