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

import org.platanios.tensorflow.api.core._
import org.platanios.tensorflow.api.core.exception.InvalidIndexerException
import org.platanios.tensorflow.api.tensors.{Context, Tensor, TensorConvertible}
import org.platanios.tensorflow.jni.generated.tensors.{Basic => NativeTensorOpsBasic}

import scala.util.DynamicVariable

/** Contains functions for executing ops related to basic tensor manipulation.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Basic {
  def stack(inputs: Seq[Tensor], axis: Int = 0)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor(NativeTensorOpsBasic.pack(context.value.nativeHandle, inputs.map(_.nativeHandle).toArray, axis))
  }

  def reshape(input: Tensor, shape: Tensor)(implicit context: DynamicVariable[Context]): Tensor = {
    Tensor(NativeTensorOpsBasic.reshape(context.value.nativeHandle, input.nativeHandle, shape.nativeHandle))
  }

  private[api] def stridedSlice(
      input: Tensor, begin: Tensor, end: Tensor, strides: Tensor = null, beginMask: Long = 0, endMask: Long = 0,
      ellipsisMask: Long = 0, newAxisMask: Long = 0, shrinkAxisMask: Long = 0)(
      implicit context: DynamicVariable[Context]): Tensor = {
    Tensor(NativeTensorOpsBasic.stridedSlice(
      context.value.nativeHandle, input.nativeHandle, begin.nativeHandle, end.nativeHandle, strides.nativeHandle,
      beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask))
  }
}

private[api] object Basic extends Basic {
  private[ops] trait Implicits {
    implicit def tensorToBasicTensorOps(tensor: Tensor): TensorOps = TensorOps(tensor)
  }

  case class TensorOps private[ops](tensor: Tensor) {
    def reshape[T: TensorConvertible](shape: T): Tensor = Basic.reshape(tensor, shape)

    def slice(indexers: Indexer*): Tensor = {
      if (indexers.count(_ == Ellipsis) > 1)
        throw InvalidIndexerException("Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
      val begin = Array.fill(indexers.length)(0)
      val end = Array.fill(indexers.length)(0)
      val strides = Array.fill(indexers.length)(1)
      var beginMask: Long = 0 // TODO: Use this.
      var endMask: Long = 0
      var ellipsisMask: Long = 0
      var newAxisMask: Long = 0
      var shrinkAxisMask: Long = 0
      indexers.zipWithIndex foreach {
        case (Ellipsis, i) => ellipsisMask |= (1 << i)
        case (NewAxis, i) => newAxisMask |= (1 << i)
        case (Index(index), i) =>
          begin(i) = index
          end(i) = index + 1
          strides(i) = 1
          shrinkAxisMask |= (1 << i)
        case (Slice(sliceBegin, sliceEnd, sliceStep, false), i) =>
          begin(i) = sliceBegin
          end(i) = sliceEnd
          strides(i) = sliceStep
        case (Slice(sliceBegin, sliceEnd, sliceStep, true), i) =>
          begin(i) = sliceBegin
          if (sliceEnd == -1) {
            end(i) = sliceEnd
            endMask |= (1 << i)
          } else {
            end(i) = sliceEnd + 1
          }
          strides(i) = sliceStep
      }
      val beginTensor: Tensor = begin
      val endTensor: Tensor = end
      val stridesTensor: Tensor = strides
      val result = Basic.stridedSlice(
        tensor, beginTensor, endTensor, stridesTensor, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask)
      beginTensor.close()
      endTensor.close()
      stridesTensor.close()
      result
    }
  }
}
