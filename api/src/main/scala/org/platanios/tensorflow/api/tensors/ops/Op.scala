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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.tensors.DeviceTensor
import org.platanios.tensorflow.api.tensors.eager.Context
import org.platanios.tensorflow.api.types.DataType
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
private[tensors] final case class Op(opType: String)(implicit context: DynamicVariable[Context]) extends Closeable {
  private[this] object NativeHandleLock
  private[this] var nativeHandle: Long = NativeTensor.eagerAllocateOp(context.value.nativeHandle, opType)

  // Keep track of references in the Scala side and notify the native library when the op is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  def addInput(input: DeviceTensor): Op = {
    NativeTensor.eagerOpAddInput(nativeHandle, input.nativeHandle)
    this
  }

  def setAttribute(name: String, value: Long): Op = {
    NativeTensor.eagerSetOpAttrInt(nativeHandle, name, value)
    this
  }

  def setAttribute(name: String, value: DataType): Op = {
    NativeTensor.eagerSetOpAttrType(nativeHandle, name, value.cValue)
    this
  }

  def setAttribute(name: String, value: Shape): Op = {
    NativeTensor.eagerSetOpAttrShape(nativeHandle, name, value.asArray.map(_.toLong), value.rank)
    this
  }

  def execute(): Seq[DeviceTensor] = context.value.execute(nativeHandle)

  /** Closes this [[Op]] and releases any resources associated with it. Note that an [[Op]] is not usable after it has
    * been closed. */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        NativeTensor.eagerDelete(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}
