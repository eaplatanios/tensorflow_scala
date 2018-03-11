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

package org.platanios.tensorflow.jni

import java.nio.ByteBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
object Tensor {
  TensorFlow.load()

  @native def allocate(dataType: Int, shape: Array[Long], numBytes: Long): Long
  @native def fromBuffer(dataType: Int, shape: Array[Long], numBytes: Long, buffer: ByteBuffer): Long
  // @native def fromBuffer(buffer: ByteBuffer, dataType: Int, shape: Array[Long], byteSize: Long): Long
  @native def dataType(handle: Long): Int
  @native def shape(handle: Long): Array[Long]
  @native def buffer(handle: Long): ByteBuffer
  @native def delete(handle: Long): Unit
  @native def getEncodedStringSize(numStringBytes: Int): Int
  @native def setStringBytes(stringBytes: Array[Byte], buffer: ByteBuffer): Int
  @native def getStringBytes(buffer: ByteBuffer): Array[Byte]

  //region Eager Execution API

  // TODO: [SESSION] Add support for session options.
  @native def eagerAllocateContext(): Long
  @native def eagerDeleteContext(handle: Long): Unit
  // TODO: [SESSION] "listDevices".

  @native def eagerAllocate(tensorHandle: Long): Long
  @native def eagerDataType(handle: Long): Int
  @native def eagerShape(handle: Long): Array[Long]
  @native def eagerDevice(handle: Long): String
  @native def eagerDelete(handle: Long): Unit
  @native def eagerResolve(handle: Long): Long
  @native def eagerCopyToDevice(handle: Long, contextHandle: Long, device: String): Long
  @native def eagerSetOpDevice(opHandle: Long, device: String): Unit

  //endregion Eager Execution API
}
