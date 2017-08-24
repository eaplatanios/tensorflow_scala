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

  @native def eagerAllocateOp(contextHandle: Long, opType: String): Long
  @native def eagerDeleteOp(opHandle: Long): Unit
  @native def eagerSetOpDevice(opHandle: Long, contextHandle: Long, device: String): Unit
  @native def eagerOpAddInput(opHandle: Long, eagerTensorHandle: Long): Unit

  // The names of all the setAttr* family functions below correspond to the C library types, not the
  // Java library types. Roughly, setAttrFoo calls the TensorFlow C library function: TFE_OpSetAttrFoo.
  @native def eagerSetOpAttrString(opHandle: Long, name: String, value: Array[Byte]): Unit
  @native def eagerSetOpAttrStringList(opHandle: Long, name: String, value: Array[Array[Byte]]): Unit
  @native def eagerSetOpAttrInt(opHandle: Long, name: String, value: Long): Unit
  @native def eagerSetOpAttrIntList(opHandle: Long, name: String, value: Array[Long]): Unit
  @native def eagerSetOpAttrFloat(opHandle: Long, name: String, value: Float): Unit
  @native def eagerSetOpAttrFloatList(opHandle: Long, name: String, value: Array[Float]): Unit
  @native def eagerSetOpAttrBool(opHandle: Long, name: String, value: Boolean): Unit
  @native def eagerSetOpAttrBoolList(opHandle: Long, name: String, value: Array[Boolean]): Unit
  @native def eagerSetOpAttrType(opHandle: Long, name: String, dataType: Int): Unit
  @native def eagerSetOpAttrTypeList(opHandle: Long, name: String, dataType: Array[Int]): Unit
  @native def eagerSetOpAttrShape(opHandle: Long, name: String, shape: Array[Long], numDims: Int): Unit
  @native def eagerSetOpAttrShapeList(
      opHandle: Long, name: String, shapes: Array[Array[Long]], numDims: Array[Int], numShapes: Int): Unit

  @native def eagerExecuteOp(opHandle: Long): Array[Long]

  //endregion Eager Execution API

  //region Ops

  @native def cast(contextHandle: Long, tensorHandle: Long, dataType: Int): Long
  @native def pack(contextHandle: Long, tensorHandles: Array[Long], axis: Long): Long
  @native def stridedSlice(
      contextHandle: Long, tensorHandle: Long, beginTensorHandle: Long, endTensorHandle: Long,
      stridesTensorHandle: Long, beginMask: Long, endMask: Long, ellipsisMask: Long, newAxisMask: Long,
      shrinkAxisMask: Long): Long
  @native def reshape(contextHandle: Long, tensorHandle: Long, shapeTensorHandle: Long): Long
  @native def add(contextHandle: Long, tensor1Handle: Long, tensor2Handle: Long): Long

  //endregion Ops
}
