// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.jni

import java.nio.ByteBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
object Tensor {
  TensorFlow.load()

  @native def fromBuffer(buffer: ByteBuffer, dataType: Int, shape: Array[Long], byteSize: Long): Long
  @native def dataType(handle: Long): Int
  @native def shape(handle: Long): Array[Long]
  @native def buffer(handle: Long): ByteBuffer
  @native def delete(handle: Long): Unit
  @native def getEncodedStringSize(numStringBytes: Int): Int
  @native def setStringBytes(stringBytes: Array[Byte], buffer: ByteBuffer): Int
  @native def getStringBytes(buffer: ByteBuffer): Array[Byte]
}
