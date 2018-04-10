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

/**
  * @author Emmanouil Antonios Platanios
  */
object RecordReader {
  TensorFlow.load()

  @native def newRandomAccessFile(filename: String): Long
  @native def deleteRandomAccessFile(fileHandle: Long): Unit

  @native def newRecordReader(fileHandle: Long, compressionType: String): Long
  @native def recordReaderRead(readerHandle: Long, offset: Long): Array[Byte]
  @native def deleteRecordReader(readerHandle: Long): Unit

  @native def newSequentialRecordReader(fileHandle: Long, compressionType: String): Long
  @native def sequentialRecordReaderReadNext(readerHandle: Long): Array[Byte]
  @native def deleteSequentialRecordReader(readerHandle: Long): Unit

  @native def newRecordReaderWrapper(filename: String, compressionType: String, startOffset: Long): Long
  @native def recordReaderWrapperReadNext(readerHandle: Long): Array[Byte]
  @native def recordReaderWrapperOffset(readerHandle: Long): Long
  @native def deleteRecordReaderWrapper(readerHandle: Long): Unit
}
