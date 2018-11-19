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

package org.platanios.tensorflow.api.io

import java.io.BufferedOutputStream
import java.nio.file.{Files, Path, StandardOpenOption}

import org.platanios.tensorflow.api.utilities.{CRC32C, Coding}
import org.tensorflow.example.Example

/** Helper used to write `Example` protocol buffers to TensorFlow record files.
  *
  * @param  filePath TensorFlow record file path.
  *
  * @author Emmanouil Antonios Platanios
  */
class TFRecordWriter(val filePath: Path) {
  protected var fileStream: BufferedOutputStream = {
    new BufferedOutputStream(Files.newOutputStream(
      filePath, StandardOpenOption.CREATE_NEW, StandardOpenOption.APPEND))
  }

  /** Appends `example` to the TensorFlow records file. */
  def write(example: Example): Unit = {
    val recordBytes = example.toByteArray
    // Format of a single record:
    //  uint64    length
    //  uint32    masked crc of length
    //  byte      data[length]
    //  uint32    masked crc of data
    val encLength = Coding.encodeFixedInt64(recordBytes.length)
    val encLengthMaskedCrc = Coding.encodeFixedInt32(CRC32C.mask(CRC32C.value(encLength)))
    val encDataMaskedCrc = Coding.encodeFixedInt32(CRC32C.mask(CRC32C.value(recordBytes)))
    fileStream.write(encLength ++ encLengthMaskedCrc ++ recordBytes ++ encDataMaskedCrc)
  }

  /** Pushes outstanding examples to disk. */
  def flush(): Unit = {
    fileStream.flush()
  }

  /** Calls `flush()` and then closes the current TensorFlow records file. */
  def close(): Unit = {
    fileStream.close()
  }
}

object TFRecordWriter {
  def apply(filePath: Path): TFRecordWriter = {
    new TFRecordWriter(filePath)
  }
}
