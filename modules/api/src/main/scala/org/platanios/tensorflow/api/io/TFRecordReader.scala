/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.exception.{DataLossException, OutOfRangeException}
import org.platanios.tensorflow.api.utilities.{CRC32C, Coding}
import org.platanios.tensorflow.proto.Example

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import java.io.BufferedInputStream
import java.nio.file.{Files, Path, StandardOpenOption}

/** TensorFlow record file reader.
  *
  * An TF record reader is used to create iterators over the examples stored in the file at the provided path (i.e.,
  * `filePath`).
  *
  * Note that this reader ignores any corrupted records at the end of the file. That is to allow for "live tracking" of
  * summary files while they're being written to.
  *
  * @param  filePath        Path to the file being read.
  * @param  compressionType Compression type used for the file.
  *
  * @author Emmanouil Antonios Platanios
  */
case class TFRecordReader(filePath: Path, compressionType: CompressionType = NoCompression) extends Loader[Example] {
  protected val fileStream: BufferedInputStream = {
    new BufferedInputStream(Files.newInputStream(filePath, StandardOpenOption.READ))
  }

  // TODO: !!! This has weird semantics given that the tests currently expect the file stream to be reused.
  def load(): Iterator[Example] = {
    TFRecordReader.logger.info(s"Opening a TensorFlow records file located at '${filePath.toAbsolutePath}'.")

    new Iterator[Example] {
      /** Caches the next example stored in the file. */
      private[this] var nextExample: Example = _

      /** Reads the next example stored in the file. */
      private[this] def readNext(): Example = {
        try {
          // Read the header data.
          val encLength = new Array[Byte](12)
          fileStream.read(encLength)
          val recordLength = Coding.decodeFixedInt64(encLength).toInt
          val encLengthMaskedCrc = CRC32C.mask(CRC32C.value(encLength.take(8)))
          if (Coding.decodeFixedInt32(encLength, offset = 8) != encLengthMaskedCrc) {
            throw DataLossException("Encountered corrupted TensorFlow record.")
          }

          // Read the data.
          val encData = new Array[Byte](recordLength + 4)
          fileStream.read(encData)
          val recordData = encData.take(recordLength)
          val encDataMaskedCrc = CRC32C.mask(CRC32C.value(encData.take(recordLength)))
          if (Coding.decodeFixedInt32(encData, offset = recordLength) != encDataMaskedCrc) {
            throw DataLossException("Encountered corrupted TensorFlow record.")
          }

          Example.parseFrom(recordData)
        } catch {
          case _: OutOfRangeException | _: DataLossException =>
            // We ignore partial read exceptions, because a record may be truncated. The record reader holds the offset
            // prior to the failed read, and so retrying will succeed.
            TFRecordReader.logger.info(s"No more TF records stored at '${filePath.toAbsolutePath}'.")
            null
        }
      }

      override def hasNext: Boolean = {
        if (nextExample == null)
          nextExample = readNext()
        nextExample != null
      }

      override def next(): Example = {
        val example = {
          if (nextExample == null)
            readNext()
          else
            nextExample
        }
        if (example != null)
          nextExample = readNext()
        example
      }
    }
  }
}

private[io] object TFRecordReader {
  private[TFRecordReader] val logger: Logger = Logger(LoggerFactory.getLogger("TF Record Reader"))
}
