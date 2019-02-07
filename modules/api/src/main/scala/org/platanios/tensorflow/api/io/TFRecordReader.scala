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
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{RecordReader => NativeReader}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.example.Example

import java.nio.file.Path

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
class TFRecordReader protected (
    val filePath: Path,
    val compressionType: CompressionType = NoCompression,
    private[this] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
) extends Closeable with Loader[Example] {
  /** Lock for the native handle. */
  private[TFRecordReader] def NativeHandleLock = nativeHandleWrapper.Lock

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = nativeHandleWrapper.handle

  def load(): Iterator[Example] = new Iterator[Example] {
    /** Caches the next example stored in the file. */
    private[this] var nextExample: Example = _

    /** Reads the next example stored in the file. */
    private[this] def readNext(): Example = {
      try {
        Example.parseFrom(NativeReader.recordReaderWrapperReadNext(nativeHandle))
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

private[io] object TFRecordReader {
  private[TFRecordReader] val logger: Logger = Logger(LoggerFactory.getLogger("TF Record Reader"))

  /** Creates a new events file reader.
    *
    * @param  filePath        Path to the file being read.
    * @param  compressionType Compression type used for the file.
    * @return Newly constructed events file reader.
    */
  def apply(filePath: Path, compressionType: CompressionType = NoCompression): TFRecordReader = {
    TFRecordReader.logger.info(s"Opening a TensorFlow records file located at '${filePath.toAbsolutePath}'.")
    val nativeHandle = NativeReader.newRecordReaderWrapper(filePath.toAbsolutePath.toString, compressionType.name, 0)
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val closeFn = () => {
      nativeHandleWrapper.Lock.synchronized {
        if (nativeHandleWrapper.handle != 0) {
          NativeReader.deleteRecordReaderWrapper(nativeHandleWrapper.handle)
          nativeHandleWrapper.handle = 0
        }
      }
    }
    val eventFileReader = new TFRecordReader(filePath, compressionType, nativeHandleWrapper, closeFn)
    // Keep track of references in the Scala side and notify the native library when the TF record file reader is not
    // referenced anymore anywhere in the Scala side. This will let the native library free the allocated resources and
    // prevent a potential memory leak.
    Disposer.add(eventFileReader, closeFn)
    eventFileReader
  }
}
