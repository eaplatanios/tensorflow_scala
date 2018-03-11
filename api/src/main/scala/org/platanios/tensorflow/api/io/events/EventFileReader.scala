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

package org.platanios.tensorflow.api.io.events

import org.platanios.tensorflow.api.core.exception.{DataLossException, OutOfRangeException}
import org.platanios.tensorflow.api.io.{CompressionType, Loader, NoCompression}
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{RecordReader => NativeReader}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.util.Event

import java.nio.file.Path

/** Event file reader.
  *
  * An event file reader is used to create iterators over the events stored in the file at the provided path (i.e.,
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
class EventFileReader(val filePath: Path, val compressionType: CompressionType = NoCompression)
    extends Closeable with Loader[Event] {
  EventFileReader.logger.info(s"Opening a TensorFlow events file located at '${filePath.toAbsolutePath}'.")

  private[this] var nativeHandle: Long = {
    NativeReader.newRecordReaderWrapper(filePath.toAbsolutePath.toString, compressionType.name, 0)
  }

  private[this] object NativeHandleLock

  // Keep track of references in the Scala side and notify the native library when the reader is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  def load(): Iterator[Event] = new Iterator[Event] {
    /** Caches the next event stored in the file. */
    private[this] var nextEvent: Event = _

    /** Reads the next event stored in the file. */
    private[this] def readNext(): Event = {
      try {
        Event.parseFrom(NativeReader.recordReaderWrapperReadNext(nativeHandle))
      } catch {
        case _: OutOfRangeException | _: DataLossException =>
          // We ignore partial read exceptions, because a record may be truncated. The record readers holds the offset
          // prior to the failed read, and so retrying will succeed.
          EventFileReader.logger.info(s"No more events stored at '${filePath.toAbsolutePath}'.")
          null
      }
    }

    override def hasNext: Boolean = {
      if (nextEvent == null)
        nextEvent = readNext()
      nextEvent != null
    }

    override def next(): Event = {
      val event = {
        if (nextEvent == null)
          readNext()
        else
          nextEvent
      }
      if (event != null)
        nextEvent = readNext()
      event
    }
  }

  /** Closes this reader and releases any resources associated with it. Note that an events file reader is not usable
    * after it has been closed. */
  override def close(): Unit = {
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        NativeReader.deleteRecordReaderWrapper(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

private[io] object EventFileReader {
  private[EventFileReader] val logger: Logger = Logger(LoggerFactory.getLogger("Event File Reader"))

  /** Creates a new events file reader.
    *
    * @param  filePath        Path to the file being read.
    * @param  compressionType Compression type used for the file.
    * @return Newly constructed events file reader.
    */
  def apply(filePath: Path, compressionType: CompressionType = NoCompression): EventFileReader = {
    new EventFileReader(filePath, compressionType)
  }
}
