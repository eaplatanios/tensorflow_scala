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

package org.platanios.tensorflow.api.io.events

import org.platanios.tensorflow.api.utilities.{CRC32C, Coding}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory
import org.tensorflow.util.Event

import java.io.BufferedOutputStream
import java.nio.file.{Files, Path, StandardOpenOption}
import java.util.concurrent.{BlockingQueue, LinkedBlockingDeque}

/** Writes `Event` protocol buffers to files.
  *
  * The `EventFileWriter` class creates an event file in the specified directory and asynchronously writes `Event`
  * protocol buffers to it. The file is encoded using the `TFRecord` format, which is similar to `RecordIO`.
  *
  * On construction the event file writer creates a new event file in `workingDir`. This event file will contain `Event`
  * protocol buffers, which are written to disk via the `EventFileWriter.write()` method.
  *
  * @param  workingDir     Directory in which to write the event file.
  * @param  queueCapacity  Maximum number of events pending to be written to disk before a call to `write()` blocks.
  * @param  flushFrequency Specifies how often to flush the written events to disk (in seconds).
  * @param  filenameSuffix Filename suffix to use for the event file.
  *
  * @author Emmanouil Antonios Platanios
  */
class EventFileWriter private[io](
    val workingDir: Path,
    val queueCapacity: Int = 10,
    val flushFrequency: Int = 10,
    val filenameSuffix: String = "") {
  if (!Files.isDirectory(workingDir))
    Files.createDirectories(workingDir)

  private[this] var _closed: Boolean = false

  private[this] val queue            : BlockingQueue[Event] = new LinkedBlockingDeque[Event](queueCapacity)
  private[this] val sentinelEvent    : Event                = Event.newBuilder().build()
  private[this] var eventWriter      : EventWriter          = new EventWriter(workingDir, "events", filenameSuffix)
  private[this] var eventWriterThread: Thread               = newEventWriterThread()

  eventWriterThread.start()

  /** Writes the provided event to the event file. */
  def write(event: Event): Unit = {
    if (!_closed)
      queue.put(event)
  }

  /** Pushes outstanding events to disk. */
  def flush(): Unit = queue synchronized {
    while (!queue.isEmpty)
      queue.wait()
    eventWriter.flush()
  }

  /** Calls `flush()` and then closes the current event file. */
  def close(): Unit = {
    write(sentinelEvent)
    flush()
    eventWriterThread.join()
    eventWriter.close()
    _closed = true
  }

  /** Returns `true` if this event file writer has been closed. */
  def closed: Boolean = _closed

  /** Reopens this event file writer. */
  def reopen(): Unit = {
    if (_closed) {
      eventWriter = new EventWriter(workingDir, "events", filenameSuffix)
      eventWriterThread = newEventWriterThread()
      eventWriterThread.start()
      _closed = false
    }
  }

  /** Creates a thread that pulls events from the queue and writes them to the event file. */
  private[this] def newEventWriterThread(): Thread = {
    val thread = new Thread(new Runnable {
      // The first event will be flushed immediately.
      private[this] var nextEventFlushTime: Int = 0

      override def run(): Unit = {
        var sentinelReceived = false
        while (!sentinelReceived) {
          val event = queue.take()
          if (event == sentinelEvent)
            sentinelReceived = true
          else
            eventWriter.write(event)
          queue synchronized {
            if (queue.isEmpty)
              queue.notify()
          }
          // Flush the event writer every so often.
          val currentTime = System.currentTimeMillis() / 1000
          if (currentTime > nextEventFlushTime) {
            eventWriter.flush()
            nextEventFlushTime += flushFrequency
          }
        }
      }
    })
    thread.setDaemon(true)
    thread
  }
}

object EventFileWriter {
  /** Creates a new [[EventFileWriter]].
    *
    * @param  workingDir     Directory in which to write the event file.
    * @param  queueCapacity  Maximum number of events pending to be written to disk before a call to `write()` blocks.
    * @param  flushFrequency Specifies how often to flush the written events to disk (in seconds).
    * @param  filenameSuffix Filename suffix to use for the event file.
    * @return Constructed event file writer.
    */
  def apply(
      workingDir: Path,
      queueCapacity: Int = 10,
      flushFrequency: Int = 10,
      filenameSuffix: String = ""
  ): EventFileWriter = {
    new EventFileWriter(workingDir, queueCapacity, flushFrequency, filenameSuffix)
  }
}

/** Helper class used by the [[EventFileWriter]] class, to write `Event` protocol buffers to files.
  *
  * @param  workingDir     Directory in which to write the event file.
  * @param  filenamePrefix Filename prefix to use for the event file.
  * @param  filenameSuffix Filename suffix to use for the event file.
  */
private[io] class EventWriter private[io](
    val workingDir: Path,
    val filenamePrefix: String,
    val filenameSuffix: String = ""
) {
  private[this] var _filePath            : Path                  = _
  private[this] var _fileStream          : Option[BufferedOutputStream] = None
  private[this] var _numOutstandingEvents: Int                   = 0

  /** Returns the path of the current events file. */
  def filePath: Path = {
    if (_filePath == null)
      initialize()
    _filePath
  }

  /** Determines the filename and opens the file for writing. If not called by the user, this method will be invoked
    * automatically by a call to `filename()` or `write()`. Returns `false` if the file could not be opened. If the file
    * exists and is open this is a no-op. If on the other hand the file was opened, but has since disappeared (e.g.,
    * deleted by another process), this method will open a new file with a new timestamp in its filename.
    *
    * The filename is set to `[filenamePrefix].out.events.[timestamp].[hostname][filenameSuffix]`.
    */
  def initialize(): Unit = {
    var initialized = false
    if (_fileStream.isDefined) {
      if (fileHasDisappeared) {
        // Warn the user about the data loss and then do some basic cleanup.
        if (_numOutstandingEvents > 0)
          EventWriter.logger.warn(
            s"Re-initialization: attempting to open a new file. ${_numOutstandingEvents} events will be lost.")
      } else {
        // No-op. File is present and the writer has been initialized.
        initialized = true
      }
    }

    if (!initialized) {
      val currentTime = System.currentTimeMillis().toDouble / 1000.0
      val hostname = java.net.InetAddress.getLocalHost.getHostName
      _filePath = workingDir.resolve(f"$filenamePrefix.out.tfevents.${currentTime.toInt}%010d.$hostname$filenameSuffix")
      _fileStream = Some(new BufferedOutputStream(Files.newOutputStream(
        _filePath, StandardOpenOption.CREATE_NEW, StandardOpenOption.APPEND)))
      _numOutstandingEvents = 0
      // Write the first event with the current version, and flush right away so the file contents can be easily
      // determined.
      val eventBuilder = Event.newBuilder()
      eventBuilder.setWallTime(currentTime)
      eventBuilder.setFileVersion(s"${EventWriter.VERSION_PREFIX}${EventWriter.VERSION_NUMBER}")
      write(eventBuilder.build())
      flush()
    }
  }

  /** Appends `event` to the events file. */
  def write(event: Event): Unit = {
    if (_filePath == null)
      initialize()
    _numOutstandingEvents += 1
    val recordBytes = event.toByteArray
    // Format of a single record:
    //  uint64    length
    //  uint32    masked crc of length
    //  byte      data[length]
    //  uint32    masked crc of data
    val encLength = Coding.encodeFixedInt64(recordBytes.length)
    val encLengthMaskedCrc = Coding.encodeFixedInt32(CRC32C.mask(CRC32C.value(encLength)))
    val encDataMaskedCrc = Coding.encodeFixedInt32(CRC32C.mask(CRC32C.value(recordBytes)))
    _fileStream.foreach(_.write(encLength ++ encLengthMaskedCrc ++ recordBytes ++ encDataMaskedCrc))
  }

  /** Pushes outstanding events to disk. */
  def flush(): Unit = {
    _fileStream.foreach(_.flush())
    _numOutstandingEvents = 0
  }

  /** Calls `flush()` and then closes the current event file. */
  def close(): Unit = {
    _fileStream.foreach(_.close())
    _numOutstandingEvents = 0
  }

  private[this] def fileHasDisappeared: Boolean = {
    if (Files.exists(_filePath)) {
      false
    } else {
      // This could happen if some other process removes the file.
      EventWriter.logger.error(s"The events file '${_filePath}' has disappeared.")
      true
    }
  }
}

private[io] object EventWriter {
  private[EventWriter] val logger: Logger = Logger(LoggerFactory.getLogger("Event Writer"))

  /** Prefix of the version string present in the first entry of every event file. */
  private[EventWriter] val VERSION_PREFIX: String = "brain.Event:"
  private[EventWriter] val VERSION_NUMBER: Int    = 2
}
