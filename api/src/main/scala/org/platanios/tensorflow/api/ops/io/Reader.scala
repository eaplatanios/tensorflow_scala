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

package org.platanios.tensorflow.api.ops.io

import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}

/** Class that supports all TensorFlow reader implementations.
  *
  * Conceptually, readers convert string "work units" into records (i.e., key-value pairs). Typically the "work units"
  * are filenames and the records are extracted from the contents of those files. We want a single record produced per
  * step, but a work unit can correspond to many records.
  *
  * Therefore we introduce some decoupling using a [[Queue]]. The queue contains the work units and the reader dequeues
  * from the queue when it is asked to produce a record (e.g., via its `read` method) but it has already finished the
  * last work unit.
  *
  * @param handle Handle to the underlying TensorFlow reader.
  * @author Emmanouil Antonios Platanios
  */
class Reader private[ops](val handle: Output) {
  /** Name of this reader. */
  protected val name: String = handle.op.name.split("/").last

  /** Creates an op that reads the next record (i.e., key-value pair) produced by this reader.
    *
    * The op will dequeue from the input queue if necessary (e.g., when the reader needs to start reading from a new
    * file since it has finished with the previous file).
    *
    * @param  queue Queue to obtain the work units from.
    * @param  name  Name for the created op.
    * @return Created op outputs as a key-value pair.
    */
  def read(queue: Queue, name: String = s"$name/Read"): (Output, Output) = {
    Reader.readerRead(handle, queue.handle, name)
  }

  /** Creates an op that reads up to the next `numRecords` records (i.e., key-value pairs) produced by this reader.
    *
    * The op will dequeue from the input queue if necessary (e.g., when the reader needs to start reading from a new
    * file since it has finished with the previous file).
    *
    * @param  queue      Queue to obtain the work units from.
    * @param  numRecords `INT64` scalar tensor specifying how many records to read.
    * @param  name       Name for the created op.
    * @return Created op outputs as a key-value pair of one-dimensional tensors.
    */
  def readUpTo(queue: Queue, numRecords: Output, name: String = s"$name/ReadUpTo"): (Output, Output) = {
    Reader.readerReadUpTo(handle, queue.handle, numRecords, name)
  }

  /** Creates an op that returns the number of records that this reader has produced.
    *
    * This is the same as the number of [[read]] executions that have succeeded.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is an `INT64` scalar tensor.
    */
  def numRecordsProduced(name: String = s"$name/NumRecordsProduced"): Output = {
    Reader.readerNumRecordsProduced(handle, name)
  }

  /** Creates an op that returns the number of work units that this reader has finished processing.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is an `INT64` scalar tensor.
    */
  def numWorkUnitsCompleted(name: String = s"$name/NumWorkUnitsCompleted"): Output = {
    Reader.readerNumWorkUnitsCompleted(handle, name)
  }

  /** Creates an op that restores this reader to its initial clean state.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def reset(name: String = s"$name/Reset"): Op = {
    Reader.readerReset(handle, name)
  }
}

class SerializableReader private[ops](override val handle: Output) extends Reader(handle) {
  /** Creates an op that produces a string tensor that encodes the state of this reader.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor.
    */
  def serializeState(name: String = s"$name/SerializeState"): Output = {
    Reader.readerSerializeState(handle, name)
  }

  /** Creates an op that restores this reader to a previously serialized state.
    *
    * @param  state `STRING` scalar tensor containing the serialized reader state, matching this reader's type.
    * @param  name  Name for the created op.
    * @return Created op.
    */
  def restoreState(state: Output, name: String = s"$name/RestoreState"): Op = {
    Reader.readerRestoreState(handle, state, name)
  }
}

object Reader {
  private[io] trait API {
    /** Creates a reader that outputs the entire contents of a file as a value.
      *
      * To use, enqueue the filenames in a [[Queue]]. The output of [[Reader.read]] will be a filename (key) and the
      * contents of that file (value).
      *
      * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
      *                    multiple sessions.
      * @param  name       Name for the created op.
      * @return Constructed reader.
      */
    def wholeFileReader(sharedName: String = "", name: String = "WholeFileReader"): SerializableReader = {
      new SerializableReader(createWholeFileReader(sharedName = sharedName, name = name))
    }

    /** Creates a reader that outputs the lines of a text file delimited by the new-line character `\n`.
      *
      * **Note:** New-line characters are stripped from the output.
      *
      * @param  skipHeaderLines Number of lines to skip from the beginning of every file.
      * @param  sharedName      If non-empty, then the constructed reader will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the created op.
      * @return Constructed reader.
      */
    def textLineReader(skipHeaderLines: Int = 0, sharedName: String = "", name: String = "TextLineReader"): Reader = {
      new Reader(createTextLineReader(skipHeaderLines, sharedName = sharedName, name = name))
    }

    /** Creates a reader that outputs fixed-length records from a file.
      *
      * @param  recordBytes     Number of bytes in the record.
      * @param  headerBytes     Number of bytes in the header.
      * @param  footerBytes     Number of bytes in the footer.
      * @param  hopBytes        Number of bytes to hop before each read.
      * @param  compressionType Type of compression for the file.
      * @param  sharedName      If non-empty, then the constructed reader will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the created op.
      * @return Constructed reader.
      */
    def fixedLengthRecordReader(
        recordBytes: Int, headerBytes: Int = 0, footerBytes: Int = 0, hopBytes: Int = 0,
        compressionType: Files.CompressionType = Files.NoCompression, sharedName: String = "",
        name: String = "FixedLengthRecordReader"): Reader = {
      new Reader(createFixedLengthRecordReader(
        recordBytes, headerBytes, footerBytes, hopBytes, compressionType.name, sharedName = sharedName, name = name))
    }

    /** Creates a reader that outputs the records from a TensorFlow records file.
      *
      * @param  compressionType Type of compression for the file.
      * @param  sharedName      If non-empty, then the constructed reader will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the created op.
      * @return Constructed reader.
      */
    def tfRecordReader(
        compressionType: Files.CompressionType = Files.NoCompression, sharedName: String = "",
        name: String = "TFRecordReader"): Reader = {
      new Reader(createTFRecordReader(compressionType.name, sharedName = sharedName, name = name))
    }

    /** Creates a reader that outputs the queued work as both the key and value.
      *
      * To use, enqueue strings in a [[Queue]]. The output of [[Reader.read]] will be a string (key) and the same string
      * repeated (value).
      *
      * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
      *                    multiple sessions.
      * @param  name       Name for the created op.
      * @return Constructed reader.
      */
    def identityReader(sharedName: String = "", name: String = "IdentityReader"): Reader = {
      new Reader(createIdentityReader(sharedName = sharedName, name = name))
    }
  }

  /** Creates an op that outputs the entire contents of a file as a value.
    *
    * To use, enqueue the filenames in a [[Queue]]. The output of [[readerRead]] will be a filename (key) and the
    * contents of that file (value).
    *
    * @param  container  If non-empty, then the constructed reader is placed in the provided container. Otherwise, a
    *                    default container is used.
    * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
    *                    multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[io] def createWholeFileReader(
      container: String = "", sharedName: String = "", name: String = "WholeFileReader"): Output = {
    Op.Builder(opType = "WholeFileReaderV2", name = name)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that outputs the lines of a text file delimited by the new line character `\n`.
    *
    * **Note:** New-line characters are stripped from the output.
    *
    * @param  skipHeaderLines Number of lines to skip from the beginning of every file.
    * @param  container       If non-empty, then the constructed reader is placed in the provided container. Otherwise,
    *                         a default container is used.
    * @param  sharedName      If non-empty, then the constructed reader will be shared under the the provided name
    *                         across multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[io] def createTextLineReader(
      skipHeaderLines: Int = 0, container: String = "", sharedName: String = "",
      name: String = "TextLineReader"): Output = {
    Op.Builder(opType = "TextLineReaderV2", name = name)
        .setAttribute("skip_header_lines", skipHeaderLines)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that outputs fixed-length records from a file.
    *
    * @param  recordBytes     Number of bytes in the record.
    * @param  headerBytes     Number of bytes in the header.
    * @param  footerBytes     Number of bytes in the footer.
    * @param  hopBytes        Number of bytes to hop before each read.
    * @param  compressionType Type of compression for the file. Currently ZLIB and GZIP are supported. Defaults to `""`,
    *                         meaning no compression.
    * @param  container       If non-empty, then the constructed reader is placed in the provided container. Otherwise,
    *                         a default container is used.
    * @param  sharedName      If non-empty, then the constructed reader will be shared under the the provided name
    *                         across multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[io] def createFixedLengthRecordReader(
      recordBytes: Int, headerBytes: Int = 0, footerBytes: Int = 0, hopBytes: Int = 0, compressionType: String = "",
      container: String = "", sharedName: String = "", name: String = "FixedLengthRecordReader"): Output = {
    Op.Builder(opType = "FixedLengthRecordReaderV2", name = name)
        .setAttribute("record_bytes", recordBytes)
        .setAttribute("header_bytes", headerBytes)
        .setAttribute("footer_bytes", footerBytes)
        .setAttribute("hop_bytes", hopBytes)
        .setAttribute("encoding", compressionType)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that outputs the records from a TensorFlow records file.
    *
    * @param  compressionType Type of compression for the file. Currently ZLIB and GZIP are supported. Defaults to `""`,
    *                         meaning no compression.
    * @param  container       If non-empty, then the constructed reader is placed in the provided container. Otherwise,
    *                         a default container is used.
    * @param  sharedName      If non-empty, then the constructed reader will be shared under the the provided name
    *                         across multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[io] def createTFRecordReader(
      compressionType: String = "", container: String = "", sharedName: String = "",
      name: String = "TFRecordReader"): Output = {
    Op.Builder(opType = "TFRecordReaderV2", name = name)
        .setAttribute("compression_type", compressionType)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that outputs the queued work as both the key and value.
    *
    * To use, enqueue strings in a [[Queue]]. The output of [[readerRead]] will be a string (key) and the same string
    * repeated (value).
    *
    * @param  container  If non-empty, then the constructed reader is placed in the provided container. Otherwise, a
    *                    default container is used.
    * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
    *                    multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[io] def createIdentityReader(
      container: String = "", sharedName: String = "", name: String = "IdentityReader"): Output = {
    Op.Builder(opType = "IdentityReaderV2", name = name)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that reads the next record (i.e., key-value pair) produced by a reader.
    *
    * The op will dequeue from the input queue if necessary (e.g., when the reader needs to start reading from a new
    * file since it has finished with the previous file).
    *
    * @param  readerHandle Handle to a reader.
    * @param  queueHandle  Handle to a queue.
    * @param  name         Name for the created op.
    * @return Created op outputs as a key-value pair.
    */
  private[io] def readerRead(
      readerHandle: Output, queueHandle: Output, name: String = "ReaderRead"): (Output, Output) = {
    val outputs = Op.Builder(opType = "ReaderReadV2", name = name)
        .addInput(readerHandle)
        .addInput(queueHandle)
        .build().outputs
    (outputs(0), outputs(1))
  }

  /** Creates an op that reads up to the next `numRecords` records (i.e., key-value pairs) produced by a reader.
    *
    * The op will dequeue from the input queue if necessary (e.g., when the reader needs to start reading from a new
    * file since it has finished with the previous file).
    *
    * @param  readerHandle Handle to a reader.
    * @param  queueHandle  Handle to a queue.
    * @param  numRecords   `INT64` scalar tensor specifying how many records to read.
    * @param  name         Name for the created op.
    * @return Created op outputs as a key-value pair of one-dimensional tensors,
    */
  private[io] def readerReadUpTo(
      readerHandle: Output, queueHandle: Output, numRecords: Output,
      name: String = "ReaderReadUpTo"): (Output, Output) = {
    val outputs = Op.Builder(opType = "ReaderReadUpToV2", name = name)
        .addInput(readerHandle)
        .addInput(queueHandle)
        .addInput(numRecords)
        .build().outputs
    (outputs(0), outputs(1))
  }

  /** Creates an op that returns the number of records that the provided reader has produced.
    *
    * This is the same as the number of [[readerRead]] executions that have succeeded.
    *
    * @param  readerHandle Handle to a reader.
    * @param  name         Name for the created op.
    * @return Created op output, which is an `INT64` scalar tensor.
    */
  private[io] def readerNumRecordsProduced(readerHandle: Output, name: String = "ReaderNumRecordsProduced"): Output = {
    Op.Builder(opType = "ReaderNumRecordsProducedV2", name = name)
        .addInput(readerHandle)
        .build().outputs(0)
  }

  /** Creates an op that returns the number of work units that the provided reader has finished processing.
    *
    * @param  readerHandle Handle to a reader.
    * @param  name         Name for the created op.
    * @return Created op output, which is an `INT64` scalar tensor.
    */
  private[io] def readerNumWorkUnitsCompleted(
      readerHandle: Output, name: String = "ReaderNumWorkUnitsCompleted"): Output = {
    Op.Builder(opType = "ReaderNumWorkUnitsCompletedV2", name = name)
        .addInput(readerHandle)
        .build().outputs(0)
  }

  /** Creates an op that produces a string tensor that encodes the state of the provided reader.
    *
    * **Note:** Not all readers support being serialized and so this function could result in an `Unimplemented` error
    * being thrown.
    *
    * @param  readerHandle Handle to a reader.
    * @param  name         Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor.
    */
  private[io] def readerSerializeState(readerHandle: Output, name: String = "ReaderSerializeState"): Output = {
    Op.Builder(opType = "ReaderSerializeStateV2", name = name)
        .addInput(readerHandle)
        .build().outputs(0)
  }

  /** Creates an op that restores the provided reader to a previously serialized state.
    *
    * **Note:** Not all readers support being restored and so this function could result in an `Unimplemented` error
    * being thrown.
    *
    * @param  readerHandle Handle to a reader.
    * @param  state        `STRING` scalar tensor containing the serialized reader state, matching the provided reader
    *                      type.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[io] def readerRestoreState(
      readerHandle: Output, state: Output, name: String = "ReaderRestoreState"): Op = {
    Op.Builder(opType = "ReaderRestoreStateV2", name = name)
        .addInput(readerHandle)
        .addInput(state)
        .build()
  }

  /** Creates an op that restores the provided reader to its initial clean state.
    *
    * @param  readerHandle Handle to a reader.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  private[io] def readerReset(readerHandle: Output, name: String = "ReaderReset"): Op = {
    Op.Builder(opType = "ReaderResetV2", name = name)
        .addInput(readerHandle)
        .build()
  }

  private[io] object Gradients {
    GradientsRegistry.registerNonDifferentiable("WholeFileReader")
    GradientsRegistry.registerNonDifferentiable("WholeFileReaderV2")
    GradientsRegistry.registerNonDifferentiable("TextLineReader")
    GradientsRegistry.registerNonDifferentiable("TextLineReaderV2")
    GradientsRegistry.registerNonDifferentiable("FixedLengthRecordReader")
    GradientsRegistry.registerNonDifferentiable("FixedLengthRecordReaderV2")
    GradientsRegistry.registerNonDifferentiable("TFRecordReader")
    GradientsRegistry.registerNonDifferentiable("TFRecordReaderV2")
    GradientsRegistry.registerNonDifferentiable("LMDBReader")
    GradientsRegistry.registerNonDifferentiable("IdentityReader")
    GradientsRegistry.registerNonDifferentiable("IdentityReaderV2")
    GradientsRegistry.registerNonDifferentiable("ReaderRead")
    GradientsRegistry.registerNonDifferentiable("ReaderReadV2")
    GradientsRegistry.registerNonDifferentiable("ReaderReadUpTo")
    GradientsRegistry.registerNonDifferentiable("ReaderReadUpToV2")
    GradientsRegistry.registerNonDifferentiable("ReaderNumRecordsProduced")
    GradientsRegistry.registerNonDifferentiable("ReaderNumRecordsProducedV2")
    GradientsRegistry.registerNonDifferentiable("ReaderNumWorkUnitsCompleted")
    GradientsRegistry.registerNonDifferentiable("ReaderNumWorkUnitsCompletedV2")
    GradientsRegistry.registerNonDifferentiable("ReaderSerializeState")
    GradientsRegistry.registerNonDifferentiable("ReaderSerializeStateV2")
    GradientsRegistry.registerNonDifferentiable("ReaderRestoreState")
    GradientsRegistry.registerNonDifferentiable("ReaderRestoreStateV2")
    GradientsRegistry.registerNonDifferentiable("ReaderReset")
    GradientsRegistry.registerNonDifferentiable("ReaderResetV2")
  }
}
