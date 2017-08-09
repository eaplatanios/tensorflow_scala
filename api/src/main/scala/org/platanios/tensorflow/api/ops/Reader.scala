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

package org.platanios.tensorflow.api.ops

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
  *
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
    IO.readerRead(handle, queue.handle, name)
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
    IO.readerReadUpTo(handle, queue.handle, numRecords, name)
  }

  /** Creates an op that returns the number of records that this reader has produced.
    *
    * This is the same as the number of [[read]] executions that have succeeded.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is an `INT64` scalar tensor.
    */
  def numRecordsProduced(name: String = s"$name/NumRecordsProduced"): Output = {
    IO.readerNumRecordsProduced(handle, name)
  }

  /** Creates an op that returns the number of work units that this reader has finished processing.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is an `INT64` scalar tensor.
    */
  def numWorkUnitsCompleted(name: String = s"$name/NumWorkUnitsCompleted"): Output = {
    IO.readerNumWorkUnitsCompleted(handle, name)
  }

  /** Creates an op that restores this reader to its initial clean state.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def reset(name: String = s"$name/Reset"): Op = {
    IO.readerReset(handle, name)
  }
}

class SerializableReader private[ops](override val handle: Output) extends Reader(handle) {
  /** Creates an op that produces a string tensor that encodes the state of this reader.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor.
    */
  def serializeState(name: String = s"$name/SerializeState"): Output = {
    IO.readerSerializeState(handle, name)
  }

  /** Creates an op that restores this reader to a previously serialized state.
    *
    * @param  state `STRING` scalar tensor containing the serialized reader state, matching this reader's type.
    * @param  name  Name for the created op.
    * @return Created op.
    */
  def restoreState(state: Output, name: String = s"$name/RestoreState"): Op = {
    IO.readerRestoreState(handle, state, name)
  }
}

object Reader {
  trait API {
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
      new SerializableReader(IO.createWholeFileReader(sharedName = sharedName, name = name))
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
      new Reader(IO.createTextLineReader(skipHeaderLines, sharedName = sharedName, name = name))
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
        compressionType: IO.CompressionType = IO.NoCompression, sharedName: String = "",
        name: String = "FixedLengthRecordReader"): Reader = {
      new Reader(IO.createFixedLengthRecordReader(
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
        compressionType: IO.CompressionType = IO.NoCompression, sharedName: String = "",
        name: String = "TFRecordReader"): Reader = {
      new Reader(IO.createTFRecordReader(compressionType.name, sharedName = sharedName, name = name))
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
      new Reader(IO.createIdentityReader(sharedName = sharedName, name = name))
    }
  }

  object API extends API
}
