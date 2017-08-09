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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.Indexer.Implicits._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.{DataType, INT64, STRING}

import scala.language.postfixOps

/** Contains helper functions for creating IO-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
object IO {
  sealed trait CompressionType {
    val name: String
  }

  case object NoCompression extends CompressionType {
    override val name: String = ""
  }

  case object ZLIBCompression extends CompressionType {
    override val name: String = "ZLIB"
  }

  case object GZIPCompression extends CompressionType {
    override val name: String = "GZIP"
  }

  //region General IO Ops

  /** Creates an op that reads and outputs the entire contents of the file pointed to by the input filename.
    *
    * @param  filename `STRING` scalar tensor containing the filename.
    * @param  name     Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the file contents.
    */
  private[api] def readFile(filename: Output, name: String = "ReadFile"): Output = {
    Op.Builder(opType = "ReadFile", name = name)
        .addInput(filename)
        .build().outputs(0)
  }

  /** Creates an op that writes `contents` to the file pointed to by the input filename.
    *
    * The op creates the file and recursively creates the directory, if it does not already exist.
    *
    * @param  filename `STRING` scalar tensor containing the filename.
    * @param  contents `STRING` scalar tensor containing the contents to write to the provided file.
    * @param  name     Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the file contents.
    */
  private[api] def writeFile(filename: Output, contents: Output, name: String = "WriteFile"): Op = {
    Op.Builder(opType = "WriteFile", name = name)
        .addInput(filename)
        .addInput(contents)
        .build()
  }

  /** Creates an op that returns the set of files matching one or more glob patterns.
    *
    * **Note:** The op only supports wildcard characters in the basename portion of the pattern and not in the directory
    * portion.
    *
    * @param  pattern `STRING` scalar or vector tensor containing the shell wildcard pattern(s).
    * @param  name    Name for the created op.
    * @return Created op output, which is a `STRING` vector tensor containing the matching filenames.
    */
  private[api] def matchingFiles(pattern: Output, name: String = "MatchingFiles"): Output = {
    Op.Builder(opType = "MatchingFiles", name = name)
        .addInput(pattern)
        .build().outputs(0)
  }

  //endregion General IO Ops

  //region Reader Ops

  /** Creates an op that outputs the entire contents of a file as a value.
    *
    * To use, enqueue the filenames in a [[Queue]]. The output of [[readerRead]] will be a filename (key) and the
    * contents of that file (value).
    *
    * @param  container  If non-empty, then the constructed reader is placed in the provided container. Otherwise, a
    *                    default container is used.
    * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
    *                    multiple sessions.
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[api] def createWholeFileReader(
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
    *                         across multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[api] def createTextLineReader(
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
    *                         across multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[api] def createFixedLengthRecordReader(
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
    *                         across multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[api] def createTFRecordReader(
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
    *                    multiple sessions.
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to constructed reader.
    */
  private[api] def createIdentityReader(
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
  private[api] def readerRead(
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
  private[api] def readerReadUpTo(
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
  private[api] def readerNumRecordsProduced(readerHandle: Output, name: String = "ReaderNumRecordsProduced"): Output = {
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
  private[api] def readerNumWorkUnitsCompleted(
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
  private[api] def readerSerializeState(readerHandle: Output, name: String = "ReaderSerializeState"): Output = {
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
  private[api] def readerRestoreState(
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
  private[api] def readerReset(readerHandle: Output, name: String = "ReaderReset"): Op = {
    Op.Builder(opType = "ReaderResetV2", name = name)
        .addInput(readerHandle)
        .build()
  }

  //endregion Reader Ops

  //region Dataset Ops

  /** Creates a tensor dataset op.
    *
    * A tensor dataset is a dataset that emits `components` as a tuple of tensors once.
    *
    * @param  components Tensors to emit.
    * @param  shapes     Shapes of the emitted tensors.
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[api] def createTensorDataset(
      components: Seq[Output], shapes: Seq[Shape], name: String = "TensorDataset"): Output = {
    if (components.zip(shapes).exists(p => !p._1.shape.isCompatibleWith(p._2)))
      throw new IllegalArgumentException(
        "Each tensor in 'components' must have shape compatible with the corresponding shape in 'shapes'.")
    Op.Builder(opType = "TensorDataset", name = name)
        .addInputList(components)
        .setAttribute("output_shapes", shapes.toArray)
        .build().outputs(0)
  }

  /** Creates a tensor slice dataset op.
    *
    * A tensor slice dataset is a dataset that emits each axis-0 slice of `components` once.
    *
    * @param  components Tensors, whose axis-0 slices to emit.
    * @param  shapes     Shapes of the emitted tensors.
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[api] def createTensorSliceDataset(
      components: Seq[Output], shapes: Seq[Shape], name: String = "TensorSliceDataset"): Output = {
    if (components.zip(shapes).exists(p => !p._1.shape(1 ::).isCompatibleWith(p._2)))
      throw new IllegalArgumentException(
        "The axis-0 slice of each tensor in 'components' " +
            "must have shape compatible with the corresponding shape in 'shapes'.")
    Op.Builder(opType = "TensorSliceDataset", name = name)
        .addInputList(components)
        .setAttribute("output_shapes", shapes.toArray)
        .build().outputs(0)
  }

  /** Creates a range dataset op.
    *
    * A range dataset is a dataset that contains a range of values.
    *
    * @param  start           `INT64` tensor containing the start value for the range.
    * @param  stop            `INT64` tensor containing the stop value for the range.
    * @param  step            `INT64` tensor containing the step value for the range.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If any of `start`, `stop`, or `step` has invalid data type.
    */
  @throws[IllegalArgumentException]
  private[api] def createRangeDataset(
      start: Output, stop: Output, step: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "RangeDataset"): Output = {
    if (start.dataType != INT64)
      throw new IllegalArgumentException(s"'start' (dataType = ${start.dataType}) must be an INT64 tensor.")
    if (stop.dataType != INT64)
      throw new IllegalArgumentException(s"'stop' (dataType = ${stop.dataType}) must be an INT64 tensor.")
    if (step.dataType != INT64)
      throw new IllegalArgumentException(s"'step' (dataType = ${step.dataType}) must be an INT64 tensor.")
    Op.Builder(opType = "RangeDataset", name = name)
        .addInput(start)
        .addInput(stop)
        .addInput(step)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates a sparse tensor slice dataset op.
    *
    * A tensor slice dataset is a dataset that that splits a sparse tensor into elements row-wise and emits each such
    * element once.
    *
    * @param  indices    `INT64` tensor containing the indices of the non-zero elements of the tensor.
    * @param  values     Tensor containing the values of the tensor corresponding to `indices`.
    * @param  denseShape `INT64` tensor containing the full/dense shape of the tensor.
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `indices` or `denseShape` have invalid data type.
    */
  @throws[IllegalArgumentException]
  private[api] def createSparseTensorSliceDataset(
      indices: Output, values: Output, denseShape: Output, name: String = "SparseTensorSliceDataset"): Output = {
    if (indices.dataType != INT64)
      throw new IllegalArgumentException(s"'indices' (dataType = ${indices.dataType}) must be an INT64 tensor.")
    if (denseShape.dataType != INT64)
      throw new IllegalArgumentException(s"'denseShape' (dataType = ${denseShape.dataType}) must be an INT64 tensor.")
    Op.Builder(opType = "SparseTensorSliceDataset", name = name)
        .addInput(indices)
        .addInput(values)
        .addInput(denseShape)
        .build().outputs(0)
  }

  /** Creates a text-line dataset op.
    *
    * A text-line dataset emits the lines of one or more text files.
    *
    * **Note:** New-line characters are stripped from the output.
    *
    * @param  filenames       `STRING` scalar or vector tensor containing the the name(s) of the file(s) to be read.
    * @param  compressionType `STRING` scalar tensor containing the type of compression for the file. Currently ZLIB and
    *                         GZIP are supported. Defaults to `""`, meaning no compression.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If any of the arguments has invalid data type or shape.
    */
  @throws[IllegalArgumentException]
  private[api] def createTextLineDataset(
      filenames: Output, compressionType: Output, name: String = "TextLineDataset"): Output = {
    if (filenames.dataType != STRING)
      throw new IllegalArgumentException(s"'filenames' (dataType = ${filenames.dataType}) must be a STRING tensor.")
    if (filenames.rank != -1 && filenames.rank > 1)
      throw new IllegalArgumentException(s"'filenames' (rank = ${filenames.rank}) must be at most 1.")
    if (compressionType.dataType != STRING)
      throw new IllegalArgumentException(
        s"'compressionType' (dataType = ${compressionType.dataType}) must be a STRING tensor.")
    if (compressionType.rank != -1 && compressionType.rank > 0)
      throw new IllegalArgumentException(s"'compressionType' (rank = ${compressionType.rank}) must be equal to 0.")
    Op.Builder(opType = "TextLineDataset", name = name)
        .addInput(filenames)
        .addInput(compressionType)
        .build().outputs(0)
  }

  /** Creates an op that outputs fixed-length records from a file.
    *
    * @param  filenames   `STRING` scalar or vector tensor containing the the name(s) of the file(s) to be read.
    * @param  recordBytes `INT64` scalar tensor containing the number of bytes in the record.
    * @param  headerBytes `INT64` scalar tensor containing the number of bytes in the header (i.e., the number of
    *                     bytes to skip at the beginning of a file).
    * @param  footerBytes `INT64` scalar tensor containing the number of bytes in the footer (i.e., the number of
    *                     bytes to skip at the end of a file).
    * @param  name        Name for the created op.
    * @return Created op output, which is a handle to constructed dataset.
    * @throws IllegalArgumentException If any of the arguments has invalid data type or shape.
    */
  @throws[IllegalArgumentException]
  private[api] def createFixedLengthRecordDataset(
      filenames: Output, recordBytes: Output, headerBytes: Output, footerBytes: Output,
      name: String = "FixedLengthRecordDataset"): Output = {
    if (filenames.dataType != STRING)
      throw new IllegalArgumentException(s"'filenames' (dataType = ${filenames.dataType}) must be a STRING tensor.")
    if (filenames.rank != -1 && filenames.rank > 1)
      throw new IllegalArgumentException(s"'filenames' (rank = ${filenames.rank}) must be at most 1.")
    if (recordBytes.dataType != INT64)
      throw new IllegalArgumentException(
        s"'recordBytes' (dataType = ${recordBytes.dataType}) must be a INT64 tensor.")
    if (recordBytes.rank != -1 && recordBytes.rank != 0)
      throw new IllegalArgumentException(s"'recordBytes' (rank = ${recordBytes.rank}) must be equal to 0.")
    if (headerBytes.dataType != INT64)
      throw new IllegalArgumentException(
        s"'headerBytes' (dataType = ${headerBytes.dataType}) must be a INT64 tensor.")
    if (headerBytes.rank != -1 && headerBytes.rank != 0)
      throw new IllegalArgumentException(s"'headerBytes' (rank = ${headerBytes.rank}) must be equal to 0.")
    if (footerBytes.dataType != INT64)
      throw new IllegalArgumentException(
        s"'recordBytes' (dataType = ${footerBytes.dataType}) must be a INT64 tensor.")
    if (footerBytes.rank != -1 && footerBytes.rank != 0)
      throw new IllegalArgumentException(s"'footerBytes' (rank = ${footerBytes.rank}) must be equal to 0.")
    Op.Builder(opType = "FixedLengthRecordDataset", name = name)
        .addInput(filenames)
        .addInput(headerBytes)
        .addInput(recordBytes)
        .addInput(footerBytes)
        .build().outputs(0)
  }

  /** Creates a TensorFlow records dataset op.
    *
    * A TensorFlow records dataset emits the records from one or more TFRecord files.
    *
    * @param  filenames       `STRING` scalar or vector tensor containing the the name(s) of the file(s) to be read.
    * @param  compressionType `STRING` scalar tensor containing the type of compression for the file. Currently ZLIB and
    *                         GZIP are supported. Defaults to `""`, meaning no compression.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If any of the arguments has invalid data type or shape.
    */
  @throws[IllegalArgumentException]
  private[api] def createTFRecordDataset(
      filenames: Output, compressionType: Output, name: String = "TFRecordDataset"): Output = {
    if (filenames.dataType != STRING)
      throw new IllegalArgumentException(s"'filenames' (dataType = ${filenames.dataType}) must be a STRING tensor.")
    if (filenames.rank != -1 && filenames.rank > 1)
      throw new IllegalArgumentException(s"'filenames' (rank = ${filenames.rank}) must be at most 1.")
    if (compressionType.dataType != STRING)
      throw new IllegalArgumentException(
        s"'compressionType' (dataType = ${compressionType.dataType}) must be a STRING tensor.")
    if (compressionType.rank != -1 && compressionType.rank > 0)
      throw new IllegalArgumentException(s"'compressionType' (rank = ${compressionType.rank}) must be equal to 0.")
    Op.Builder(opType = "TFRecordDataset", name = name)
        .addInput(filenames)
        .addInput(compressionType)
        .build().outputs(0)
  }

  /** Creates an op that zips multiple datasets together.
    *
    * A zip dataset is a dataset that zips together multiple datasets.
    *
    * @param  datasets        Tensors containing the handles of the datasets to zip together.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[api] def datasetZip(
      datasets: Seq[Output], outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetZip"): Output = {
    Op.Builder(opType = "SparseTensorSliceDataset", name = name)
        .addInputList(datasets)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op that concatenates two datasets.
    *
    * A concatenated dataset is a dataset that concatenates together two other datasets.
    *
    * @param  dataset1        First dataset handle.
    * @param  dataset2        Second dataset handle.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[api] def datasetConcatenate(
      dataset1: Output, dataset2: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetConcatenate"): Output = {
    Op.Builder(opType = "ConcatenateDataset", name = name)
        .addInput(dataset1)
        .addInput(dataset2)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op that repeats a dataset.
    *
    * A repeated dataset is a dataset that emits the outputs of another dataset a number of times.
    *
    * @param  dataset         Handle of the dataset to repeat.
    * @param  count           `INT64` scalar tensor containing the number of times to repeat the provided dataset. A
    *                         value of `-1` corresponds to repeating it indefinitely.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `count` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[api] def datasetRepeat(
      dataset: Output, count: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetRepeat"): Output = {
    if (count.dataType != INT64)
      throw new IllegalArgumentException(s"'count' (dataType = ${count.dataType}) must be an INT64 tensor.")
    if (count.rank != -1 && count.rank > 0)
      throw new IllegalArgumentException(s"'count' (rank = ${count.rank}) must be equal to 0.")
    Op.Builder(opType = "RepeatDataset", name = name)
        .addInput(dataset)
        .addInput(count)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains `count` entries from the provided dataset.
    *
    * @param  dataset         Handle of the dataset to take entries from.
    * @param  count           `INT64` scalar tensor containing the number of entries to take from the provided dataset.
    *                         A value of `-1` corresponds to taking all the entries.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `count` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[api] def datasetTake(
      dataset: Output, count: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetTake"): Output = {
    if (count.dataType != INT64)
      throw new IllegalArgumentException(s"'count' (dataType = ${count.dataType}) must be an INT64 tensor.")
    if (count.rank != -1 && count.rank > 0)
      throw new IllegalArgumentException(s"'count' (rank = ${count.rank}) must be equal to 0.")
    Op.Builder(opType = "TakeDataset", name = name)
        .addInput(dataset)
        .addInput(count)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains all entries from the provided dataset except the first `count`.
    *
    * @param  dataset         Handle of the dataset to skip entries from.
    * @param  count           `INT64` scalar tensor containing the number of entries to skip from the provided dataset.
    *                         A value of `-1` corresponds to skipping all the entries.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `count` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[api] def datasetSkip(
      dataset: Output, count: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetSkip"): Output = {
    if (count.dataType != INT64)
      throw new IllegalArgumentException(s"'count' (dataType = ${count.dataType}) must be an INT64 tensor.")
    if (count.rank != -1 && count.rank > 0)
      throw new IllegalArgumentException(s"'count' (rank = ${count.rank}) must be equal to 0.")
    Op.Builder(opType = "SkipDataset", name = name)
        .addInput(dataset)
        .addInput(count)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that batches `batchSize` elements from `dataset`.
    *
    * @param  dataset         Handle of the dataset to batch elements from.
    * @param  batchSize       `INT64` scalar tensor containing the batch size to use.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `batchSize` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[api] def datasetBatch(
      dataset: Output, batchSize: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetBatch"): Output = {
    if (batchSize.dataType != INT64)
      throw new IllegalArgumentException(s"'batchSize' (dataType = ${batchSize.dataType}) must be an INT64 tensor.")
    if (batchSize.rank != -1 && batchSize.rank > 0)
      throw new IllegalArgumentException(s"'batchSize' (rank = ${batchSize.rank}) must be equal to 0.")
    Op.Builder(opType = "BatchDataset", name = name)
        .addInput(dataset)
        .addInput(batchSize)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that batches and pads `batchSize` elements from `dataset`.
    *
    * @param  dataset       Handle of the dataset to batch elements from.
    * @param  batchSize     `INT64` scalar tensor containing the batch size to use.
    * @param  paddedShapes  Sequence of `INT64` rank-1 tensors (i.e., vectors) representing the desired padded shapes of
    *                       the corresponding output components. These shapes may be partially specified, using `-1` to
    *                       indicate that a particular dimension should be padded to the maximum size of all batch
    *                       elements.
    * @param  paddingValues Sequence of scalar tensors containing the padding value to use for each of the outputs.
    * @param  outputShapes  Output shapes of the created dataset.
    * @param  name          Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If any of the provided arguments has invalid data type or shape.
    */
  @throws[IllegalArgumentException]
  private[api] def datasetPaddedBatch(
      dataset: Output, batchSize: Output, paddedShapes: Seq[Output], paddingValues: Seq[Output],
      outputShapes: Seq[Shape], name: String = "DatasetPaddedBatch"): Output = {
    if (batchSize.dataType != INT64)
      throw new IllegalArgumentException(s"'batchSize' (dataType = ${batchSize.dataType}) must be an INT64 tensor.")
    if (batchSize.rank != -1 && batchSize.rank > 0)
      throw new IllegalArgumentException(s"'batchSize' (rank = ${batchSize.rank}) must be equal to 0.")
    if (paddedShapes.exists(_.dataType != INT64))
      throw new IllegalArgumentException("'paddedShapes' must all be INT64 tensors.")
    if (paddedShapes.exists(v => v.rank != -1 && v.rank != 1))
      throw new IllegalArgumentException("'paddedShapes' must all be vector tensors (i.e., must have rank 1).")
    if (paddingValues.exists(v => v.rank != -1 && v.rank != 0))
      throw new IllegalArgumentException("'paddingValues' must all be scalar tensors (i.e., must have rank 0).")
    if (paddedShapes.size != paddingValues.size)
      throw new IllegalArgumentException(
        s"'paddedShapes' (number = ${paddedShapes.size}) and 'paddingValues' (number = ${paddingValues.size}) must " +
            "contain the same number of tensors.")
    if (paddedShapes.size != outputShapes.size)
      throw new IllegalArgumentException(
        s"'paddedShapes' (number = ${paddedShapes.size}) and 'outputShapes' (number = ${outputShapes.size}) must " +
            "contain the same number of tensors.")
    Op.Builder(opType = "PaddedBatchDataset", name = name)
        .addInput(dataset)
        .addInput(batchSize)
        .addInputList(paddedShapes)
        .addInputList(paddingValues)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  // TODO: [DATASETS] "denseToSparseBatch".

  /** Creates an op representing a dataset that shuffles elements from `dataset` pseudorandomly and batches them in
    * batches with size equal to `batchSize`.
    *
    * @param  dataset         Handle of the dataset to batch elements from.
    * @param  batchSize       `INT64` scalar tensor containing the batch size to use.
    * @param  seed1           `INT64` scalar tensor containing a seed value for the random number generator. If either
    *                         seed or seed2 is set to be non-zero, the random number generator is seeded by the given
    *                         seed. Otherwise, a random seed is used.
    * @param  seed2           `INT64` scalar tensor containing a second seed value to avoid seed collision.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If any of `batchSize`, `seed1`, or `seed2` has invalid data type or rank (i.e.,
    *                                  if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[api] def datasetShuffle(
      dataset: Output, batchSize: Output, seed1: Output, seed2: Output, outputDataTypes: Seq[DataType],
      outputShapes: Seq[Shape], name: String = "DatasetShuffle"): Output = {
    if (batchSize.dataType != INT64)
      throw new IllegalArgumentException(s"'batchSize' (dataType = ${batchSize.dataType}) must be an INT64 tensor.")
    if (batchSize.rank != -1 && batchSize.rank > 0)
      throw new IllegalArgumentException(s"'batchSize' (rank = ${batchSize.rank}) must be equal to 0.")
    if (seed1.dataType != INT64)
      throw new IllegalArgumentException(s"'seed1' (dataType = ${seed1.dataType}) must be an INT64 tensor.")
    if (seed1.rank != -1 && seed1.rank > 0)
      throw new IllegalArgumentException(s"'seed1' (rank = ${seed1.rank}) must be equal to 0.")
    if (seed2.dataType != INT64)
      throw new IllegalArgumentException(s"'seed2' (dataType = ${seed2.dataType}) must be an INT64 tensor.")
    if (seed2.rank != -1 && seed2.rank > 0)
      throw new IllegalArgumentException(s"'seed2' (rank = ${seed2.rank}) must be equal to 0.")
    Op.Builder(opType = "ShuffleDataset", name = name)
        .addInput(dataset)
        .addInput(batchSize)
        .addInput(seed1)
        .addInput(seed2)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that caches elements from `dataset`.
    *
    * A cached dataset will iterate over the input dataset and store the tensors it gets. If the cache already exists,
    * it will be used. If the cache is inappropriate (e.g., it cannot be opened, or contains tensors of the wrong shape
    * or size), an error will the returned when used.
    *
    * @param  dataset         Handle of the dataset to cache.
    * @param  filename        `STRING` scalar tensor containing a path in the filesystem where the dataset should be
    *                         cached. Note that this should be a directory and not a file.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `filename` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[api] def datasetCache(
      dataset: Output, filename: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetCache"): Output = {
    if (filename.dataType != STRING)
      throw new IllegalArgumentException(s"'filename' (dataType = ${filename.dataType}) must be a STRING tensor.")
    if (filename.rank != -1 && filename.rank > 0)
      throw new IllegalArgumentException(s"'filename' (rank = ${filename.rank}) must be equal to 0.")
    Op.Builder(opType = "CacheDataset", name = name)
        .addInput(dataset)
        .addInput(filename)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains all entries from the provided dataset, but ignores all errors.
    *
    * @param  dataset         Handle of the dataset to take entries from.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[api] def datasetIgnoreErrors(
      dataset: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetIgnoreErrors"): Output = {
    Op.Builder(opType = "IgnoreErrorsDataset", name = name)
        .addInput(dataset)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  // TODO: [DATASETS] [FUNCTIONS] "map", "parallelMap", "flatMap", "interleave", "groupByWindow", and "filter".

  //endregion Dataset Ops

  //region Iterator Ops

  /** Creates an op that is a container for an `Iterator` resource.
    *
    * @param  container       If non-empty, then the constructed iterator is placed in the provided container.
    *                         Otherwise, a default container is used.
    * @param  sharedName      If non-empty, then the constructed iterator will be shared under the the provided name
    *                         across multiple sessions.
    * @param  outputDataTypes Output data types of the created iterator.
    * @param  outputShapes    Output shapes of the created iterator.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the constructed iterator.
    */
  private[api] def createIterator(
      container: String = "", sharedName: String = "", outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "Iterator"): Output = {
    Op.Builder(opType = "Iterator", name = name)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op that makes a new iterator for the provided dataset and stores it in the container pointed to by the
    * provided iterator handle.
    *
    * **Note:** The created op may be executed multiple times. Each execution will reset the iterator in `iterator` to
    * the first element of `dataset`.
    *
    * @param  dataset  Handle of the dataset.
    * @param  iterator Handle of the iterator.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private[api] def makeIterator(dataset: Output, iterator: Output, name: String = "MakeIterator"): Op = {
    Op.Builder(opType = "Iterator", name = name)
        .addInput(dataset)
        .addInput(iterator)
        .build()
  }

  // TODO: [DATASETS] [FUNCTIONS] "oneShotIterator".

  /** Creates an op that gets the next output from the provided iterator.
    *
    * @param  iterator        Handle of the iterator.
    * @param  outputDataTypes Output data types of the iterator.
    * @param  outputShapes    Output shapes of the iterator.
    * @param  name            Name for the created op.
    * @return Created op outputs, which correspond to the iterator outputs.
    */
  private[api] def iteratorGetNext(
      iterator: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "IteratorGetNext"): Seq[Output] = {
    Op.Builder(opType = "IteratorGetNext", name = name)
        .addInput(iterator)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs.toSeq
  }

  /** Creates an op that releases any resources used by the provided iterator.
    *
    * @param  iterator Handle of the iterator.
    * @param  name     Name for the created op.
    * @return Created op.
    */
  private[api] def iteratorDispose(iterator: Output, name: String = "IteratorDispose"): Op = {
    Op.Builder(opType = "IteratorDispose", name = name)
        .addInput(iterator)
        .build()
  }

  /** Creates an op that converts the provided resource handle representing an iterator to a string.
    *
    * @param  iterator Handle of the iterator.
    * @param  name     Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the string handle.
    */
  private[api] def iteratorToStringHandle(iterator: Output, name: String = "IteratorToStringHandle"): Output = {
    Op.Builder(opType = "IteratorToStringHandle", name = name)
        .addInput(iterator)
        .build().outputs(0)
  }

  /** Creates an op that converts the provided string representing a handle to an iterator to the corresponding iterator
    * handle.
    *
    * @param  stringHandle `STRING` scalar tensor containing the string representation of a handle of an iterator.
    * @param  name          Name for the created op.
    * @return Created op output, which is a `RESOURCE` scalar tensor containing the iterator handle.
    */
  private[api] def iteratorFromStringHandle(
      stringHandle: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "IteratorFromStringHandle"): Output = {
    Op.Builder(opType = "IteratorFromStringHandle", name = name)
        .addInput(stringHandle)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  //endregion Iterator Ops

  private[api] object Gradients {
    //region General IO Ops

    GradientsRegistry.registerNonDifferentiable("ReadFile")
    GradientsRegistry.registerNonDifferentiable("WriteFile")
    GradientsRegistry.registerNonDifferentiable("MatchingFiles")

    //endregion General IO Ops

    //region Reader Ops

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

    //endregion Reader Ops

    //region Dataset Ops

    GradientsRegistry.registerNonDifferentiable("TensorDataset")
    GradientsRegistry.registerNonDifferentiable("TensorSliceDataset")
    GradientsRegistry.registerNonDifferentiable("RangeDataset")
    GradientsRegistry.registerNonDifferentiable("SparseTensorSliceDataset")
    GradientsRegistry.registerNonDifferentiable("TextLineDataset")
    GradientsRegistry.registerNonDifferentiable("FixedLengthRecordDataset")
    GradientsRegistry.registerNonDifferentiable("TFRecordDataset")
    GradientsRegistry.registerNonDifferentiable("ZipDataset")
    GradientsRegistry.registerNonDifferentiable("ConcatenateDataset")
    GradientsRegistry.registerNonDifferentiable("RepeatDataset")
    GradientsRegistry.registerNonDifferentiable("TakeDataset")
    GradientsRegistry.registerNonDifferentiable("SkipDataset")
    GradientsRegistry.registerNonDifferentiable("BatchDataset")
    GradientsRegistry.registerNonDifferentiable("PaddedBatchDataset")
    GradientsRegistry.registerNonDifferentiable("DenseToSparseBatchDataset")
    GradientsRegistry.registerNonDifferentiable("ShuffleDataset")
    GradientsRegistry.registerNonDifferentiable("CacheDataset")
    GradientsRegistry.registerNonDifferentiable("IgnoreErrorsDataset")
    GradientsRegistry.registerNonDifferentiable("MapDataset")
    GradientsRegistry.registerNonDifferentiable("ParallelMapDataset")
    GradientsRegistry.registerNonDifferentiable("FlatMapDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")
    GradientsRegistry.registerNonDifferentiable("GroupByWindowDataset")
    GradientsRegistry.registerNonDifferentiable("FilterDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")

    //endregion Dataset Ops

    //region Iterator Ops

    GradientsRegistry.registerNonDifferentiable("Iterator")
    GradientsRegistry.registerNonDifferentiable("MakeIterator")
    GradientsRegistry.registerNonDifferentiable("OneShotIterator")
    GradientsRegistry.registerNonDifferentiable("IteratorGetNext")
    GradientsRegistry.registerNonDifferentiable("IteratorDispose")
    GradientsRegistry.registerNonDifferentiable("IteratorToStringHandle")
    GradientsRegistry.registerNonDifferentiable("IteratorFromStringHandle")

    //endregion Iterator Ops
  }
}
