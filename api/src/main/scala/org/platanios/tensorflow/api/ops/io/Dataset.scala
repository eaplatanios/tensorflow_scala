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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.Indexer.Implicits._
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.{DataType, INT64, STRING}

import scala.language.postfixOps

/** Represents a potentially large set of elements.
  *
  * A [[Dataset]] can be used to represent an input pipeline as a collection of elements (i.e., nested structures of
  * tensors) and a "logical plan" of transformations that act on those elements.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Dataset[T, D, S] private[io](implicit ev: Data.Aux[T, D, S]) {
  /** Creates a `RESOURCE` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  def createResource(): Output

  /** Creates an [[Iterator]] for enumerating the elements of this dataset.
    *
    * **Note:** The returned iterator will be in an uninitialized state. You must execute the
    * [[InitializableIterator.initializer]] op before using it.
    *
    * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
    *                    multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name       Name for the op created in relation to the iterator.
    * @return Created iterator.
    */
  def createInitializableIterator(
      sharedName: String = "", name: String = "InitializableIterator"): InitializableIterator[T, D, S] = ???

  def outputDataTypes: D
  def outputShapes: S

  /** Returns a sequence of [[DataType]]s that correspond to the flattened data types of the nested [[Output]] structure
    * of the elements of this dataset. */
  private[io] def flattenedOutputDataTypes: Seq[DataType]

  /** Returns a sequence of [[Shape]]s that correspond to the flattened shapes of the nested [[Output]] structure of the
    * elements of this dataset. */
  private[io] def flattenedOutputShapes: Seq[Shape]
}

object Dataset {
  /** Creates a tensor dataset op.
    *
    * A tensor dataset is a dataset that emits `components` as a tuple of tensors once.
    *
    * @param  components Tensors to emit.
    * @param  shapes     Shapes of the emitted tensors.
    * @param  name       Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[io] def createTensorDataset(
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
  private[io] def createTensorSliceDataset(
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
  private[io] def createRangeDataset(
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
  private[io] def createSparseTensorSliceDataset(
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
  private[io] def createTextLineDataset(
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
  private[io] def createFixedLengthRecordDataset(
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
  private[io] def createTFRecordDataset(
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
  private[io] def datasetZip(
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
  private[io] def datasetConcatenate(
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
    * @param  datasetHandle   Handle of the dataset to repeat.
    * @param  count           `INT64` scalar tensor containing the number of times to repeat the provided dataset. A
    *                         value of `-1` corresponds to repeating it indefinitely.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `count` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetRepeat(
      datasetHandle: Output, count: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetRepeat"): Output = {
    if (count.dataType != INT64)
      throw new IllegalArgumentException(s"'count' (dataType = ${count.dataType}) must be an INT64 tensor.")
    if (count.rank != -1 && count.rank > 0)
      throw new IllegalArgumentException(s"'count' (rank = ${count.rank}) must be equal to 0.")
    Op.Builder(opType = "RepeatDataset", name = name)
        .addInput(datasetHandle)
        .addInput(count)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains `count` entries from the provided dataset.
    *
    * @param  datasetHandle   Handle of the dataset to take entries from.
    * @param  count           `INT64` scalar tensor containing the number of entries to take from the provided dataset.
    *                         A value of `-1` corresponds to taking all the entries.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `count` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetTake(
      datasetHandle: Output, count: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetTake"): Output = {
    if (count.dataType != INT64)
      throw new IllegalArgumentException(s"'count' (dataType = ${count.dataType}) must be an INT64 tensor.")
    if (count.rank != -1 && count.rank > 0)
      throw new IllegalArgumentException(s"'count' (rank = ${count.rank}) must be equal to 0.")
    Op.Builder(opType = "TakeDataset", name = name)
        .addInput(datasetHandle)
        .addInput(count)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains all entries from the provided dataset except the first `count`.
    *
    * @param  datasetHandle   Handle of the dataset to skip entries from.
    * @param  count           `INT64` scalar tensor containing the number of entries to skip from the provided dataset.
    *                         A value of `-1` corresponds to skipping all the entries.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `count` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetSkip(
      datasetHandle: Output, count: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetSkip"): Output = {
    if (count.dataType != INT64)
      throw new IllegalArgumentException(s"'count' (dataType = ${count.dataType}) must be an INT64 tensor.")
    if (count.rank != -1 && count.rank > 0)
      throw new IllegalArgumentException(s"'count' (rank = ${count.rank}) must be equal to 0.")
    Op.Builder(opType = "SkipDataset", name = name)
        .addInput(datasetHandle)
        .addInput(count)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that batches `batchSize` elements from `dataset`.
    *
    * @param  datasetHandle   Handle of the dataset to batch elements from.
    * @param  batchSize       `INT64` scalar tensor containing the batch size to use.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `batchSize` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetBatch(
      datasetHandle: Output, batchSize: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetBatch"): Output = {
    if (batchSize.dataType != INT64)
      throw new IllegalArgumentException(s"'batchSize' (dataType = ${batchSize.dataType}) must be an INT64 tensor.")
    if (batchSize.rank != -1 && batchSize.rank > 0)
      throw new IllegalArgumentException(s"'batchSize' (rank = ${batchSize.rank}) must be equal to 0.")
    Op.Builder(opType = "BatchDataset", name = name)
        .addInput(datasetHandle)
        .addInput(batchSize)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that batches and pads `batchSize` elements from `dataset`.
    *
    * @param  datasetHandle Handle of the dataset to batch elements from.
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
  private[io] def datasetPaddedBatch(
      datasetHandle: Output, batchSize: Output, paddedShapes: Seq[Output], paddingValues: Seq[Output],
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
        .addInput(datasetHandle)
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
    * @param  datasetHandle   Handle of the dataset to batch elements from.
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
  private[io] def datasetShuffle(
      datasetHandle: Output, batchSize: Output, seed1: Output, seed2: Output, outputDataTypes: Seq[DataType],
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
        .addInput(datasetHandle)
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
    * @param  datasetHandle   Handle of the dataset to cache.
    * @param  filename        `STRING` scalar tensor containing a path in the filesystem where the dataset should be
    *                         cached. Note that this should be a directory and not a file.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `filename` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetCache(
      datasetHandle: Output, filename: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetCache"): Output = {
    if (filename.dataType != STRING)
      throw new IllegalArgumentException(s"'filename' (dataType = ${filename.dataType}) must be a STRING tensor.")
    if (filename.rank != -1 && filename.rank > 0)
      throw new IllegalArgumentException(s"'filename' (rank = ${filename.rank}) must be equal to 0.")
    Op.Builder(opType = "CacheDataset", name = name)
        .addInput(datasetHandle)
        .addInput(filename)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that asynchronously prefetches elements from `dataset`.
    *
    * A cached dataset will iterate over the input dataset and store the tensors it gets. If the cache already exists,
    * it will be used. If the cache is inappropriate (e.g., it cannot be opened, or contains tensors of the wrong shape
    * or size), an error will the returned when used.
    *
    * @param  datasetHandle   Handle of the dataset to cache.
    * @param  bufferSize      `INT64` scalar tensor containing the maximum number of elements to buffer in an iterator
    *                         over this dataset.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `bufferSize` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetPrefetch(
      datasetHandle: Output, bufferSize: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetPrefetch"): Output = {
    if (bufferSize.dataType != INT64)
      throw new IllegalArgumentException(s"'bufferSize' (dataType = ${bufferSize.dataType}) must be an INT64 tensor.")
    if (bufferSize.rank != -1 && bufferSize.rank != 0)
      throw new IllegalArgumentException(s"'bufferSize' (rank = ${bufferSize.rank}) must be equal to 0.")
    Op.Builder(opType = "PrefetchDataset", name = name)
        .addInput(datasetHandle)
        .addInput(bufferSize)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains all entries from the provided dataset, but ignores all errors.
    *
    * @param  datasetHandle   Handle of the dataset to take entries from.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[io] def datasetIgnoreErrors(
      datasetHandle: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetIgnoreErrors"): Output = {
    Op.Builder(opType = "IgnoreErrorsDataset", name = name)
        .addInput(datasetHandle)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  // TODO: [DATASETS] [FUNCTIONS] "map", "parallelMap", "flatMap", "interleave", "groupByWindow", and "filter".

  private[io] object Gradients {
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
  }
}
