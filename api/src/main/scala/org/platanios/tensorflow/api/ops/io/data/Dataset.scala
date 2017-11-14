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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.ops.{Callback, Function, Op, Output}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.ops.io.data
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT64, STRING}

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable
import scala.language.postfixOps

/** Represents a potentially large set of elements.
  *
  * A [[Dataset]] can be used to represent an input pipeline as a collection of elements (i.e., nested structures of
  * tensors) and a "logical plan" of transformations that act on those elements.
  *
  * @param  name Name for this dataset.
  * @tparam T    Tensor type (i.e., nested structure of tensors).
  * @tparam O    Output type (i.e., nested structure of symbolic tensors).
  * @tparam D    Data type of the outputs (i.e., nested structure of TensorFlow data types).
  * @tparam S    Shape type of the outputs (i.e., nested structure of TensorFlow shapes).
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Dataset[T, O, D, S] private[io](val name: String = "Dataset")(implicit ev: Data.Aux[T, O, D, S]) {
  /** Creates a `RESOURCE` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  def createHandle(): Output

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
      sharedName: String = "", name: String = "InitializableIterator"): InitializableIterator[T, O, D, S] = {
    Iterator.fromDataset(this, sharedName, name)
  }

  // TODO: [DATASETS] "createOneShotIterator".

  /** Returns the data types corresponding to each element of this dataset, matching the structure of the elements. */
  def outputDataTypes: D

  /** Returns the shapes corresponding to each element of this dataset, matching the structure of the elements. */
  def outputShapes: S

  /** Returns a sequence of [[DataType]]s that correspond to the flattened data types of the nested [[Output]] structure
    * of the elements of this dataset. */
  private[io] def flattenedOutputDataTypes: Seq[DataType] = ev.flattenedDataTypes(outputDataTypes)

  /** Returns a sequence of [[Shape]]s that correspond to the flattened shapes of the nested [[Output]] structure of the
    * elements of this dataset. */
  private[io] def flattenedOutputShapes: Seq[Shape] = ev.flattenedShapes(outputShapes)

  override def toString: String = {
    "Dataset[" +
        s"outputDataTypes = ${ev.dataTypesToString(outputDataTypes)}, " +
        s"outputShapes = ${ev.shapesToString(outputShapes)}" +
        "]"
  }
}

object Dataset {
  private[io] trait API {
    type Dataset[T, O, D, S] = data.Dataset[T, O, D, S]

    type RangeDataset = data.RangeDataset
    type TensorDataset[T, O, D, S] = data.TensorDataset[T, O, D, S]
    type OutputDataset[T, O, D, S] = data.OutputDataset[T, O, D, S]
    type TensorSlicesDataset[T, O, D, S] = data.TensorSlicesDataset[T, O, D, S]
    type OutputSlicesDataset[T, O, D, S] = data.OutputSlicesDataset[T, O, D, S]
    type SparseTensorSlicesDataset = data.SparseTensorSlicesDataset
    type SparseOutputSlicesDataset = data.SparseOutputSlicesDataset
    type TextLineDataset = data.TextLineDataset

    type BatchDataset[T, O, D, S] = data.BatchDataset[T, O, D, S]
    type PrefetchDataset[T, O, D, S] = data.PrefetchDataset[T, O, D, S]
    type CacheDataset[T, O, D, S] = data.CacheDataset[T, O, D, S]
    type ShuffleDataset[T, O, D, S] = data.ShuffleDataset[T, O, D, S]
    type RepeatDataset[T, O, D, S] = data.RepeatDataset[T, O, D, S]
    type IgnoreErrorsDataset[T, O, D, S] = data.IgnoreErrorsDataset[T, O, D, S]

    type TakeDataset[T, O, D, S] = data.TakeDataset[T, O, D, S]
    type DropDataset[T, O, D, S] = data.DropDataset[T, O, D, S]

    type MapDataset[T, O, D, S, RT, RO, RD, RS] = data.MapDataset[T, O, D, S, RT, RO, RD, RS]
    type FlatMapDataset[T, O, D, S, RT, RO, RD, RS] = data.FlatMapDataset[T, O, D, S, RT, RO, RD, RS]

    type ZipDataset[T1, O1, D1, S1, T2, O2, D2, S2] = data.ZipDataset[T1, O1, D1, S1, T2, O2, D2, S2]
    type Zip3Dataset[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3] = data.Zip3Dataset[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3]
    type ZipMultipleDataset[T, O, D, S] = data.ZipMultipleDataset[T, O, D, S]

    type ConcatenatedDataset[T, O, D, S] = data.ConcatenatedDataset[T, O, D, S]

    val RangeDataset             : data.RangeDataset.type              = data.RangeDataset
    val TensorDataset            : data.TensorDataset.type             = data.TensorDataset
    val OutputDataset            : data.OutputDataset.type             = data.OutputDataset
    val TensorSlicesDataset      : data.TensorSlicesDataset.type       = data.TensorSlicesDataset
    val OutputSlicesDataset      : data.OutputSlicesDataset.type       = data.OutputSlicesDataset
    val SparseTensorSlicesDataset: data.SparseTensorSlicesDataset.type = data.SparseTensorSlicesDataset
    val SparseOutputSlicesDataset: data.SparseOutputSlicesDataset.type = data.SparseOutputSlicesDataset
    val TextLineDataset          : data.TextLineDataset.type           = data.TextLineDataset

    val BatchDataset       : data.BatchDataset.type        = data.BatchDataset
    val PrefetchDataset    : data.PrefetchDataset.type     = data.PrefetchDataset
    val CacheDataset       : data.CacheDataset.type        = data.CacheDataset
    val ShuffleDataset     : data.ShuffleDataset.type      = data.ShuffleDataset
    val RepeatDataset      : data.RepeatDataset.type       = data.RepeatDataset
    val IgnoreErrorsDataset: data.IgnoreErrorsDataset.type = data.IgnoreErrorsDataset

    val TakeDataset: data.TakeDataset.type = data.TakeDataset
    val DropDataset: data.DropDataset.type = data.DropDataset

    val MapDataset    : data.MapDataset.type     = data.MapDataset
    val FlatMapDataset: data.FlatMapDataset.type = data.FlatMapDataset

    val ZipDataset        : data.ZipDataset.type         = data.ZipDataset
    val Zip3Dataset       : data.Zip3Dataset.type        = data.Zip3Dataset
    val ZipMultipleDataset: data.ZipMultipleDataset.type = data.ZipMultipleDataset

    val ConcatenatedDataset: data.ConcatenatedDataset.type = data.ConcatenatedDataset

    def fromGenerator[T, O, D, S](
        generator: () => Iterable[T], outputDataType: D, outputShape: S = null
    )(implicit
        ev: Data.Aux[T, O, D, S],
        evFunctionOutput: Function.ArgType[O]
    ): Dataset[T, O, D, S] = {
      Dataset.fromGenerator[T, O, D, S](generator, outputDataType, outputShape)(ev, evFunctionOutput)
    }
  }

  /** Stores outstanding iterators created from a Scala iterable.
    *
    * This class keeps track of potentially multiple iterators that may have been created from an iterable, e.g., in the
    * case that the dataset is repeated, or nested within a parallel computation.
    *
    * @param  generator Function that generates an iterable containing dataset elements.
    */
  private[this] case class GeneratorState[T, O, D, S](generator: () => Iterable[T])(implicit ev: Data.Aux[T, O, D, S]) {
    private[this] val _nextId   = new AtomicLong(0)
    private[this] val iterators = mutable.Map.empty[Long, scala.Iterator[T]]

    private[Dataset] def nextId: Long = _nextId.getAndIncrement()
    private[Dataset] def getIterator(id: Long): scala.Iterator[T] = iterators.getOrElseUpdate(id, generator().iterator)
    private[Dataset] def deleteIterator(id: Long): Unit = iterators.remove(id)
  }

  /** Creates a [[Dataset]] whose elements are generated by Scala iterables.
    *
    * The `generator` argument must be a function that takes no arguments and returns an [[Iterable]] over dataset
    * elements. The elements contained in that [[Iterable]] must be compatible with the provided `outputDataType` and
    * `outputShape` arguments.
    *
    * For example:
    * {{{
    *   // TODO: !!! Improve this example with variable shapes -- like in the Python API.
    *   val generator = () => Range(0, 10)
    *   val dataset = Dataset.fromGenerator(generator, INT32, Shape.scalar())
    *   val value = dataset.createOneShotIterator().next()
    *   session.run(value) ==> 0
    *   session.run(value) ==> 1
    * }}}
    *
    * @param  generator      Function that takes no arguments and returns an [[Iterable]] over dataset elements.
    * @param  outputDataType Output data type structure for the tensor structure of the generated [[Iterable]] elements.
    * @param  outputShape    Output shape structure for the tensor structure of the generated [[Iterable]] elements.
    * @return Constructed dataset.
    */
  private[api] def fromGenerator[T, O, D, S](
      generator: () => Iterable[T], outputDataType: D, outputShape: S = null)(implicit
      ev: Data.Aux[T, O, D, S],
      evFunctionOutput: Function.ArgType[O]
  ): Dataset[T, O, D, S] = {
    val inferredOutputShape: S = {
      if (outputShape != null)
        outputShape
      else
        ev.unflattenShapes(outputDataType, Seq.fill(ev.size(outputDataType))(Shape.unknown()))
    }

    val flattenedTypes = ev.flattenedDataTypes(outputDataType)
    val flattenedShapes = ev.flattenedShapes(inferredOutputShape)
    val generatorState = GeneratorState(generator)(ev)

    /** Creates an op that generates the next element from iterator with ID, `iteratorId`.
      *
      * We map this function across an infinite repetition of the `iteratorId`, and throw an `OutOfRange` to terminate
      * the iteration.
      *
      * @param  iteratorId [[INT64]] scalar tensor whose value uniquely identifies the iterator in the internal
      *                    generator state, from which to generate an element.
      * @return Created op outputs structured according to the output data type of this dataset.
      */
    def generatorMapFn(iteratorId: Output): O = {
      /** Scala callback function that will be called to invoke the iterator. */
      @throws[OutOfRangeException]
      def generatorScalaCallback(iteratorId: Tensor): Seq[Tensor] = {
        val iterator = generatorState.getIterator(iteratorId.scalar.asInstanceOf[Long])
        val value = {
          if (iterator.hasNext)
            iterator.next()
          else
            throw OutOfRangeException("The iterator does not contain any more elements.")
        }
        val flattenedTensors = ev.flattenedTensors(value)
        // Additional type and shape checking to ensure that the components of the generated element match the
        // output data types and output shapes arguments.
        (flattenedTensors, flattenedTypes, flattenedShapes).zipped.foreach((tensor, dataType, shape) => {
          if (tensor.dataType != dataType)
            throw InvalidDataTypeException(
              s"The generator yielded an element of type ${tensor.dataType} " +
                  s"where an element of type $dataType was expected.")
          if (tensor.shape != shape)
            throw InvalidShapeException(
              s"The generator yielded an element with shape ${tensor.shape} " +
                  s"where an element with shape $shape was expected.")
        })
        flattenedTensors
      }

      val flattenedValues = Callback.callback(generatorScalaCallback, iteratorId, flattenedTypes, stateful = true)
      // The Scala callback op drops the inferred shapes, so we add them back in here.
      if (outputShape != null) {
        flattenedValues.zip(flattenedShapes).foreach(p => {
          if (!p._1.shape.isCompatibleWith(p._2)) {
            throw new IllegalArgumentException(
              s"Generator output shape ${p._1.shape} is not compatible with provided shape ${p._2}.")
          } else {
            p._1.setShape(p._2)
          }
        })
      }
      ev.unflattenOutputs(outputDataType, flattenedValues)
    }

    /** Associates each traversal of the provided `generator` with a unique iterator ID. */
    def flatMapFn(iteratorId: Output): Dataset[T, O, D, S] = {
      // First, generate an infinite dataset containing the iterator ID repeated forever. Then, map using the
      // `generatorMapFn`, which gets the next element from the iterator with the relevant ID, and throws an
      // IndexOutOfBoundsException when that iterator contains no more elements.
      OutputDataset(iteratorId).repeat().map(generatorMapFn)
    }

    // A single-element dataset that, each time it is evaluated, contains a freshly-generated and unique (for the
    // returned dataset) INT64 ID that will be used to identify the appropriate Scala state, which is encapsulated in
    // the internal generator state, and captured in the provided callback function. The ID disambiguates between
    // multiple concurrently existing iterators.
    val idDataset = TensorDataset(Tensor(INT64, 0)).map(
      (_: Output) => Callback.callback((_: Unit) => Tensor(INT64, generatorState.nextId), (), INT64, stateful = true))

    // A dataset that contains all of the elements generated by a single iterator created from the provided generator,
    // identified by the iterator ID contained in `idDataset`. Lifting the iteration into a `flatMap` here enables
    // multiple repetitions and/or nested versions of the returned dataset to be created, because it forces the
    // generation of a new ID for each version.
    idDataset.flatMap(flatMapFn)
  }

  /** Creates a TensorFlow records dataset op.
    *
    * A TensorFlow records dataset emits the records from one or more TFRecord files.
    *
    * @param  filenames       [[STRING]] scalar or vector tensor containing the the name(s) of the file(s) to be read.
    * @param  compressionType [[STRING]] scalar tensor containing the type of compression for the file. Currently ZLIB
    *                         and GZIP are supported. Defaults to `""`, meaning no compression.
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

  /** Creates an op representing a dataset that batches and pads `batchSize` elements from `dataset`.
    *
    * @param  datasetHandle Handle of the dataset to batch elements from.
    * @param  batchSize     [[INT64]] scalar tensor containing the batch size to use.
    * @param  paddedShapes  Sequence of [[INT64]] rank-1 tensors (i.e., vectors) representing the desired padded shapes
    *                       of the corresponding output components. These shapes may be partially specified, using `-1`
    *                       to indicate that a particular dimension should be padded to the maximum size of all batch
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

  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("TensorDataset")
    GradientsRegistry.registerNonDifferentiable("TensorSliceDataset")
    GradientsRegistry.registerNonDifferentiable("SparseTensorSliceDataset")
    GradientsRegistry.registerNonDifferentiable("RangeDataset")
    GradientsRegistry.registerNonDifferentiable("TextLineDataset")
    GradientsRegistry.registerNonDifferentiable("FixedLengthRecordDataset")
    GradientsRegistry.registerNonDifferentiable("TFRecordDataset")
    GradientsRegistry.registerNonDifferentiable("BatchDataset")
    GradientsRegistry.registerNonDifferentiable("PaddedBatchDataset")
    GradientsRegistry.registerNonDifferentiable("RepeatDataset")
    GradientsRegistry.registerNonDifferentiable("CacheDataset")
    GradientsRegistry.registerNonDifferentiable("ShuffleDataset")
    GradientsRegistry.registerNonDifferentiable("TakeDataset")
    GradientsRegistry.registerNonDifferentiable("SkipDataset")
    GradientsRegistry.registerNonDifferentiable("ZipDataset")
    GradientsRegistry.registerNonDifferentiable("ConcatenateDataset")
    GradientsRegistry.registerNonDifferentiable("MapDataset")
    GradientsRegistry.registerNonDifferentiable("ParallelMapDataset")
    GradientsRegistry.registerNonDifferentiable("FlatMapDataset")
    GradientsRegistry.registerNonDifferentiable("FilterDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")
    GradientsRegistry.registerNonDifferentiable("GroupByWindowDataset")
    GradientsRegistry.registerNonDifferentiable("PrefetchDataset")
    GradientsRegistry.registerNonDifferentiable("IgnoreErrorsDataset")
    GradientsRegistry.registerNonDifferentiable("DenseToSparseBatchDataset")
  }
}
