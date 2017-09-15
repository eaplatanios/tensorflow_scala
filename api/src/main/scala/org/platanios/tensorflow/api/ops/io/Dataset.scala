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

import org.platanios.tensorflow.api.Implicits._
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, InvalidShapeException}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor}
import org.platanios.tensorflow.api.types.{DataType, INT32, INT64, STRING}

import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable
import scala.language.postfixOps

/** Represents a potentially large set of elements.
  *
  * A [[Dataset]] can be used to represent an input pipeline as a collection of elements (i.e., nested structures of
  * tensors) and a "logical plan" of transformations that act on those elements.
  *
  * @param  name Name for this dataset.
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
  val outputDataTypes: D

  /** Returns the shapes corresponding to each element of this dataset, matching the structure of the elements. */
  val outputShapes: S

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

  /** $OpDocDatasetBatch
    *
    * @param  batchSize Batch size.
    * @return Created dataset.
    */
  def batch(batchSize: Long): Dataset[T, O, D, S] = {
    Dataset.batch(this, batchSize, s"$name/Batch")
  }

  /** $OpDocDatasetRepeat
    *
    * @param  count Number of times to repeat the input dataset. A value of `-1` corresponds to repeating it
    *               indefinitely.
    * @return Created dataset.
    */
  def repeat(count: Long = -1): Dataset[T, O, D, S] = {
    Dataset.repeat(this, count, s"$name/Repeat")
  }

  /** $OpDocDatasetCache
    *
    * @param  directory Directory to use for caching. If empty, then the provided dataset will be cached in memory.
    * @return Created dataset.
    */
  def cache(directory: String): Dataset[T, O, D, S] = {
    Dataset.cache(this, directory, s"$name/Cache")
  }

  /** $OpDocDatasetShuffle
    *
    * @param  bufferSize Buffer size, meaning the number of output elements to buffer in an iterator over this dataset.
    * @param  seed       Seed value for the random number generator. If not provided, a random seed is used.
    * @return Created dataset.
    */
  def shuffle(bufferSize: Long, seed: Option[Int] = None): Dataset[T, O, D, S] = {
    Dataset.shuffle(this, bufferSize, seed, s"$name/Shuffle")
  }

  /** $OpDocDatasetTake
    *
    * @param  count Number of elements to take.
    * @return Created dataset.
    */
  def take(count: Long): Dataset[T, O, D, S] = {
    Dataset.take(this, count, s"$name/Take")
  }

  /** $OpDocDatasetDrop
    *
    * @param  count Number of elements to drop.
    * @return Created dataset.
    */
  def drop(count: Long): Dataset[T, O, D, S] = {
    Dataset.drop(this, count, s"$name/Drop")
  }

  /** $OpDocDatasetZip
    *
    * @param  other Dataset to zip with the current dataset.
    * @return Created dataset.
    */
  def zip[T2, O2, D2, S2](other: Dataset[T2, O2, D2, S2])(implicit
      ev2: Data.Aux[T2, O2, D2, S2]
  ): Dataset[(T, T2), (O, O2), (D, D2), (S, S2)] = {
    Dataset.zip(this, other, s"${name}_${other.name}/Zip")
  }

  /** $OpDocDatasetZip
    *
    * @param  other1 First dataset to zip with the current dataset.
    * @param  other2 Second dataset to zip with the current dataset.
    * @return Created dataset.
    */
  def zip3[T2, O2, D2, S2, T3, O3, D3, S3](other1: Dataset[T2, O2, D2, S2], other2: Dataset[T3, O3, D3, S3])(implicit
      ev2: Data.Aux[T2, O2, D2, S2],
      ev3: Data.Aux[T3, O3, D3, S3]
  ): Dataset[(T, T2, T3), (O, O2, O3), (D, D2, D3), (S, S2, S3)] = {
    Dataset.zip3(this, other1, other2, s"${name}_${other1.name}_${other2.name}/Zip3")
  }

  /** $OpDocDatasetConcatenate
    *
    * @param  other Dataset to concatenate with the current dataset.
    * @return Created dataset.
    */
  def concatenate(other: Dataset[T, O, D, S]): Dataset[T, O, D, S] = {
    Dataset.concatenate(this, other, s"${name}_${other.name}/Concatenated")
  }

  /** $OpDocDatasetMap
    *
    * @param  function         Mapping function.
    * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
    *                          sequentially.
    * @param  bufferSize       Maximum number of processed elements that will be buffered.
    * @return Created dataset.
    */
  def map[RT, RO, RD, RS](function: (T) => RT, numParallelCalls: Int = 1, bufferSize: Long = 1)(implicit
      evR: Data.Aux[RT, RO, RD, RS],
      evFunctionInput: Function.ArgType[T],
      evFunctionOutput: Function.ArgType[RT]
  ): Dataset[RT, RO, RD, RS] = {
    Dataset.map(this, function, numParallelCalls, bufferSize, s"$name/Map")
  }

  /** $OpDocDatasetPrefetch
    *
    * @param  bufferSize Number of elements to prefetch.
    * @return Created dataset.
    */
  def prefetch(bufferSize: Long): Dataset[T, O, D, S] = {
    Dataset.prefetch(this, bufferSize, s"$name/Prefetch")
  }

  /** $OpDocDatasetIgnoreErrors
    *
    * @return Created dataset.
    */
  def ignoreErrors(): Dataset[T, O, D, S] = {
    Dataset.ignoreErrors(this, s"$name/IgnoreErrors")
  }
}

object Dataset {
  private[io] trait API {
    def datasetFrom[T, O, D, S](
        data: T, name: String = "TensorDataset")(implicit ev: Data.Aux[T, O, D, S]): Dataset[T, O, D, S] = {
      from(data, name)(ev)
    }

    def datasetFromSlices[T, O, D, S](
        data: T, name: String = "TensorSliceDataset")(implicit ev: Data.Aux[T, O, D, S]): Dataset[T, O, D, S] = {
      fromSlices(data, name)(ev)
    }

    /** Creates a [[Dataset]] that splits a sparse tensor into its rows.
      *
      * @param  tensor Sparse tensor.
      * @param  name   Name for the created dataset.
      * @return Created dataset.
      */
    private[api] def datasetFromSparseSlices(tensor: SparseTensor, name: String = "SparseTensorSliceDataset"):
    Dataset[SparseTensor, SparseOutput, (DataType, DataType, DataType), (Shape, Shape, Shape)] = {
      fromSparseSlices(tensor, name)
    }
  }

  private[api] def from[T, O, D, S](
      data: T, name: String = "TensorDataset")(implicit ev: Data.Aux[T, O, D, S]): Dataset[T, O, D, S] = {
    // TODO: !!! [DATASETS] What happens when one provides a structure with Tensor objects?
    TensorDataset(data, name = name)(ev)
  }

  private[api] def fromSlices[T, O, D, S](
      data: T, name: String = "TensorSliceDataset")(implicit ev: Data.Aux[T, O, D, S]): Dataset[T, O, D, S] = {
    // TODO: !!! [DATASETS] What happens when one provides a structure with Tensor objects?
    TensorSliceDataset(data, name = name)(ev)
  }

  /** Creates a [[Dataset]] that splits a sparse tensor into its rows.
    *
    * @param  tensor Sparse tensor.
    * @param  name   Name for the created dataset.
    * @return Created dataset.
    */
  private[api] def fromSparseSlices(tensor: SparseTensor, name: String = "SparseTensorSliceDataset"):
  Dataset[SparseTensor, SparseOutput, (DataType, DataType, DataType), (Shape, Shape, Shape)] = {
    SparseTensorSliceDataset(tensor, name)
  }

  /** $OpDocDatasetRange
    *
    * @param  start Starting value of the number sequence.
    * @param  limit Ending value (exclusive) of the number sequence.
    * @param  delta Difference between consecutive numbers in the sequence.
    * @param  name Name for the new dataset.
    * @return Constructed dataset.
    */
  private[api] def range(start: Long, limit: Long, delta: Long, name: String = "RangeDataset"):
  Dataset[Tensor, Output, DataType, Shape] = {
    RangeDataset(start, limit, delta, name = name)
  }

//  /** Stores outstanding iterators created from a Scala iterable.
//    *
//    * This class keeps track of potentially multiple iterators that may have been created from an iterable, e.g., in the
//    * case that the dataset is repeated, or nested within a parallel computation.
//    *
//    * @param  generator Function that generates an iterable containing dataset elements.
//    */
//  private[this] case class GeneratorState[T, O, D, S](generator: () => Iterable[T])(implicit ev: Data.Aux[T, O, D, S]) {
//    private[this] val _nextId   = new AtomicLong(0)
//    private[this] val iterators = mutable.Map.empty[Long, scala.Iterator[T]]
//
//    private[Dataset] def nextId: Long = _nextId.getAndIncrement()
//    private[Dataset] def getIterator(id: Long): scala.Iterator[T] = iterators.getOrElseUpdate(id, generator().iterator)
//    private[Dataset] def deleteIterator(id: Long): Unit = iterators.remove(id)
//  }
//
//  /** Creates a [[Dataset]] whose elements are generated by Scala iterables.
//    *
//    * The `generator` argument must be a function that takes no arguments and returns an [[Iterable]] over dataset
//    * elements. The elements contained in that [[Iterable]] must be compatible with the provided `outputDataType` and
//    * `outputShape` arguments.
//    *
//    * For example:
//    * {{{
//    *   // TODO: !!! Improve this example with variable shapes -- like in the Python API.
//    *   val generator = () => Range(0, 10)
//    *   val dataset = Dataset.fromGenerator(generator, INT32, Shape.scalar())
//    *   val value = dataset.createOneShotIterator().next()
//    *   session.run(value) ==> 0
//    *   session.run(value) ==> 1
//    * }}}
//    *
//    * @param  generator      Function that takes no arguments and returns an [[Iterable]] over dataset elements.
//    * @param  outputDataType Output data type structure for the tensor structure of the generated [[Iterable]] elements.
//    * @param  outputShape    Output shape structure for the tensor structure of the generated [[Iterable]] elements.
//    * @return Constructed dataset.
//    */
//  private[io] def fromGenerator[T, O, D, S](
//      generator: () => Iterable[T], outputDataType: D, outputShape: S = null)(implicit
//      ev: Data.Aux[T, O, D, S]
//  ): Dataset[O, D, S] = {
//    val inferredOutputShape: S = {
//      if (outputShape != null)
//        outputShape
//      else
//        ev.unflattenShapes(outputDataType, Seq.fill(ev.numberOfOutputs(outputDataType))(Shape.unknown()))
//    }
//    val flattenedTypes = ev.flattenedOutputDataTypes(outputDataType)
//    val flattenedShapes = ev.flattenedOutputShapes(inferredOutputShape)
//    val generatorState = GeneratorState(generator)(ev)
//
//    /** Creates an op that generates the next element from iterator with ID, `iterator_id`.
//      *
//      * We map this function across an infinite repetition of the `iterator_id`, and throw an
//      * `IndexOutOfBoundsException` to terminate the iteration.
//      *
//      * @param  iteratorId [[INT64]] scalar tensor whose value uniquely identifies the iterator in the internal
//      *                    generator state, from which to generate an element.
//      * @return Created op outputs structured according to the output data type of this dataset.
//      */
//    def generatorMapFn(iteratorId: Output): O = {
//      /** Scala callback function that will be called to invoke the iterator. */
//      @throws[IndexOutOfBoundsException]
//      def generatorScalaCallback(iteratorId: Tensor): Seq[Tensor] = {
//        val iterator = generatorState.getIterator(iteratorId.scalar.asInstanceOf[Long])
//        val values = {
//          if (iterator.hasNext)
//            iterator.next()
//          else
//            throw new IndexOutOfBoundsException("The iterator does not contain any more elements.")
//        }
//        val flattenedTensors = ev.flattenedTensors(values)
//        // Additional type and shape checking to ensure that the components of the generated element match the
//        // output data types and output shapes arguments.
//        for (
//          tensor <- flattenedTensors;
//          dataType <- flattenedTypes;
//          shape <- flattenedShapes) {
//          if (tensor.dataType != dataType)
//            throw InvalidDataTypeException(
//              s"The generator yielded an element of type ${tensor.dataType} " +
//                  s"where an element of type $dataType was expected.")
//          if (tensor.shape != shape)
//            throw InvalidShapeException(
//              s"The generator yielded an element with shape ${tensor.shape} " +
//                  s"where an element with shape $shape was expected.")
//        }
//        flattenedTensors
//      }
//      val flattenedValues = Callback.callback(generatorScalaCallback, iteratorId, flattenedTypes, stateful = true)
//      // The Scala callback op drops the inferred shapes, so we add them back in here.
//      if (outputShape != null)
//        flattenedValues.zip(flattenedShapes).foreach(p => p._1.setShape(p._2))
//      ev.unflattenOutputs(outputDataType, flattenedValues)
//    }
//
//    /** Associates each traversal of the provided `generator` with a unique iterator ID. */
//    def flatMapFn(iteratorId: Tensor): Dataset[O, D, S] = {
//      // First, generate an infinite dataset containing the iterator ID repeated forever. Then, map using the
//      // `generatorMapFn`, which gets the next element from the iterator with the relevant ID, and throws an
//      // IndexOutOfBoundsException when that iterator contains no more elements.
//      from(iteratorId).repeat().map(generatorMapFn)
//    }
//
//    // A single-element dataset that, each time it is evaluated, contains a freshly-generated and unique (for the
//    // returned dataset) INT64 ID that will be used to identify the appropriate Scala state, which is encapsulated in
//    // the internal generator state, and captured in the provided callback function. The ID disambiguates between
//    // multiple concurrently existing iterators.
//    val idDataset = Dataset.from(Tensor(INT64, 0)).map(
//      Callback.callback(_ => Tensor(INT64, generatorState.nextId), (), INT64, stateful = true))
//
//    // A dataset that contains all of the elements generated by a single iterator created from the provided generator,
//    // identified by the iterator ID contained in `idDataset`. Lifting the iteration into a `flatMap` here enables
//    // multiple repetitions and/or nested versions of the returned dataset to be created, because it forces the
//    // generation of a new ID for each version.
//    idDataset.flatMap(flatMapFn)
//  }

  /** $OpDocDatasetBatch
    *
    * @param  dataset   Input dataset.
    * @param  batchSize Batch size.
    * @param  name      Name for the created dataset.
    * @return Created dataset.
    */
  def batch[T, O, D, S](dataset: Dataset[T, O, D, S], batchSize: Long, name: String = "BatchDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    BatchDataset(dataset, batchSize, name)
  }

  /** $OpDocDatasetRepeat
    *
    * @param  dataset Input dataset.
    * @param  count   Number of times to repeat the input dataset. A value of `-1` corresponds to repeating it
    *                 indefinitely.
    * @param  name    Name for the created dataset.
    * @return Created dataset.
    */
  def repeat[T, O, D, S](dataset: Dataset[T, O, D, S], count: Long = -1, name: String = "RepeatDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    RepeatDataset(dataset, count, name)
  }

  /** $OpDocDatasetCache
    *
    * @param  dataset   Input dataset.
    * @param  directory Directory to use for caching. If empty, then the provided dataset will be cached in memory.
    * @param  name      Name for the created dataset.
    * @return Created dataset.
    */
  def cache[T, O, D, S](dataset: Dataset[T, O, D, S], directory: String, name: String = "CacheDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    CacheDataset(dataset, directory, name)
  }

  /** $OpDocDatasetShuffle
    *
    * @param  dataset    Input dataset.
    * @param  bufferSize Buffer size, meaning the number of output elements to buffer in an iterator over this dataset.
    * @param  seed       Seed value for the random number generator. If not provided, a random seed is used.
    * @param  name       Name for the created dataset.
    * @return Created dataset.
    */
  def shuffle[T, O, D, S](
      dataset: Dataset[T, O, D, S], bufferSize: Long, seed: Option[Int] = None,
      name: String = "ShuffleDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    ShuffleDataset(dataset, bufferSize, seed, name)
  }

  /** $OpDocDatasetTake
    *
    * @param  dataset Input dataset.
    * @param  count   Number of elements to take.
    * @param  name    Name for the created dataset.
    * @return Created dataset.
    */
  def take[T, O, D, S](dataset: Dataset[T, O, D, S], count: Long, name: String = "TakeDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    TakeDataset(dataset, count, name)
  }

  /** $OpDocDatasetDrop
    *
    * @param  dataset Input dataset.
    * @param  count   Number of elements to drop.
    * @param  name    Name for the created dataset.
    * @return Created dataset.
    */
  def drop[T, O, D, S](dataset: Dataset[T, O, D, S], count: Long, name: String = "DropDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    DropDataset(dataset, count, name)
  }

  /** $OpDocDatasetZip
    *
    * @param  dataset1 First dataset to zip.
    * @param  dataset2 Second dataset to zip.
    * @param  name     Name for the created dataset.
    * @return Created dataset.
    */
  def zip[T1, O1, D1, S1, T2, O2, D2, S2](
      dataset1: Dataset[T1, O1, D1, S1], dataset2: Dataset[T2, O2, D2, S2], name: String = "ZipDataset")(implicit
      ev1: Data.Aux[T1, O1, D1, S1],
      ev2: Data.Aux[T2, O2, D2, S2]
  ): Dataset[(T1, T2), (O1, O2), (D1, D2), (S1, S2)] = {
    ZipDataset(dataset1, dataset2, name)
  }

  /** $OpDocDatasetZip
    *
    * @param  dataset1 First dataset to zip.
    * @param  dataset2 Second dataset to zip.
    * @param  dataset3 Third dataset to zip.
    * @param  name     Name for the created dataset.
    * @return Created dataset.
    */
  def zip3[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3](
      dataset1: Dataset[T1, O1, D1, S1], dataset2: Dataset[T2, O2, D2, S2], dataset3: Dataset[T3, O3, D3, S3],
      name: String = "Zip3Dataset")(implicit
      ev1: Data.Aux[T1, O1, D1, S1],
      ev2: Data.Aux[T2, O2, D2, S2],
      ev3: Data.Aux[T3, O3, D3, S3]
  ): Dataset[(T1, T2, T3), (O1, O2, O3), (D1, D2, D3), (S1, S2, S3)] = {
    Zip3Dataset(dataset1, dataset2, dataset3, name)
  }

  /** $OpDocDatasetZip
    *
    * @param  datasets Datasets to zip.
    * @param  name     Name for the created dataset.
    * @return Created dataset.
    */
  def zipMultiple[T, O, D, S](datasets: Seq[Dataset[T, O, D, S]], name: String = "ZipMultipleDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[Seq[T], Seq[O], Seq[D], Seq[S]] = {
    ZipMultipleDataset(datasets, name)
  }

  /** $OpDocDatasetConcatenate
    *
    * @param  dataset1 First dataset to concatenate.
    * @param  dataset2 Second dataset to concatenate.
    * @param  name     Name for the created dataset.
    * @return Created dataset.
    */
  def concatenate[T, O, D, S](
      dataset1: Dataset[T, O, D, S], dataset2: Dataset[T, O, D, S], name: String = "ConcatenatedDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    ConcatenatedDataset(dataset1, dataset2, name)
  }

  /** $OpDocDatasetMap
    *
    * @param  dataset          Input dataset.
    * @param  function         Mapping function.
    * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
    *                          sequentially.
    * @param  bufferSize       Maximum number of processed elements that will be buffered.
    * @param  name             Name for the created dataset.
    * @return Created dataset.
    */
  def map[T, O, D, S, RT, RO, RD, RS](
      dataset: Dataset[T, O, D, S], function: (T) => RT, numParallelCalls: Int = 1, bufferSize: Long = 1,
      name: String = "MapDataset")(implicit
      ev: Data.Aux[T, O, D, S],
      evR: Data.Aux[RT, RO, RD, RS],
      evFunctionInput: Function.ArgType[T],
      evFunctionOutput: Function.ArgType[RT]
  ): Dataset[RT, RO, RD, RS] = {
    val mappedDataset: Dataset[RT, RO, RD, RS] = {
      if (numParallelCalls > 1)
        ParallelMapDataset(dataset, function, numParallelCalls, name)
      else
        MapDataset(dataset, function, name)
    }
    if (bufferSize > 1)
      mappedDataset.prefetch(bufferSize)
    else
      mappedDataset
  }

  /** $OpDocDatasetPrefetch
    *
    * @param  dataset    Input dataset.
    * @param  bufferSize Number of elements to prefetch.
    * @param  name       Name for the created dataset.
    * @return Created dataset.
    */
  def prefetch[T, O, D, S](dataset: Dataset[T, O, D, S], bufferSize: Long, name: String = "PrefetchDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): Dataset[T, O, D, S] = {
    PrefetchDataset(dataset, bufferSize, name)
  }

  /** $OpDocDatasetIgnoreErrors
    *
    * @param  dataset Input dataset.
    * @param  name    Name for the created dataset.
    * @return Created dataset.
    */
  def ignoreErrors[T, O, D, S](dataset: Dataset[T, O, D, S], name: String = "IgnoreErrorsDataset")(implicit
      ev: Data.Aux[T, O, D, S]
  ): IgnoreErrorsDataset[T, O, D, S] = {
    IgnoreErrorsDataset(dataset, name)
  }

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
    if (components.zip(shapes).exists(p => {
      (p._1.shape.rank > 1 && !p._1.shape(1 ::).isCompatibleWith(p._2)) ||
          (p._1.shape.rank == 0 && !Shape().isCompatibleWith(p._2))
    }))
      throw new IllegalArgumentException(
        "The axis-0 slice of each tensor in 'components' " +
            "must have shape compatible with the corresponding shape in 'shapes'.")
    Op.Builder(opType = "TensorSliceDataset", name = name)
        .addInputList(components)
        .setAttribute("output_shapes", shapes.toArray)
        .build().outputs(0)
  }

  /** Creates a sparse tensor slice dataset op.
    *
    * A tensor slice dataset is a dataset that that splits a sparse tensor into elements row-wise and emits each such
    * element once.
    *
    * @param  indices    [[INT64]] tensor containing the indices of the non-zero elements of the tensor.
    * @param  values     Tensor containing the values of the tensor corresponding to `indices`.
    * @param  denseShape [[INT64]] tensor containing the full/dense shape of the tensor.
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

  /** Creates a range dataset op.
    *
    * A range dataset is a dataset that contains a range of values.
    *
    * @param  start           [[INT64]] tensor containing the start value for the range.
    * @param  stop            [[INT64]] tensor containing the stop value for the range.
    * @param  step            [[INT64]] tensor containing the step value for the range.
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

  /** Creates a text-line dataset op.
    *
    * A text-line dataset emits the lines of one or more text files.
    *
    * **Note:** New-line characters are stripped from the output.
    *
    * @param  filenames       [[STRING]] scalar or vector tensor containing the the name(s) of the file(s) to be read.
    * @param  compressionType [[STRING]] scalar tensor containing the type of compression for the file. Currently ZLIB
    *                         and GZIP are supported. Defaults to `""`, meaning no compression.
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
    * @param  filenames   [[STRING]] scalar or vector tensor containing the the name(s) of the file(s) to be read.
    * @param  recordBytes [[INT64]] scalar tensor containing the number of bytes in the record.
    * @param  headerBytes [[INT64]] scalar tensor containing the number of bytes in the header (i.e., the number of
    *                     bytes to skip at the beginning of a file).
    * @param  footerBytes [[INT64]] scalar tensor containing the number of bytes in the footer (i.e., the number of
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

  /** Creates an op representing a dataset that batches `batchSize` elements from `dataset`.
    *
    * @param  datasetHandle   Handle of the dataset to batch elements from.
    * @param  batchSize       [[INT64]] scalar tensor containing the batch size to use.
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

  /** Creates an op that repeats a dataset.
    *
    * A repeated dataset is a dataset that emits the outputs of another dataset a number of times.
    *
    * @param  datasetHandle   Handle of the dataset to repeat.
    * @param  count           [[INT64]] scalar tensor containing the number of times to repeat the provided dataset. A
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

  /** Creates an op representing a dataset that caches elements from `dataset`.
    *
    * A cache dataset will iterate over the input dataset and store tensors. If the cache already exists, then it will
    * be used. If the cache is inappropriate (e.g., cannot be opened or contains tensors of the wrong shape / size),
    * then an error will the returned when used.
    *
    * @param  datasetHandle   Handle of the dataset to cache.
    * @param  directory       [[STRING]] scalar tensor containing the directory to use for caching. If empty, then the
    *                         dataset is cached in memory.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If `directory` has invalid data type or rank (i.e., if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetCache(
      datasetHandle: Output, directory: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetCache"): Output = {
    if (directory.dataType != STRING)
      throw new IllegalArgumentException(s"'directory' (dataType = ${directory.dataType}) must be an STRING tensor.")
    if (directory.rank != -1 && directory.rank > 0)
      throw new IllegalArgumentException(s"'directory' (rank = ${directory.rank}) must be equal to 0.")
    Op.Builder(opType = "CacheDataset", name = name)
        .addInput(datasetHandle)
        .addInput(directory)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that shuffles elements from `dataset` pseudo-randomly.
    *
    * @param  datasetHandle   Handle of the dataset to batch elements from.
    * @param  bufferSize      [[INT64]] scalar tensor containing the buffer size, meaning the number of output elements
    *                         to buffer in an iterator over the created dataset.
    * @param  seed1           [[INT64]] scalar tensor containing a seed value for the random number generator. If either
    *                         seed or seed2 is set to be non-zero, the random number generator is seeded by the given
    *                         seed. Otherwise, a random seed is used.
    * @param  seed2           [[INT64]] scalar tensor containing a second seed value to avoid seed collision.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    * @throws IllegalArgumentException If any of `batchSize`, `seed1`, or `seed2` has invalid data type or rank (i.e.,
    *                                  if it not a scalar).
    */
  @throws[IllegalArgumentException]
  private[io] def datasetShuffle(
      datasetHandle: Output, bufferSize: Output, seed1: Output, seed2: Output, outputDataTypes: Seq[DataType],
      outputShapes: Seq[Shape], name: String = "DatasetShuffle"): Output = {
    if (bufferSize.dataType != INT64)
      throw new IllegalArgumentException(s"'batchSize' (dataType = ${bufferSize.dataType}) must be an INT64 tensor.")
    if (bufferSize.rank != -1 && bufferSize.rank > 0)
      throw new IllegalArgumentException(s"'batchSize' (rank = ${bufferSize.rank}) must be equal to 0.")
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
        .addInput(bufferSize)
        .addInput(seed1)
        .addInput(seed2)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that contains `count` entries from the provided dataset.
    *
    * @param  datasetHandle   Handle of the dataset to take entries from.
    * @param  count           [[INT64]] scalar tensor containing the number of entries to take from the provided
    *                         dataset.
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
    * @param  count           [[INT64]] scalar tensor containing the number of entries to skip from the provided
    *                         dataset.
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
    Op.Builder(opType = "ZipDataset", name = name)
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

  /** Creates an op representing a dataset that maps a function over another dataset.
    *
    * @param  datasetHandle   Handle of the dataset to map `function` over.
    * @param  function        Mapping function.
    * @param  outputDataTypes Output data types of the created dataset.
    * @param  outputShapes    Output shapes of the created dataset.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[io] def datasetMap(
      datasetHandle: Output, otherArguments: Seq[Output], function: InstantiatedFunction[_, _],
      outputDataTypes: Seq[DataType], outputShapes: Seq[Shape], name: String = "DatasetMap"): Output = {
    Op.Builder(opType = "MapDataset", name = name)
        .addInput(datasetHandle)
        .addInputList(otherArguments)
        .setAttribute("f", function)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  /** Creates an op representing a dataset that maps a function over another dataset and support parallel execution of
    * the map operations.
    *
    * @param  datasetHandle    Handle of the dataset to map `function` over.
    * @param  function         Mapping function.
    * @param  numParallelCalls [[INT32]] scalar tensor specifying the number of concurrent invocations of `function`
    *                          that process elements from `inputDataset` in parallel.
    * @param  outputDataTypes  Output data types of the created dataset.
    * @param  outputShapes     Output shapes of the created dataset.
    * @param  name             Name for the created op.
    * @return Created op output, which is a handle to the created dataset.
    */
  private[io] def datasetParallelMap(
      datasetHandle: Output, otherArguments: Seq[Output], function: InstantiatedFunction[_, _],
      numParallelCalls: Output, outputDataTypes: Seq[DataType], outputShapes: Seq[Shape],
      name: String = "DatasetParallelMap"): Output = {
    Op.Builder(opType = "ParallelMapDataset", name = name)
        .addInput(datasetHandle)
        .addInputList(otherArguments)
        .addInput(numParallelCalls)
        .setAttribute("f", function)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  // TODO: [DATASETS] "denseToSparseBatch".

  /** Creates an op representing a dataset that asynchronously prefetches elements from `dataset`.
    *
    * A cached dataset will iterate over the input dataset and store the tensors it gets. If the cache already exists,
    * it will be used. If the cache is inappropriate (e.g., it cannot be opened, or contains tensors of the wrong shape
    * or size), an error will the returned when used.
    *
    * @param  datasetHandle   Handle of the dataset to cache.
    * @param  bufferSize      [[INT64]] scalar tensor containing the maximum number of elements to buffer in an iterator
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
    GradientsRegistry.registerNonDifferentiable("PrefetchDataset")
    GradientsRegistry.registerNonDifferentiable("IgnoreErrorsDataset")
    GradientsRegistry.registerNonDifferentiable("DenseToSparseBatchDataset")
    GradientsRegistry.registerNonDifferentiable("MapDataset")
    GradientsRegistry.registerNonDifferentiable("ParallelMapDataset")
    GradientsRegistry.registerNonDifferentiable("FlatMapDataset")
    GradientsRegistry.registerNonDifferentiable("InterleaveDataset")
    GradientsRegistry.registerNonDifferentiable("GroupByWindowDataset")
    GradientsRegistry.registerNonDifferentiable("FilterDataset")
  }

  /** @define OpDocDatasetBatch
    *   The dataset `batch` op combines consecutive elements of a dataset into batches.
    *
    * @define OpDocDatasetRepeat
    *   The dataset `repeat` op repeats a dataset a specified number of times. If the provided number of times to repeat
    *   is set to `-1`, then the dataset is repeated indefinitely.
    *
    * @define OpDocDatasetCache
    *   The dataset `cache` op caches the elements in a dataset in the provided directory. If the provided directory is
    *   empty, then the elements are cached in memory.
    *
    * @define OpDocDatasetShuffle
    *   The dataset `shuffle` op randomly shuffles the elements of a dataset.
    *
    * @define OpDocDatasetTake
    *   The dataset `take` op takes at most the provided number of elements from a dataset, forming a new dataset. If
    *   the provided number is `-1`, then all of the elements are taken.
    *
    *   The op has similar semantics to the built-in Scala collections `take` function.
    *
    * @define OpDocDatasetDrop
    *   The dataset `drop` op drops at most the provided number of elements from a dataset, forming a new dataset. If
    *   the provided number is `-1`, then all of the elements are dropped.
    *
    *   The op has similar semantics to the built-in Scala collections `drop` function.
    *
    * @define OpDocDatasetZip
    *   The dataset `zip`, `zip3`, and `zipMultiple` ops create a new dataset by zipping together multiple datasets.
    *
    *   The ops have similar semantics to the built-in Scala collections `zip` and `zip3` functions.
    *
    *   The main difference between the three ops is that `zip` is limited to two datasets of potentially
    *   differently-typed elements, `zip3` is similarly limited to three datasets of potentially differently-typed
    *   elements, and `zipMultiple` can zip together an arbitrary number of datasets containing elements of the same
    *   type.
    *
    *   For example:
    *   {{{
    *     // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *     a = { 1, 2, 3 }
    *     b = { 4, 5, 6 }
    *     c = { (7, 8), (9, 10), (11, 12) }
    *     d = { 13, 14 }
    *
    *     // The nested structure of the `datasets` argument determines the structure of elements in the resulting
    *     // dataset.
    *     a.zip(b) ==> { (1, 4), (2, 5), (3, 6) }
    *     b.zip(a) ==> { (4, 1), (5, 2), (6, 3) }
    *
    *     // The `datasets` argument may contain an arbitrary number of datasets.
    *     a.zip3(b, c) ==> { (1, 4, (7, 8)), (2, 5, (9, 10)), (3, 6, (11, 12)) }
    *
    *     // The number of elements in the resulting dataset is the same as the size of the smallest provided dataset.
    *     a.zip(d) ==> { (1, 13), (2, 14) }
    *
    *     // The `zipMultiple` op returns datasets with sequence-valued elements.
    *     a.zipMultiple(b) ==> { Seq(1, 4), Seq(2, 5), Seq(3, 6) }
    *   }}}
    *
    * @define OpDocDatasetConcatenate
    *   The dataset `concatenate` op creates a new dataset by concatenating the provided datasets.
    *
    *   For example:
    *   {{{
    *     // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *     a = { 1, 2, 3 }
    *     b = { 4, 5, 6, 7 }
    *     a.concatenate(b) ==> { 1, 2, 3, 4, 5, 6, 7 }
    *
    *     // The datasets to be concatenated should have the same nested structures and output types.
    *     c = { (8, 9), (10, 11), (12, 13) }
    *     d = { 14.0, 15.0, 16.0 }
    *     // a.concatenate(c) and a.concatenate(d) would result in exceptions being thrown.
    *   }}}
    *
    * @define OpDocDatasetMap
    *   The dataset `map` op creates a new dataset by a function across all elements of another dataset.
    *
    *   The op has similar semantics to the built-in Scala collections `map` function.
    *
    * @define OpDocDatasetPrefetch
    *   The dataset `prefetch` op creates a new dataset by asynchronously prefetching elements from the provided
    *   dataset.
    *
    * @define OpDocDatasetIgnoreErrors
    *   The dataset `ignoreErrors` creates a new dataset from the provided one and silently ignores any errors.
    *
    *   Use this transformation to produce a dataset that contains the same elements as the input, but silently drops
    *   any elements that caused an error. For example:
    *   {{{
    *     dataset = datasetFromSlices(Tensor(1.0, 2.0, 0.0, 4.0))
    *
    *     // Computing `checkNumerics(1.0 / 0.0)` will raise an [[IllegalArgumentException]].
    *     dataset = dataset.map(x => checkNumerics(1.0 / x, "error"))
    *
    *     // Using `ignoreErrors` will drop the elements that cause errors.
    *     dataset = dataset.ignoreErrors()  // ==> { 1.0, 0.5, 0.2 }
    *   }}}
    */
  private[ops] trait Documentation
}

/** [[Dataset]] with a single element.
  *
  * @param  data Data representing the single element of this dataset.
  * @param  name Name for this dataset.
  */
private[io] case class TensorDataset[T, O, D, S] private[io](
    data: T, override val name: String = "TensorDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val flattenedOutputs = ev.flattenedOutputs(data)
    Dataset.createTensorDataset(flattenedOutputs, flattenedOutputShapes, name)
  }

  override val outputDataTypes: D = ev.dataTypes(data)
  override val outputShapes   : S = ev.shapes(data)
}

/** [[Dataset]] with slices from the nested structure of [[Output]]s (i.e., a [[Data]]-supported type). The slices are
  * taken along the first axis of each [[Output]] in the nested structure.
  *
  * @param  data Data representing the elements of this dataset.
  * @param  name Name for this dataset.
  */
private[io] case class TensorSliceDataset[T, O, D, S] private[io](
    data: T, override val name: String = "TensorSliceDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val flattenedOutputs = ev.flattenedOutputs(data)
    Dataset.createTensorSliceDataset(flattenedOutputs, flattenedOutputShapes, name)
  }

  override val outputDataTypes: D = ev.dataTypes(data)

  override val outputShapes: S = {
    val flattenedShapes = ev.flattenedShapes(ev.shapes(data))
    ev.unflattenShapes(outputDataTypes, flattenedShapes.map(s => if (s.rank > 1) s(1 ::) else Shape.scalar()))
  }
}

/** [[Dataset]] that splits a sparse tensor into its rows.
  *
  * @param  tensor Sparse tensor.
  * @param  name   Name for this dataset.
  */
private[io] case class SparseTensorSliceDataset private[io](
    tensor: SparseTensor, override val name: String = "SparseTensorSliceDataset")
    extends Dataset[SparseTensor, SparseOutput, (DataType, DataType, DataType), (Shape, Shape, Shape)](name) {
  /** Creates a `RESOURCE` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  override def createHandle(): Output = {
    Dataset.createSparseTensorSliceDataset(
      tensor.indices,
      tensor.values,
      tensor.denseShape,
      name)
  }

  override val outputDataTypes: (DataType, DataType, DataType) = (INT64, tensor.dataType, INT64)

  override val outputShapes: (Shape, Shape, Shape) = {
    val indicesShape = tensor.indices.shape
    val denseShapeShape = tensor.denseShape.shape
    val rank = Shape(indicesShape(1) - 1).mergeWith(Shape(denseShapeShape(0) - 1))(0)
    (Shape(-1, rank), Shape(-1), Shape(rank))
  }
}

/** [[Dataset]] that wraps the application of the `range` op.
  *
  * $OpDocDatasetRange
  *
  * @param  start Starting value of the number sequence.
  * @param  limit Ending value (exclusive) of the number sequence.
  * @param  delta Difference between consecutive numbers in the sequence.
  * @param  name  Name for this dataset.
  */
private[io] case class RangeDataset private[io](
    start: Long, limit: Long, delta: Long, override val name: String = "RangeDataset")
    extends Dataset[Tensor, Output, DataType, Shape](name) {
  override def createHandle(): Output = {
    Dataset.createRangeDataset(
      Op.createWithNameScope(name)(Basic.constant(start)),
      Op.createWithNameScope(name)(Basic.constant(limit)),
      Op.createWithNameScope(name)(Basic.constant(delta)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: DataType = INT64
  override val outputShapes   : Shape    = Shape.scalar()
}

/** [[Dataset]] that wraps the application of the `batch` op.
  *
  * $OpDocDatasetBatch
  *
  * @param  inputDataset Input dataset.
  * @param  batchSize    Batch size to use.
  * @param  name         Name for this dataset.
  */
private[io] case class BatchDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], batchSize: Long, override val name: String = "BatchDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetBatch(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(batchSize)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = {
    ev.unflattenShapes(outputDataTypes, inputDataset.flattenedOutputShapes.map(Shape(-1) ++ _))
  }
}

// TODO: !!! PaddedBatchDataset

/** [[Dataset]] that wraps the application of the `repeat` op.
  *
  * $OpDocDatasetRepeat
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of times to repeat the input dataset. A value of `-1` corresponds to repeating it
  *                      indefinitely.
  * @param  name         Name for this dataset.
  */
private[io] case class RepeatDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], count: Long, override val name: String = "RepeatDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetRepeat(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(count)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}

/** [[Dataset]] that wraps the application of the `cache` op.
  *
  * $OpDocDatasetCache
  *
  * @param  inputDataset Input dataset.
  * @param  directory    Directory to use for caching. If empty, then the provided dataset will be cached in memory.
  * @param  name         Name for this dataset.
  */
private[io] case class CacheDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], directory: String, override val name: String = "CacheDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetCache(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(directory)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}

/** [[Dataset]] that wraps the application of the `shuffle` op.
  *
  * $OpDocDatasetShuffle
  *
  * @param  inputDataset Input dataset.
  * @param  bufferSize   Buffer size, meaning the number of output elements to buffer in an iterator over this dataset.
  * @param  seed         Seed value for the random number generator. If not provided, a random seed is used.
  * @param  name         Name for this dataset.
  */
private[io] case class ShuffleDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], bufferSize: Long, seed: Option[Int],
    override val name: String = "ShuffleDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val seed1 = graphSeed.getOrElse(0)
    val seed2 = opSeed.getOrElse(0)
    Dataset.datasetShuffle(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(bufferSize)),
      Op.createWithNameScope(name)(Basic.constant(seed1, INT64)),
      Op.createWithNameScope(name)(Basic.constant(seed2, INT64)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}

/** [[Dataset]] that wraps the application of the `take` op.
  *
  * $OpDocDatasetTake
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of elements to take.
  * @param  name         Name for this dataset.
  */
private[io] case class TakeDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], count: Long, override val name: String = "TakeDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetTake(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(count)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}

/** [[Dataset]] that wraps the application of the `drop` op.
  *
  * $OpDocDatasetDrop
  *
  * @param  inputDataset Input dataset.
  * @param  count        Number of elements to drop.
  * @param  name         Name for this dataset.
  */
private[io] case class DropDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], count: Long, override val name: String = "DropDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetSkip(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(count)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}

/** [[Dataset]] that wraps the application of the `zip` op.
  *
  * $OpDocDatasetZip
  *
  * @param  inputDataset1 First input dataset.
  * @param  inputDataset2 Second input dataset.
  * @param  name          Name for this dataset.
  */
private[io] case class ZipDataset[T1, O1, D1, S1, T2, O2, D2, S2] private[io](
    inputDataset1: Dataset[T1, O1, D1, S1],
    inputDataset2: Dataset[T2, O2, D2, S2],
    override val name: String = "ZipDataset")(implicit
    ev1: Data.Aux[T1, O1, D1, S1],
    ev2: Data.Aux[T2, O2, D2, S2]
) extends Dataset[(T1, T2), (O1, O2), (D1, D2), (S1, S2)](name) {
  override def createHandle(): Output = {
    Dataset.datasetZip(
      Op.createWithNameScope(name)(Seq(inputDataset1.createHandle(), inputDataset2.createHandle())),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: (D1, D2) = (inputDataset1.outputDataTypes, inputDataset2.outputDataTypes)
  override val outputShapes   : (S1, S2) = (inputDataset1.outputShapes, inputDataset2.outputShapes)
}

/** [[Dataset]] that wraps the application of the `zip3` op.
  *
  * $OpDocDatasetZip
  *
  * @param  inputDataset1 First input dataset.
  * @param  inputDataset2 Second input dataset.
  * @param  inputDataset3 Second input dataset.
  * @param  name          Name for this dataset.
  */
private[io] case class Zip3Dataset[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3] private[io](
    inputDataset1: Dataset[T1, O1, D1, S1],
    inputDataset2: Dataset[T2, O2, D2, S2],
    inputDataset3: Dataset[T3, O3, D3, S3],
    override val name: String = "Zip3Dataset")(implicit
    ev1: Data.Aux[T1, O1, D1, S1],
    ev2: Data.Aux[T2, O2, D2, S2],
    ev3: Data.Aux[T3, O3, D3, S3]
) extends Dataset[(T1, T2, T3), (O1, O2, O3), (D1, D2, D3), (S1, S2, S3)](name) {
  override def createHandle(): Output = {
    Dataset.datasetZip(
      Op.createWithNameScope(name)(Seq(
        inputDataset1.createHandle(), inputDataset2.createHandle(), inputDataset3.createHandle())),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: (D1, D2, D3) = {
    (inputDataset1.outputDataTypes, inputDataset2.outputDataTypes, inputDataset3.outputDataTypes)
  }

  override val outputShapes: (S1, S2, S3) = {
    (inputDataset1.outputShapes, inputDataset2.outputShapes, inputDataset3.outputShapes)
  }
}

/** [[Dataset]] that wraps the application of the `zipMultiple` op.
  *
  * $OpDocDatasetZip
  *
  * @param  inputDatasets Input datasets.
  * @param  name          Name for this dataset.
  */
private[io] case class ZipMultipleDataset[T, O, D, S] private[io](
    inputDatasets: Seq[Dataset[T, O, D, S]], override val name: String = "ZipMultipleDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[Seq[T], Seq[O], Seq[D], Seq[S]](name) {
  override def createHandle(): Output = {
    Dataset.datasetZip(
      Op.createWithNameScope(name)(inputDatasets.map(_.createHandle())),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: Seq[D] = inputDatasets.map(_.outputDataTypes)
  override val outputShapes   : Seq[S] = inputDatasets.map(_.outputShapes)
}

/** [[Dataset]] that wraps the application of the `concatenate` op.
  *
  * $OpDocDatasetConcatenate
  *
  * @param  inputDataset1 First input dataset.
  * @param  inputDataset2 Second input dataset.
  * @param  name         Name for this dataset.
  * @throws IllegalArgumentException If the data types of the input datasets are not identical of if their shapes are
  *                                  not compatible.
  */
@throws[IllegalArgumentException]
private[io] case class ConcatenatedDataset[T, O, D, S] private[io](
    inputDataset1: Dataset[T, O, D, S],
    inputDataset2: Dataset[T, O, D, S],
    override val name: String = "ConcatenatedDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  if (inputDataset1.flattenedOutputDataTypes != inputDataset2.flattenedOutputDataTypes)
    throw new IllegalArgumentException("The data types of the datasets being concatenated are not the identical.")
  private[this] val mostSpecificFlattenedShapes = {
    inputDataset1.flattenedOutputShapes.zip(inputDataset2.flattenedOutputShapes).map(p => {
      if (!p._1.isCompatibleWith(p._2))
        throw new IllegalArgumentException("The shapes of the datasets being concatenated are not compatible.")
      p._1.mergeWith(p._2)
    })
  }

  override def createHandle(): Output = {
    Dataset.datasetConcatenate(
      Op.createWithNameScope(name)(inputDataset1.createHandle()),
      Op.createWithNameScope(name)(inputDataset2.createHandle()),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset1.outputDataTypes
  override val outputShapes   : S = ev.unflattenShapes(outputDataTypes, mostSpecificFlattenedShapes)
}

/** [[Dataset]] that wraps the application of the `map` op.
  *
  * $OpDocDatasetMap
  *
  * @param  inputDataset Input dataset.
  * @param  function     Mapping function.
  * @param  name         Name for this dataset.
  */
private[io] case class MapDataset[T, O, D, S, RT, RO, RD, RS] private[io](
    inputDataset: Dataset[T, O, D, S],
    function: (T) => RT,
    override val name: String = "MapDataset")(implicit
    ev: Data.Aux[T, O, D, S],
    evR: Data.Aux[RT, RO, RD, RS],
    evFunctionInput: Function.ArgType[T],
    evFunctionOutput: Function.ArgType[RT]
) extends Dataset[RT, RO, RD, RS](name) {
  private[this] val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(inputDataset.flattenedOutputDataTypes)
  }

  override def createHandle(): Output = {
    Dataset.datasetMap(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      instantiatedFunction.extraInputs,
      instantiatedFunction,
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val (outputDataTypes: RD, outputShapes: RS) = {
    val dataTypes = evR.dataTypes(instantiatedFunction.dummyOutputs)
    (evR.unflattenDataTypes(dataTypes, instantiatedFunction.outputDataTypes),
        evR.unflattenShapes(dataTypes, instantiatedFunction.outputShapes))
  }
}

/** [[Dataset]] that wraps the application of the `parallelMap` op.
  *
  * $OpDocDatasetMap
  *
  * @param  inputDataset     Input dataset.
  * @param  function         Mapping function.
  * @param  numParallelCalls Number of concurrent invocations of `function` that process elements from `inputDataset` in
  *                          parallel.
  * @param  name             Name for this dataset.
  */
private[io] case class ParallelMapDataset[T, O, D, S, RT, RO, RD, RS] private[io](
    inputDataset: Dataset[T, O, D, S],
    function: (T) => RT,
    numParallelCalls: Int,
    override val name: String = "ParallelMapDataset")(implicit
    ev: Data.Aux[T, O, D, S],
    evR: Data.Aux[RT, RO, RD, RS],
    evFunctionInput: Function.ArgType[T],
    evFunctionOutput: Function.ArgType[RT]
) extends Dataset[RT, RO, RD, RS](name) {
  private[this] val instantiatedFunction = {
    Function(s"$name/Function", function).instantiate(inputDataset.flattenedOutputDataTypes)
  }

  override def createHandle(): Output = {
    Dataset.datasetParallelMap(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      instantiatedFunction.extraInputs,
      instantiatedFunction,
      Op.createWithNameScope(name)(Basic.constant(numParallelCalls, name = "NumParallelCalls")),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val (outputDataTypes: RD, outputShapes: RS) = {
    val dataTypes = evR.dataTypes(instantiatedFunction.dummyOutputs)
    (evR.unflattenDataTypes(dataTypes, instantiatedFunction.outputDataTypes),
        evR.unflattenShapes(dataTypes, instantiatedFunction.outputShapes))
  }
}

/** [[Dataset]] that wraps the application of the `prefetch` op.
  *
  * $OpDocDatasetPrefetch
  *
  * @param  inputDataset Input dataset.
  * @param  bufferSize   Number of elements to prefetch.
  * @param  name         Name for this dataset.
  */
private[io] case class PrefetchDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], bufferSize: Long, override val name: String = "PrefetchDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetPrefetch(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      Op.createWithNameScope(name)(Basic.constant(bufferSize)),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}

/** [[Dataset]] that wraps the application of the `ignoreErrors` op.
  *
  * $OpDocDatasetIgnoreErrors
  *
  * @param  inputDataset Input dataset.
  * @param  name         Name for this dataset.
  */
private[io] case class IgnoreErrorsDataset[T, O, D, S] private[io](
    inputDataset: Dataset[T, O, D, S], override val name: String = "IgnoreErrorsDataset")(implicit
    ev: Data.Aux[T, O, D, S]
) extends Dataset[T, O, D, S](name) {
  override def createHandle(): Output = {
    Dataset.datasetIgnoreErrors(
      Op.createWithNameScope(name)(inputDataset.createHandle()),
      flattenedOutputDataTypes,
      flattenedOutputShapes,
      name)
  }

  override val outputDataTypes: D = inputDataset.outputDataTypes
  override val outputShapes   : S = inputDataset.outputShapes
}
