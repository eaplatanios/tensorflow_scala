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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.implicits.helpers.{StructureFromDataType, StructureFromOutput, StructureFromTensor}
import org.platanios.tensorflow.api.ops.{Callback, Function, Math, Op, Output}
import org.platanios.tensorflow.api.ops.io.data
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT64}

import java.util.concurrent.atomic.AtomicLong

import scala.collection.generic.CanBuildFrom
import scala.collection.{SeqLike, mutable}
import scala.language.postfixOps

/** Represents a potentially large set of elements.
  *
  * A dataset can be used to represent an input pipeline as a collection of elements (i.e., nested structures of
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
abstract class Dataset[T, O, D, S](
    val name: String = "Dataset"
)(implicit
    val evStructure: StructureFromOutput.Aux[T, O, D, S],
    val evData: Data.Aux[T, O, D, S],
    val evFunctionInput: Function.ArgType[O]
) {
  /** Creates a `VARIANT` scalar tensor representing this dataset. This function adds ops to the current graph, that
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
      sharedName: String = "",
      name: String = "InitializableIterator"
  ): InitializableIterator[T, O, D, S] = {
    Iterator.fromDataset(this, sharedName, name)
  }

  // TODO: [DATASETS] "createOneShotIterator".

  /** Returns the data types corresponding to each element of this dataset, matching the structure of the elements. */
  def outputDataTypes: D

  /** Returns the shapes corresponding to each element of this dataset, matching the structure of the elements. */
  def outputShapes: S

  /** Returns a sequence of data types that correspond to the flattened data types of the nested [[Output]] structure
    * of the elements of this dataset. */
  private[io] def flattenedOutputDataTypes: Seq[DataType[_]] = evData.flattenedDataTypes(outputDataTypes)

  /** Returns a sequence of [[Shape]]s that correspond to the flattened shapes of the nested [[Output]] structure of the
    * elements of this dataset. */
  private[io] def flattenedOutputShapes: Seq[Shape] = evData.flattenedShapes(outputShapes)

  /** Creates a dataset that includes only `1 / numShards` of the elements of this dataset.
    *
    * This operator is very useful when running distributed training, as it allows each worker to read a unique subset
    * of the dataset.
    *
    * When reading a single input file, you can skip elements as follows:
    * {{{
    *   tf.data.TFRecordDataset(inputFile)
    *     .shard(numWorkers, workerIndex)
    *     .repeat(numEpochs)
    *     .shuffle(shuffleBufferSize)
    *     .map(parserFn, numParallelCalls)
    * }}}
    *
    * Important caveats:
    *
    *   - Be sure to shard before you use any randomizing operator (such as shuffle).
    *   - Generally it is best if the shard operator is used early in the dataset pipeline. For example, when reading
    *     from a set of TensorFlow record files, shard before converting the dataset to input samples. This avoids
    *     reading every file on every worker. The following is an example of an efficient sharding strategy within a
    *     complete pipeline:
    *     {{{
    *       tf.data.listFiles(pattern)
    *         .shard(numWorkers, workerIndex)
    *         .repeat(numEpochs)
    *         .shuffle(shuffleBufferSize)
    *         .repeat()
    *         .interleave(tf.data.TFRecordDataset, cycleLength = numReaders, blockLength = 1)
    *         .map(parserFn, numParallelCalls)
    *     }}}
    *
    * @param  numShards  Number of shards to use.
    * @param  shardIndex Index of the shard to obtain.
    * @return Created (sharded) dataset.
    */
  def shard(numShards: Long, shardIndex: Long): Dataset[T, O, D, S] = {
    if (shardIndex >= numShards)
      throw InvalidArgumentException(s"'index' (= $shardIndex) must be smaller than 'numShards' (= $numShards).")
    if (numShards == 1)
      this
    else
      this.zip(RangeDataset(0, Long.MaxValue))
          .filter(t => Math.equal(Math.mod(t._2, numShards), shardIndex))
          .map(o => o._1)
  }

  /** Applies a transformation function to this dataset.
    *
    * `transform()` enables chaining of custom dataset transformations, which are represented as functions that take one
    * dataset argument and return a transformed dataset.
    *
    * @param  transformFn Dataset transformation function.
    * @return Transformed dataset.
    */
  def transform[TT, TO, TD, TS](transformFn: Dataset[T, O, D, S] => Dataset[TT, TO, TD, TS])(implicit
      evStructure: StructureFromOutput.Aux[TT, TO, TD, TS],
      evT: Data.Aux[TT, TO, TD, TS],
      evFunctionInputT: Function.ArgType[TO]
  ): Dataset[TT, TO, TD, TS] = {
    transformFn(this)
  }

  override def toString: String = {
    "Dataset[" +
        s"outputDataTypes = ${evData.dataTypesToString(outputDataTypes)}, " +
        s"outputShapes = ${evData.shapesToString(outputShapes)}" +
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
    type SparseTensorSlicesDataset[T] = data.SparseTensorSlicesDataset[T]
    type SparseOutputSlicesDataset[T] = data.SparseOutputSlicesDataset[T]
    type TextLinesDataset = data.TextLinesDataset
    type DynamicTextLinesDataset = data.DynamicTextLinesDataset
    type FixedLengthRecordDataset = data.FixedLengthRecordDataset
    type TFRecordDataset = data.TFRecordDataset
    type DynamicTFRecordDataset = data.DynamicTFRecordDataset

    type BatchDataset[T, O, D, S] = data.BatchDataset[T, O, D, S]
    type PaddedBatchDataset[T, O, D, S] = data.PaddedBatchDataset[T, O, D, S]
    type PrefetchDataset[T, O, D, S] = data.PrefetchDataset[T, O, D, S]
    type CacheDataset[T, O, D, S] = data.CacheDataset[T, O, D, S]
    type ShuffleDataset[T, O, D, S] = data.ShuffleDataset[T, O, D, S]
    type RepeatDataset[T, O, D, S] = data.RepeatDataset[T, O, D, S]
    type IgnoreErrorsDataset[T, O, D, S] = data.IgnoreErrorsDataset[T, O, D, S]

    type TakeDataset[T, O, D, S] = data.TakeDataset[T, O, D, S]
    type DropDataset[T, O, D, S] = data.DropDataset[T, O, D, S]

    type FilterDataset[T, O, D, S] = data.FilterDataset[T, O, D, S]
    type MapDataset[T, O, D, S, RT, RO, RD, RS] = data.MapDataset[T, O, D, S, RT, RO, RD, RS]
    type FlatMapDataset[T, O, D, S, RT, RO, RD, RS] = data.FlatMapDataset[T, O, D, S, RT, RO, RD, RS]

    type ZipDataset[T1, O1, D1, S1, T2, O2, D2, S2] = data.ZipDataset[T1, O1, D1, S1, T2, O2, D2, S2]
    type Zip3Dataset[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3] = data.Zip3Dataset[T1, O1, D1, S1, T2, O2, D2, S2, T3, O3, D3, S3]
    type ZipMultipleDataset[T, O, D, S] = data.ZipMultipleDataset[T, O, D, S]

    type ConcatenatedDataset[T, O, D, S] = data.ConcatenatedDataset[T, O, D, S]

    type GroupByWindowDataset[T, O, D, S] = data.GroupByWindowDataset[T, O, D, S]

    val RangeDataset             : data.RangeDataset.type              = data.RangeDataset
    val TensorDataset            : data.TensorDataset.type             = data.TensorDataset
    val OutputDataset            : data.OutputDataset.type             = data.OutputDataset
    val TensorSlicesDataset      : data.TensorSlicesDataset.type       = data.TensorSlicesDataset
    val OutputSlicesDataset      : data.OutputSlicesDataset.type       = data.OutputSlicesDataset
    val SparseTensorSlicesDataset: data.SparseTensorSlicesDataset.type = data.SparseTensorSlicesDataset
    val SparseOutputSlicesDataset: data.SparseOutputSlicesDataset.type = data.SparseOutputSlicesDataset
    val TextLinesDataset         : data.TextLinesDataset.type          = data.TextLinesDataset
    val DynamicTextLinesDataset  : data.DynamicTextLinesDataset.type   = data.DynamicTextLinesDataset
    val FixedLengthRecordDataset : data.FixedLengthRecordDataset.type  = data.FixedLengthRecordDataset
    val TFRecordDataset          : data.TFRecordDataset.type           = data.TFRecordDataset
    val DynamicTFRecordDataset   : data.DynamicTFRecordDataset.type    = data.DynamicTFRecordDataset

    val BatchDataset       : data.BatchDataset.type        = data.BatchDataset
    val PaddedBatchDataset : data.PaddedBatchDataset.type  = data.PaddedBatchDataset
    val PrefetchDataset    : data.PrefetchDataset.type     = data.PrefetchDataset
    val CacheDataset       : data.CacheDataset.type        = data.CacheDataset
    val ShuffleDataset     : data.ShuffleDataset.type      = data.ShuffleDataset
    val RepeatDataset      : data.RepeatDataset.type       = data.RepeatDataset
    val IgnoreErrorsDataset: data.IgnoreErrorsDataset.type = data.IgnoreErrorsDataset

    val TakeDataset: data.TakeDataset.type = data.TakeDataset
    val DropDataset: data.DropDataset.type = data.DropDataset

    val FilterDataset            : data.FilterDataset.type             = data.FilterDataset
    val MapDataset               : data.MapDataset.type                = data.MapDataset
    val FlatMapDataset           : data.FlatMapDataset.type            = data.FlatMapDataset
    val InterleaveDataset        : data.InterleaveDataset.type         = data.InterleaveDataset
    val ParallelInterleaveDataset: data.ParallelInterleaveDataset.type = data.ParallelInterleaveDataset

    val ZipDataset        : data.ZipDataset.type         = data.ZipDataset
    val Zip3Dataset       : data.Zip3Dataset.type        = data.Zip3Dataset
    val ZipMultipleDataset: data.ZipMultipleDataset.type = data.ZipMultipleDataset

    val ConcatenatedDataset: data.ConcatenatedDataset.type = data.ConcatenatedDataset

    val GroupByWindowDataset: data.GroupByWindowDataset.type = data.GroupByWindowDataset

    def fromGenerator[T, O, D, S](
        generator: () => Iterable[T],
        outputDataType: D,
        outputShape: S = null
    )(implicit
        evStructureFromDataType: StructureFromDataType.Aux[T, O, D, S],
        evData: Data.Aux[T, O, D, S],
        evStructureFromTensor: StructureFromTensor.Aux[T, O, D, S],
        evFunctionOutput: Function.ArgType[O]
    ): Dataset[T, O, D, S] = {
      Dataset.fromGenerator[T, O, D, S](generator, outputDataType, outputShape)
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

  /** Creates a [[Dataset]] whose elements are generated by Scala iterables over dataset elements.
    *
    * The `generator` argument must be a function that takes no arguments and returns an [[Iterable]] over dataset
    * elements. The elements contained in that [[Iterable]] must be compatible with the provided `outputDataType` and
    * `outputShape` arguments.
    *
    * For example:
    * {{{
    *   // TODO: !!! Improve this example with variable shapes -- like in the Python API.
    *   val generator = () => Range(0, 10).map(Tensor(_))
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
      generator: () => Iterable[T],
      outputDataType: D,
      outputShape: S = null
  )(implicit
      evStructureFromDataType: StructureFromDataType.Aux[T, O, D, S],
      evData: Data.Aux[T, O, D, S],
      evStructureFromTensor: StructureFromTensor.Aux[T, O, D, S],
      evFunctionOutput: Function.ArgType[O]
  ): Dataset[T, O, D, S] = {
    val inferredOutputShape: S = {
      if (outputShape != null)
        outputShape
      else
        evData.unflattenShapes(outputDataType, Seq.fill(evData.size(outputDataType))(Shape.unknown()))
    }

    val flattenedTypes = evData.flattenedDataTypes(outputDataType)
    val flattenedShapes = evData.flattenedShapes(inferredOutputShape)
    val generatorState = GeneratorState(generator)(evData)

    /** Creates an op that generates the next element from iterator with ID, `iteratorId`.
      *
      * We map this function across an infinite repetition of the `iteratorId`, and throw an `OutOfRange` to terminate
      * the iteration.
      *
      * @param  iteratorId Scalar tensor whose value uniquely identifies the iterator in the internal
      *                    generator state, from which to generate an element.
      * @return Created op outputs structured according to the output data type of this dataset.
      */
    def generatorMapFn(iteratorId: Output): O = {
      /** Scala callback function that will be called to invoke the iterator. */
      @throws[OutOfRangeException]
      def generatorScalaCallback(iteratorId: Tensor[Long]): Seq[Tensor[_]] = {
        val iterator = generatorState.getIterator(iteratorId.scalar)
        val value = {
          if (iterator.hasNext)
            iterator.next()
          else
            throw OutOfRangeException("The iterator does not contain any more elements.")
        }
        val flattenedTensors = evData.flattenedTensors(value)
        // Additional type and shape checking to ensure that the components of the generated element match the
        // output data types and output shapes arguments.
        (flattenedTensors, flattenedTypes, flattenedShapes).zipped.foreach((tensor, dataType, shape) => {
          if (tensor.dataType != dataType)
            throw InvalidDataTypeException(
              s"The generator yielded an element of type ${tensor.dataType} " +
                  s"where an element of type $dataType was expected.")
          if (!tensor.shape.isCompatibleWith(shape))
            throw InvalidShapeException(
              s"The generator yielded an element with shape ${tensor.shape} " +
                  s"where an element with shape $shape was expected.")
        })
        flattenedTensors
      }

      implicit def tensorSeqArgType[CC[A] <: SeqLike[A, CC[A]]](implicit
          cbfTensor: CanBuildFrom[Seq[Tensor[_]], Tensor[_], CC[Tensor[_]]],
          cbfOutput: CanBuildFrom[Seq[Output], Output, CC[Output]]
      ): Callback.ArgType.Aux[CC[Tensor[_]], CC[Output], CC[DataType[_]]] = {
        new Callback.ArgType[CC[Tensor[_]]] {
          override type TS = CC[Output]
          override type TD = CC[DataType[_]]

          override def tensors(arg: CC[Tensor[_]]): Seq[Tensor[_]] = arg.toSeq
          override def outputs(arg: CC[Output]): Seq[Output] = arg.toSeq
          override def dataTypes(types: CC[DataType[_]]): Seq[DataType[_]] = types.toSeq
          override def decode(tensors: Seq[Tensor[_]]): CC[Tensor[_]] = tensors.to[CC](cbfTensor)
          override def decodeSymbolic(outputs: Seq[Output]): CC[Output] = outputs.to[CC](cbfOutput)
        }
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
      evData.unflattenOutputs(outputDataType, flattenedValues)
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
    val idDataset = TensorDataset(Tensor(INT64, 0L)).map(
      (_: Output) => Callback.callback((_: Unit) => Tensor(INT64, generatorState.nextId), (), INT64, stateful = true))

    // A dataset that contains all of the elements generated by a single iterator created from the provided generator,
    // identified by the iterator ID contained in `idDataset`. Lifting the iteration into a `flatMap` here enables
    // multiple repetitions and/or nested versions of the returned dataset to be created, because it forces the
    // generation of a new ID for each version.
    idDataset.flatMap(flatMapFn)
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
      datasetHandle: Output,
      batchSize: Output,
      paddedShapes: Seq[Output],
      paddingValues: Seq[Output],
      outputShapes: Seq[Shape],
      name: String = "DatasetPaddedBatch"
  ): Output = {
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
}
