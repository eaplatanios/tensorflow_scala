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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.TensorToOutput
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.types.{DataType, INT64, VARIANT}

import scala.language.postfixOps

/** Represents a potentially large set of elements.
  *
  * A dataset can be used to represent an input pipeline as a collection of elements (i.e., nested structures of
  * tensors) and a "logical plan" of transformations that act on those elements.
  *
  * @tparam T Tensor type (i.e., nested structure of outputs).
  *
  * @author Emmanouil Antonios Platanios
  */
trait Dataset[T] { outer =>
  val name: String

  implicit val evData: SupportedData[T]

  /** Creates a `VARIANT` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  def createHandle(): Output[Long]

  /** Creates a dataset iterator for this dataset.
    *
    * **Note:** The returned iterator will be in an uninitialized state. You must execute its `initializer` op before
    * using it.
    *
    * @param  sharedName If non-empty, then the constructed reader will be shared under the the provided name across
    *                    multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name       Name for the op created in relation to the iterator.
    * @return Created iterator.
    */
  def createInitializableIterator(
      sharedName: String = "",
      name: String = "InitializableDatasetIterator"
  ): InitializableDatasetIterator[T] = {
    DatasetIterator.fromDataset(dataset = this, sharedName, name)(evData)
  }

  // TODO: [DATASETS] "createOneShotIterator".

  /** Returns the data types corresponding to each element of this dataset, matching the structure of the elements. */
  def outputDataTypes: evData.D

  /** Returns the shapes corresponding to each element of this dataset, matching the structure of the elements. */
  def outputShapes: evData.S

  /** Returns a sequence of data types that correspond to the flattened data types of the nested tensor structure
    * of the elements of this dataset. */
  private[data] def flatOutputDataTypes: Seq[DataType[Any]] = {
    evData.dataTypes(outputDataTypes)
  }

  /** Returns a sequence of [[Shape]]s that correspond to the flattened shapes of the nested tensor structure of the
    * elements of this dataset. */
  private[data] def flatOutputShapes: Seq[Shape] = {
    evData.shapes(outputShapes)
  }

  //region Transformations

  // TODO: [DATA] Add dynamic version (i.e., passing in `Output`s) for the `repeat`, `shuffle`, `interleave`, `paddedBatch` datasets.

  /** Creates a new dataset that repeats this dataset a specified number of times. If the provided number of times to
    * repeat is set to `-1` (the default), then the dataset is repeated indefinitely.
    *
    * @param  count Number of times to repeat the input dataset. A value of `-1` corresponds to repeating it
    *               indefinitely.
    * @return Created dataset.
    */
  def repeat(
      count: Long = -1L
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Repeat"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val c = Op.nameScope(name)(Basic.constant(count))
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "RepeatDataset",
          name = name,
          input = (outer.createHandle(), c)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that produces the elements of this dataset, in random order.
    *
    * @param  bufferSize Buffer size, meaning the number of output elements to buffer before shuffling them.
    * @param  seed       Seed value for the random number generator. If not provided, a random seed is used.
    * @return Created dataset.
    */
  def shuffle(
      bufferSize: Long,
      seed: Option[Int] = None
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Shuffle"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val (bs, s1, s2) = Op.nameScope(name) {
          val bs = Basic.constant(bufferSize)
          val (s1, s2) = Dataset.randomSeeds(seed, s"$name/RandomSeeds")
          (bs, s1, s2)
        }
        Op.Builder[(Output[Long], Output[Long], Output[Long], Output[Long]), Output[Long]](
          opType = "ShuffleDataset",
          name = name,
          input = (outer.createHandle(), bs, s1, s2)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that takes at most the provided number of elements from a dataset, forming a new dataset.
    * If the provided number is `-1`, then all of the elements are taken.
    *
    * The op has similar semantics to the built-in Scala collections `take` function.
    *
    * @param  count Number of elements to take.
    * @return Created dataset.
    */
  def take(
      count: Long
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Take"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val c = Basic.constant(count, name = s"$name/Count")
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "TakeDataset",
          name = name,
          input = (outer.createHandle(), c)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that takes at most the provided number of elements from a dataset, forming a new dataset.
    * If the provided number is `-1`, then all of the elements are taken.
    *
    * The op has similar semantics to the built-in Scala collections `take` function.
    *
    * @param  count Number of elements to take.
    * @return Created dataset.
    */
  def take(
      count: Output[Long]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Take"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "TakeDataset",
          name = name,
          input = (outer.createHandle(), count)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that drops at most the provided number of elements from a dataset, forming a new dataset.
    * If the provided number is `-1`, then all of the elements are dropped.
    *
    * The op has similar semantics to the built-in Scala collections `drop` function.
    *
    * @param  count Number of elements to take.
    * @return Created dataset.
    */
  def drop(
      count: Long
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Drop"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val c = Basic.constant(count, name = s"$name/Count")
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "SkipDataset",
          name = name,
          input = (outer.createHandle(), c)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that drops at most the provided number of elements from a dataset, forming a new dataset.
    * If the provided number is `-1`, then all of the elements are dropped.
    *
    * The op has similar semantics to the built-in Scala collections `drop` function.
    *
    * @param  count Number of elements to take.
    * @return Created dataset.
    */
  def drop(
      count: Output[Long]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Drop"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "SkipDataset",
          name = name,
          input = (outer.createHandle(), count)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset by filtering the elements of another dataset using the provided predicate function. The
    * predicate function must return a scalar boolean tensor.
    *
    * The op has similar semantics to the built-in Scala collections `filter` function.
    *
    * @param  predicate Filter predicate function.
    * @return Created dataset.
    */
  def filter(
      predicate: T => Output[Boolean],
      name: String = s"${this.name}/Filter"
  )(implicit evFunctionArg: Function.ArgType[T]): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      override implicit val evData: SupportedData[T] = outer.evData

      private lazy val instantiatedPredicateFunction = {
        Function(s"$name/Predicate", predicate).instantiate(
          outer.flatOutputDataTypes, outer.flatOutputShapes,
          appendHashToName = true)
      }

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Seq[Output[Any]]), Output[Long]](
          opType = "FilterDataset",
          name = name,
          input = (outer.createHandle(), instantiatedPredicateFunction.extraInputs)
        ).setAttribute("predicate", instantiatedPredicateFunction)
            .setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset by mapping a function across all elements of this dataset.
    *
    * The op has similar semantics to the built-in Scala collections `map` function.
    *
    * @param  function         Mapping function.
    * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
    *                          sequentially.
    * @tparam R                Tensor type for the resulting dataset (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def map[R](
      function: T => R,
      numParallelCalls: Int = 1,
      name: String = s"${this.name}/Map"
  )(implicit
      evFunctionArgT: Function.ArgType[T],
      evFunctionArgR: Function.ArgType[R],
      evData: SupportedData[R]
  ): Dataset[R] = {
    val providedName = name
    val providedEvData = evData
    new Dataset[R] {
      override val name: String = providedName

      override implicit val evData: SupportedData[R] = providedEvData

      private lazy val instantiatedFunction = {
        Function(s"$name/Function", function).instantiate(
          outer.flatOutputDataTypes, outer.flatOutputShapes,
          appendHashToName = true)
      }

      override def createHandle(): Output[Long] = {
        if (numParallelCalls > 1) {
          Op.Builder[(Output[Long], Seq[Output[Any]], Output[Int]), Output[Long]](
            opType = "MapDataset",
            name = name,
            input = (
                outer.createHandle(),
                instantiatedFunction.extraInputs,
                Basic.constant(numParallelCalls, name = s"$name/NumParallelCalls"))
          ).setAttribute("f", instantiatedFunction)
              .setAttribute("output_types", flatOutputDataTypes.toArray)
              .setAttribute("output_shapes", flatOutputShapes.toArray)
              .build().output
        } else {
          Op.Builder[(Output[Long], Seq[Output[Any]]), Output[Long]](
            opType = "MapDataset",
            name = name,
            input = (outer.createHandle(), instantiatedFunction.extraInputs)
          ).setAttribute("f", instantiatedFunction)
              .setAttribute("output_types", flatOutputDataTypes.toArray)
              .setAttribute("output_shapes", flatOutputShapes.toArray)
              .build().output
        }
      }

      private lazy val (_outputDataTypes, _outputShapes): (evData.D, evData.S) = {
        val dataType = evData.dataType(instantiatedFunction.dummyOutputs)
        (evData.decodeDataType(dataType, instantiatedFunction.outputDataTypes)._1,
            evData.decodeShape(dataType, instantiatedFunction.outputShapes)._1)
      }

      override def outputDataTypes: evData.D = _outputDataTypes
      override def outputShapes: evData.S = _outputShapes
    }
  }

  /** Creates a new dataset by mapping a function across all elements of this dataset and then flattening the result.
    *
    * The op has similar semantics to the built-in Scala collections `flatMap` function.
    *
    * @param  function Mapping function.
    * @tparam R        Tensor type for the resulting dataset elements (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def flatMap[R](
      function: T => Dataset[R],
      name: String = s"${this.name}/FlatMap"
  )(implicit
      evFunctionArgT: Function.ArgType[T],
      evFunctionArgR: Function.ArgType[R],
      evData: SupportedData[R]
  ): Dataset[R] = {
    val providedName = name
    val providedEvData = evData
    new Dataset[R] {
      override val name: String = providedName

      override implicit val evData: SupportedData[R] = providedEvData

      private lazy val instantiatedFunction = {
        Function(s"$name/Function", function).instantiate(
          outer.flatOutputDataTypes, outer.flatOutputShapes,
          appendHashToName = true)
      }

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Seq[Output[Any]]), Output[Long]](
          opType = "FlatMapDataset",
          name = name,
          input = (outer.createHandle(), instantiatedFunction.extraInputs)
        ).setAttribute("f", instantiatedFunction)
            .setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        instantiatedFunction.dummyOutputs.outputDataTypes.asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        instantiatedFunction.dummyOutputs.outputShapes.asInstanceOf[evData.S]
      }
    }
  }

  /** Creates a new dataset by mapping a function across all elements of this dataset and then interleaving the
    * result.
    *
    * For example, you can use `Dataset.interleave()` to process many input files concurrently:
    * {{{
    *        // Preprocess 4 files concurrently, and interleave blocks of 16 records from each file.
    *        val filenames = Tensor("/var/data/file1.txt", "/var/data/file2.txt", ...)
    *        val dataset = TensorSlicesDataset(filenames).interleave(
    *          d => TextLinesDataset(d).map(parseFn), cycleLength = 4, blockLength = 16)
    * }}}
    *
    * The `cycleLength` and `blockLength` arguments control the order in which elements are produced.
    * `cycleLength` controls the number of input elements that are processed concurrently. If you set
    * `cycleLength` to `1`, this transformation will handle one input element at a time, and produce identical
    * results to `flatMap(...)`. In general, this transformation will apply `function` to `cycleLength` input
    * elements, open iterators on the returned dataset objects, and cycle through them producing `blockLength`
    * consecutive elements from each iterator, and consuming the next input element each time it reaches the end
    * of an iterator.
    *
    * For example:
    * {{{
    *        // The following examples use `{ ... }` to represent the contents of a dataset,
    *        // and new lines indicate "block" boundaries.
    *        a = { 1, 2, 3, 4, 5 }
    *        a.interleave(d => TensorDataset(d).repeat(6), cycleLength = 2, blockLength = 4) == {
    *          1, 1, 1, 1,
    *          2, 2, 2, 2,
    *          1, 1,
    *          2, 2,
    *          3, 3, 3, 3,
    *          4, 4, 4, 4,
    *          3, 3,
    *          4, 4,
    *          5, 5, 5, 5,
    *          5, 5
    *        }
    * }}}
    *
    * Note that the order of elements yielded by this transformation is deterministic, as long as `function` is a
    * pure function. If `function` contains any stateful operations, the order in which that state is accessed is
    * undefined.
    *
    * The op has similar semantics to the built-in Scala collections `flatMap` function.
    *
    * @param  function         Mapping function.
    * @param  cycleLength      Number of elements from this dataset that will be processed concurrently.
    * @param  blockLength      Number of consecutive elements to produce from each input element before cycling to
    *                          another input element.
    * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
    *                          sequentially.
    * @tparam R                Tensor type for the resulting dataset elements (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def interleave[R](
      function: T => Dataset[R],
      cycleLength: Long,
      blockLength: Long = 1L,
      numParallelCalls: Int = 1,
      name: String = "Interleave"
  )(implicit
      evFunctionArgT: Function.ArgType[T],
      evFunctionArgR: Function.ArgType[R],
      evData: SupportedData[R]
  ): Dataset[R] = {
    val providedName = name
    val providedEvData = evData
    new Dataset[R] {
      override val name: String = s"${outer.name}/$providedName"

      override implicit val evData: SupportedData[R] = providedEvData

      private lazy val instantiatedFunction = {
        Function(s"$name/Function", function).instantiate(
          outer.flatOutputDataTypes, outer.flatOutputShapes,
          appendHashToName = true)
      }

      override def createHandle(): Output[Long] = {
        if (numParallelCalls > 1) {
          Op.Builder[(Output[Long], Seq[Output[Any]], Output[Long], Output[Long], Output[Long]), Output[Long]](
            opType = "ParallelInterleaveDatasetV2",
            name = name,
            input = (
                outer.createHandle(),
                instantiatedFunction.extraInputs,
                Basic.constant(cycleLength, name = s"$name/CycleLength"),
                Basic.constant(blockLength, name = s"$name/BlockLength"),
                Basic.constant(numParallelCalls.toLong, name = s"$name/NumParallelCalls"))
          ).setAttribute("f", instantiatedFunction)
              .setAttribute("output_types", flatOutputDataTypes.toArray)
              .setAttribute("output_shapes", flatOutputShapes.toArray)
              .build().output
        } else {
          Op.Builder[(Output[Long], Seq[Output[Any]], Output[Long], Output[Long]), Output[Long]](
            opType = "InterleaveDataset",
            name = name,
            input = (
                outer.createHandle(),
                instantiatedFunction.extraInputs,
                Basic.constant(cycleLength, name = s"$name/CycleLength"),
                Basic.constant(blockLength, name = s"$name/BlockLength"))
          ).setAttribute("f", instantiatedFunction)
              .setAttribute("output_types", flatOutputDataTypes.toArray)
              .setAttribute("output_shapes", flatOutputShapes.toArray)
              .build().output
        }
      }

      override def outputDataTypes: evData.D = {
        instantiatedFunction.dummyOutputs.outputDataTypes.asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        instantiatedFunction.dummyOutputs.outputShapes.asInstanceOf[evData.S]
      }
    }
  }

  /** Creates a new dataset by applying transformation that groups windows of elements by a key and then reduces them.
    *
    * This transformation maps each consecutive element in a dataset to a key using `keyFn` and groups the elements by
    * key. It then applies `reduceFn` to at most `windowSizeFn(key)` elements matching the same key. All except the
    * final window for each key will contain `windowSizeFn(key)` elements; the final window may be smaller.
    *
    * @param  keyFn        Function used to compute the grouping key.
    * @param  reduceFn     Function used to reduce each group.
    * @param  windowSizeFn Function used to compute the maximum window size per key.
    * @return Created dataset.
    */
  def groupByWindow(
      keyFn: T => Output[Long],
      reduceFn: ((Output[Long], Dataset[T])) => Dataset[T],
      windowSizeFn: Output[Long] => Output[Long],
      name: String = s"${this.name}/GroupByWindow"
  )(implicit
      evFunctionArgOutputLong: Function.ArgType[Output[Long]],
      evFunctionArgT: Function.ArgType[T]
  ): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      override implicit val evData: SupportedData[T] = outer.evData

      private lazy val instantiatedKeyFunction = {
        Function(s"$name/KeyFunction", keyFn).instantiate(
          outer.flatOutputDataTypes, outer.flatOutputShapes,
          appendHashToName = true)
      }

      private lazy val instantiatedReduceFunction = {
        Function(s"$name/ReduceFunction", reduceFn).instantiate(
          Seq(INT64, VARIANT), Seq(Shape.scalar(), Shape.scalar()),
          input = Some((null, outer)), appendHashToName = true)
      }

      private lazy val instantiatedWindowSizeFunction = {
        Function(s"$name/WindowSizeFunction", windowSizeFn).instantiate(
          Seq(INT64), Seq(Shape.scalar()), appendHashToName = true)
      }

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Seq[Output[Any]], Seq[Output[Any]], Seq[Output[Any]]), Output[Long]](
          opType = "GroupByWindowDataset",
          name = name,
          input = (
              outer.createHandle(),
              instantiatedKeyFunction.extraInputs,
              instantiatedReduceFunction.extraInputs,
              instantiatedWindowSizeFunction.extraInputs)
        ).setAttribute("key_func", instantiatedKeyFunction)
            .setAttribute("reduce_func", instantiatedReduceFunction)
            .setAttribute("window_size_func", instantiatedWindowSizeFunction)
            .setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        instantiatedReduceFunction.dummyOutputs.outputDataTypes.asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        instantiatedReduceFunction.dummyOutputs.outputShapes.asInstanceOf[evData.S]
      }
    }
  }

  /** Creates a new dataset that combines consecutive elements of this dataset into batches.
    *
    * @param  batchSize Batch size.
    * @return Created dataset.
    */
  def batch(
      batchSize: Long
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Batch"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val bs = Op.nameScope(name)(Basic.constant(batchSize))
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "BatchDataset",
          name = name,
          input = (outer.createHandle(), bs)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        outer.outputDataTypes.asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        evData.decodeShape(
          outputDataTypes, outer.flatOutputShapes.map(Shape(-1) ++ _)
        )._1
      }
    }
  }

  /** Creates a new dataset that combines consecutive elements of this dataset into batches.
    *
    * @param  batchSize Batch size.
    * @return Created dataset.
    */
  def batch(
      batchSize: Output[Long]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Batch"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "BatchDataset",
          name = name,
          input = (outer.createHandle(), batchSize)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]

      override def outputShapes: evData.S = {
        evData.decodeShape(
          outputDataTypes, outer.flatOutputShapes.map(Shape(-1) ++ _)
        )._1
      }
    }
  }

  /** Creates a new dataset that combines consecutive elements of this dataset into padded batches.
    *
    * Like the dataset `batch` op, this op combines multiple consecutive elements of a dataset, which might have
    * different shapes, into a single element. The tensors in the resulting element have an additional outer
    * dimension, and are padded to the respective shape in `paddedShapes`.
    *
    * This transformation combines multiple consecutive elements of the input dataset into a single element. Like the
    * dataset `batch` op, the tensors in the resulting element have an additional outer dimension, which will be
    * `batchSize` for all but the last element, and `N % batchSize` for the last element, where `N` is the number of
    * elements in this dataset. Unlike the `batch` op, the elements may have different shapes for some of their
    * components, and this transformation will pad each component to the respective shape in `paddedShapes`. The
    * `paddedShapes` argument determines the resulting shape for each dimension of each component in an output
    * element:
    *
    *   - If the dimension is a constant, then the component will be padded out to that length along that dimension.
    *   - If the dimension is unknown, then the component will be padded out to the maximum length of all elements
    *     along that dimension.
    *
    * '''NOTE:''' If the number of elements in this dataset (`N`) is not an exact multiple of `batchSize`, the final
    * batch may contain smaller tensors with shape `N % batchSize` in the batch dimension. If your program depends on
    * the batches having the same shape, consider using the `paddedBatchAndDropRemainder` transformation instead.
    *
    * See also the `denseToSparseBatch` op, which combines elements that may have different shapes
    * into a sparse tensor.
    * 
    * @param  batchSize     Batch size to use.
    * @param  paddedShapes  Shape to which the respective component of each input element should be padded prior to
    *                       batching. Any unknown dimensions (e.g., equal to `-1`) will be padded to the maximum size of
    *                       that dimension in each batch.
    * @param  paddingValues Scalar tensor structure representing the padding values to use for the respective components.
    *                       Defaults to zero for numeric types and the empty string for string types.
    * @param  name          Name for this dataset.
    * @return Created dataset.
    */
  def paddedBatch[TT, S](
      batchSize: Long,
      paddedShapes: S,
      paddingValues: TT = null.asInstanceOf[TT],
      name: String = s"${this.name}/PaddedBatch"
  )(implicit
      evTensorToOutput: TensorToOutput.Aux[TT, T],
      evData: SupportedData.Aux[T, _, S]
  ): Dataset[T] = {
    val providedEvData = evData
    new Dataset[T] {
      override val name: String = s"${outer.name}/PaddedBatch"

      override implicit val evData: SupportedData[T] = providedEvData

      private def flatPaddedShapes: Seq[Output[Long]] = {
        providedEvData.shapes(paddedShapes).map(_.toOutput(INT64))
      }

      private def flatPaddingValues: Seq[Output[Any]] = {
        if (paddingValues != null) {
          evTensorToOutput.tensors(paddingValues).map(Basic.constant(_))
        } else {
          flatOutputDataTypes.map(Basic.zeros(_, Shape()))
        }
      }

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Output[Long], Seq[Output[Long]], Seq[Output[Any]]), Output[Long]](
          opType = "PaddedBatchDataset",
          name = name,
          input = (
              outer.createHandle(),
              batchSize,
              Op.nameScope(s"$name/PaddedShapes")(flatPaddedShapes),
              Op.nameScope(s"$name/PaddingValues")(flatPaddingValues))
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]

      override def outputShapes: evData.S = {
        evData.decodeShape(
          outputDataTypes, outer.flatOutputShapes.map(Shape(-1) ++ _)
        )._1
      }
    }
  }

  /** Creates a new dataset by asynchronously prefetching elements from this dataset.
    *
    * @param  bufferSize Number of elements to prefetch.
    * @return Created dataset.
    */
  def prefetch(
      bufferSize: Long
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Prefetch"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val bs = Op.nameScope(name)(Basic.constant(bufferSize))
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "PrefetchDataset",
          name = name,
          input = (outer.createHandle(), bs)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that caches the elements of this dataset in the provided directory. If the provided
    * directory is an empty string, then the elements are cached in memory.
    *
    * @param  directory Directory to use for caching. If empty string, then the caching will happen in memory.
    * @return Created dataset.
    */
  def cache(
      directory: String
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Cache"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        val d = Op.nameScope(name)(Basic.constant(directory))
        Op.Builder[(Output[Long], Output[String]), Output[Long]](
          opType = "CacheDataset",
          name = name,
          input = (outer.createHandle(), d)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that caches the elements of this dataset in the provided directory. If the provided
    * directory is an empty string, then the elements are cached in memory.
    *
    * @param  directory Directory to use for caching. If empty string, then the caching will happen in memory.
    * @return Created dataset.
    */
  def cache(
      directory: Output[String]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Cache"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Output[String]), Output[Long]](
          opType = "CacheDataset",
          name = name,
          input = (outer.createHandle(), directory)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  /** Creates a new dataset that concatenates elements from this dataset with elements from the provided dataset.
    *
    * For example:
    * {{{
    *   // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *   a = { 1, 2, 3 }
    *   b = { 4, 5, 6, 7 }
    *   a.concatenate(b) ==> { 1, 2, 3, 4, 5, 6, 7 }
    *
    *   // The datasets to be concatenated should have the same nested structures and output types.
    *   c = { (8, 9), (10, 11), (12, 13) }
    *   d = { 14.0, 15.0, 16.0 }
    *   // a.concatenate(c) and a.concatenate(d) would result in exceptions being thrown.
    * }}}
    *
    * @param  other Dataset to concatenate with this dataset.
    * @return Created dataset.
    */
  def concatenateWith(
      other: Dataset[T],
      name: String = s"${this.name}/Concatenated"
  ): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      override implicit val evData: SupportedData[T] = outer.evData

      private lazy val mostSpecificFlattenedShapes: Seq[Shape] = {
        outer.flatOutputShapes.zip(other.flatOutputShapes).map(p => {
          Shape.fromSeq(p._1.asArray.zip(p._2.asArray).map {
            case (d1, d2) if d1 == d2 => d1
            case (d1, d2) if d1 == -1 => d2
            case (d1, d2) if d2 == -1 => d1
            case _ => -1
          })
        })
      }

      override def createHandle(): Output[Long] = {
        Op.Builder[(Output[Long], Output[Long]), Output[Long]](
          opType = "CacheDataset",
          name = name,
          input = (outer.createHandle(), other.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]

      override def outputShapes: evData.S = {
        evData.decodeShape(outputDataTypes, mostSpecificFlattenedShapes)._1
      }
    }
  }

  /** Creates a new dataset that zips this data set with another dataset.
    *
    * The op has similar semantics to the built-in Scala collections `zip` function.
    *
    * For example:
    * {{{
    *   // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *   a = { 1, 2, 3 }
    *   b = { 4, 5, 6 }
    *   c = { (7, 8), (9, 10), (11, 12) }
    *   d = { 13, 14 }
    *
    *   // The nested structure of the `datasets` argument determines the structure of elements in the resulting
    *   // dataset.
    *   a.zip(b) ==> { (1, 4), (2, 5), (3, 6) }
    *   b.zip(a) ==> { (4, 1), (5, 2), (6, 3) }
    *
    *   // The number of elements in the resulting dataset is the same as the size of the smallest provided dataset.
    *   a.zip(d) ==> { (1, 13), (2, 14) }
    * }}}
    *
    * @param  other Dataset to zip with this dataset.
    * @param  name  Name to use for the new dataset.
    * @return Created dataset.
    */
  def zip[R](
      other: Dataset[R],
      name: String = s"${this.name}/Zip"
  )(implicit
      evData: SupportedData[R]
  ): Dataset[(T, R)] = {
    implicit val evDataT: SupportedData[T] = this.evData
    val providedName = name
    new Dataset[(T, R)] {
      override val name: String = providedName

      override implicit val evData: SupportedData[(T, R)] = implicitly[SupportedData[(T, R)]]

      override def createHandle(): Output[Long] = {
        Op.Builder[Seq[Output[Long]], Output[Long]](
          opType = "ZipDataset",
          name = name,
          input = Seq(outer.createHandle(), other.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        (outer.outputDataTypes, other.outputDataTypes).asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        (outer.outputShapes, other.outputShapes).asInstanceOf[evData.S]
      }
    }
  }

  /** Creates a new dataset that zips this data set with two other datasets.
    *
    * The op has similar semantics to the built-in Scala collections `zip` function.
    *
    * For example:
    * {{{
    *   // NOTE: The following examples use `{ ... }` to represent the contents of a dataset.
    *   a = { 1, 2, 3 }
    *   b = { 4, 5, 6 }
    *   c = { (7, 8), (9, 10), (11, 12) }
    *   d = { 13, 14 }
    *
    *   // The `datasets` argument may contain an arbitrary number of datasets.
    *     a.zip3(b, c) ==> { (1, 4, (7, 8)), (2, 5, (9, 10)), (3, 6, (11, 12)) }
    * }}}
    *
    * @param  other1 First dataset to zip with this dataset.
    * @param  other2 Second dataset to zip with this dataset.
    * @param  name   Name to use for the new dataset.
    * @return Created dataset.
    */
  def zip3[R1, R2](
      other1: Dataset[R1],
      other2: Dataset[R2],
      name: String = s"${this.name}/Zip"
  )(implicit
      evDataR1: SupportedData[R1],
      evDataR2: SupportedData[R2]
  ): Dataset[(T, R1, R2)] = {
    implicit val evDataT: SupportedData[T] = this.evData
    val providedName = name
    new Dataset[(T, R1, R2)] {
      override val name: String = providedName

      override implicit val evData: SupportedData[(T, R1, R2)] = implicitly[SupportedData[(T, R1, R2)]]

      override def createHandle(): Output[Long] = {
        Op.Builder[Seq[Output[Long]], Output[Long]](
          opType = "ZipDataset",
          name = name,
          input = Seq(outer.createHandle(), other1.createHandle(), other2.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        (outer.outputDataTypes, other1.outputDataTypes, other2.outputDataTypes).asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        (outer.outputShapes, other1.outputShapes, other2.outputShapes).asInstanceOf[evData.S]
      }
    }
  }

  /** Creates a new dataset that zips this data set with multiple other datasets.
    *
    * The op has similar semantics to the built-in Scala collections `zip` function.
    *
    * @param  others Datasets to zip with this dataset.
    * @param  name   Name to use for the new dataset.
    * @return Created dataset.
    */
  def zipMultiple(
      others: Seq[Dataset[T]],
      name: String = s"${this.name}/Zip"
  ): Dataset[Seq[T]] = {
    implicit val evDataT: SupportedData[T] = this.evData
    val providedName = name
    new Dataset[Seq[T]] {
      override val name: String = providedName

      override implicit val evData: SupportedData[Seq[T]] = implicitly[SupportedData[Seq[T]]]

      override def createHandle(): Output[Long] = {
        Op.Builder[Seq[Output[Long]], Output[Long]](
          opType = "ZipDataset",
          name = name,
          input = outer.createHandle() +: others.map(_.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = {
        (outer.outputDataTypes +: others.map(_.outputDataTypes)).asInstanceOf[evData.D]
      }

      override def outputShapes: evData.S = {
        (outer.outputShapes +: others.map(_.outputShapes)).asInstanceOf[evData.S]
      }
    }
  }

  /** Creates a new dataset from this dataset, that silently ignores any errors, and contains the same elements.
    *
    * Use this transformation to produce a dataset that contains the same elements as the input, but silently drops
    * any elements that caused an error. For example:
    * {{{
    *   dataset = datasetFromSlices(Tensor(1.0, 2.0, 0.0, 4.0))
    *
    *   // Computing `checkNumerics(1.0 / 0.0)` will raise an [[IllegalArgumentException]].
    *   dataset = dataset.map(x => checkNumerics(1.0 / x, "error"))
    *
    *   // Using `ignoreErrors` will drop the elements that cause errors.
    *   dataset = dataset.ignoreErrors()  // ==> { 1.0, 0.5, 0.2 }
    * }}}
    * 
    * @return Created dataset.
    */
  def ignoreErrors(): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/IgnoreErrors"

      override implicit val evData: SupportedData[T] = outer.evData

      override def createHandle(): Output[Long] = {
        Op.Builder[Output[Long], Output[Long]](
          opType = "IgnoreErrorsDataset",
          name = name,
          input = outer.createHandle()
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes: evData.D = outer.outputDataTypes.asInstanceOf[evData.D]
      override def outputShapes: evData.S = outer.outputShapes.asInstanceOf[evData.S]
    }
  }

  //endregion Transformations

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
    * @throws InvalidArgumentException If `shardIndex >= numShards`.
    */
  @throws[InvalidArgumentException]
  def shard(
      numShards: Long,
      shardIndex: Long
  )(implicit evFunctionArg: Function.ArgType[T]): Dataset[T] = {
    if (shardIndex >= numShards)
      throw InvalidArgumentException(s"'index' (= $shardIndex) must be smaller than 'numShards' (= $numShards).")
    if (numShards == 1) {
      this
    } else {
      this.zip(Data.datasetFromRange(0L, Long.MaxValue))
          .filter(t => Math.equal(Math.mod(t._2, numShards), shardIndex))
          .map(o => o._1)
    }
  }

  /** Applies a transformation function to this dataset.
    *
    * `transform()` enables chaining of custom dataset transformations, which are represented as functions that take one
    * dataset argument and return a transformed dataset.
    *
    * @param  transformFn Dataset transformation function.
    * @return Transformed dataset.
    */
  def transform[R](transformFn: Dataset[T] => Dataset[R])(implicit
      evDataR: SupportedData[R],
      evFunctionInputT: Function.ArgType[R]
  ): Dataset[R] = {
    transformFn(this)
  }

  override def toString: String = {
    "Dataset[" +
        s"outputDataTypes = ${evData.dataTypeToString(outputDataTypes)}, " +
        s"outputShapes = ${evData.shapeToString(outputShapes)}]"
  }
}

object Dataset {
  //region Helpers

  /** Returns the local random seeds an op should use, given an optionally provided op-specific seed.
    *
    * @param  seed Optionally provided op-specific seed.
    * @param  name Name prefix for all created ops.
    * @return Local random seeds to use.
    */
  private[data] def randomSeeds(
      seed: Option[Int] = None,
      name: String = "RandomSeeds"
  ): (Output[Long], Output[Long]) = {
    val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
    val seed1 = graphSeed.map(_.toLong).getOrElse(0L)
    val seed2 = opSeed match {
      case None => 0L
      case Some(s) =>
        val seed2 = s.toLong
        if (seed1 == 0L && seed2 == 0L) {
          Long.MaxValue
        } else {
          seed2
        }
    }
    Op.nameScope(name) {
      (Basic.constant(seed1, name = "Seed1"),
          Basic.constant(seed2, name = "Seed2"))
    }
  }

  //endregion Helpers
}