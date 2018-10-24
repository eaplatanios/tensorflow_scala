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
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor

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
abstract class Dataset[T: OutputStructure] { outer =>
  val name: String

  /** Creates a `VARIANT` scalar tensor representing this dataset. This function adds ops to the current graph, that
    * create the dataset resource. */
  def createHandle[D, S]()(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): Output[Variant]

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
  def createInitializableIterator[D, S](
      sharedName: String = "",
      name: String = "InitializableDatasetIterator"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): InitializableDatasetIterator[T] = {
    DatasetIterator.fromDataset(dataset = this, sharedName, name)
  }

  // TODO: [DATASETS] "createOneShotIterator".

  /** Returns the data types corresponding to each element of this dataset, matching the structure of the elements. */
  def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D

  /** Returns the shapes corresponding to each element of this dataset, matching the structure of the elements. */
  def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S

  /** Returns a sequence of data types that correspond to the flattened data types of the nested tensor structure
    * of the elements of this dataset. */
  private[data] def flatOutputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): Seq[DataType[Any]] = {
    ev.dataTypeStructure.dataTypes(outputDataTypes)
  }

  /** Returns a sequence of [[Shape]]s that correspond to the flattened shapes of the nested tensor structure of the
    * elements of this dataset. */
  private[data] def flatOutputShapes[S](implicit ev: OutputToShape.Aux[T, S]): Seq[Shape] = {
    ev.shapeStructure.shapes(outputShapes)
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
  def repeat(count: Long = -1L): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Repeat"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val c = Op.nameScope(name)(Basic.constant(count))
        Op.Builder[(Output[Variant], Output[Long]), Output[Variant]](
          opType = "RepeatDataset",
          name = name,
          input = (outer.createHandle(), c)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
    }
  }

  /** Creates a new dataset that produces the elements of this dataset, in random order.
    *
    * @param  bufferSize Buffer size, meaning the number of output elements to buffer before shuffling them.
    * @param  seed       Seed value for the random number generator. If not provided, a random seed is used.
    * @return Created dataset.
    */
  def shuffle(bufferSize: Long, seed: Option[Int] = None): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Shuffle"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val (bs, s1, s2) = Op.nameScope(name) {
          val bs = Basic.constant(bufferSize)
          val (s1, s2) = Dataset.randomSeeds(seed, s"$name/RandomSeeds")
          (bs, s1, s2)
        }
        Op.Builder[(Output[Variant], Output[Long], Output[Long], Output[Long]), Output[Variant]](
          opType = "ShuffleDataset",
          name = name,
          input = (outer.createHandle(), bs, s1, s2)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
  def take(count: Long): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Take"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val c = Basic.constant(count, name = s"$name/Count")
        Op.Builder[(Output[Variant], Output[Long]), Output[Variant]](
          opType = "TakeDataset",
          name = name,
          input = (outer.createHandle(), c)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
  def take(count: Output[Long]): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Take"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[(Output[Variant], Output[Long]), Output[Variant]](
          opType = "TakeDataset",
          name = name,
          input = (outer.createHandle(), count)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
  def drop(count: Long): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Drop"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val c = Basic.constant(count, name = s"$name/Count")
        Op.Builder[(Output[Variant], Output[Long]), Output[Variant]](
          opType = "SkipDataset",
          name = name,
          input = (outer.createHandle(), c)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
  def drop(count: Output[Long]): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Drop"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[(Output[Variant], Output[Long]), Output[Variant]](
          opType = "SkipDataset",
          name = name,
          input = (outer.createHandle(), count)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
  ): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val instantiatedPredicateFunction = Function(s"$name/Predicate", predicate)
            .instantiate(
              inputDataType = outer.outputDataTypes,
              inputShape = Some(outer.outputShapes),
              appendHashToName = true
            )(evOutputToDataType, evOutputToShape)

        Op.Builder[(Output[Variant], Seq[Output[Any]]), Output[Variant]](
          opType = "FilterDataset",
          name = name,
          input = (outer.createHandle(), instantiatedPredicateFunction.extraInputs)
        ).setAttribute("predicate", instantiatedPredicateFunction)
            .setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
    }
  }

  /** Creates a new dataset by mapping a function across all elements of this dataset.
    *
    * The op has similar semantics to the built-in Scala collections `map` function.
    *
    * @param  function         Mapping function.
    * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
    *                          sequentially.
    * @tparam R Tensor type for the resulting dataset (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def map[D, S, R: OutputStructure](
      function: T => R,
      numParallelCalls: Int = 1,
      name: String = s"${this.name}/Map"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S]
  ): Dataset[R] = {
    val providedName = name
    new Dataset[R] {
      override val name: String = providedName

      private var instantiatedFunction: Option[InstantiatedFunction[T, R]] = None

      private def initializeInstantiatedFunction(): InstantiatedFunction[T, R] = {
        if (instantiatedFunction.isEmpty)
          instantiatedFunction = Some(
            Function(s"$name/Function", function).instantiate(
              inputDataType = outer.outputDataTypes,
              inputShape = Some(outer.outputShapes),
              appendHashToName = true)(evOutputToDataTypeT, evOutputToShapeT))
        instantiatedFunction.get
      }

      override def createHandle[RD, RS]()(implicit
          evOutputToDataType: OutputToDataType.Aux[R, RD],
          evOutputToShape: OutputToShape.Aux[R, RS]
      ): Output[Variant] = {
        val instantiatedFunction = this.instantiatedFunction.getOrElse(initializeInstantiatedFunction())
        if (numParallelCalls > 1) {
          Op.Builder[(Output[Variant], Seq[Output[Any]], Output[Int]), Output[Variant]](
            opType = "ParallelMapDataset",
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
          Op.Builder[(Output[Variant], Seq[Output[Any]]), Output[Variant]](
            opType = "MapDataset",
            name = name,
            input = (outer.createHandle(), instantiatedFunction.extraInputs)
          ).setAttribute("f", instantiatedFunction)
              .setAttribute("output_types", flatOutputDataTypes.toArray)
              .setAttribute("output_shapes", flatOutputShapes.toArray)
              .build().output
        }
      }

      override def outputDataTypes[RD](implicit ev: OutputToDataType.Aux[R, RD]): RD = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction()).outputDataTypes(ev)
      }

      override def outputShapes[RS](implicit ev: OutputToShape.Aux[R, RS]): RS = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction()).outputShapes(ev)
      }
    }
  }

  /** Creates a new dataset by mapping a function across all elements of this dataset and batching the resulting
    * elements.
    *
    * The op has similar semantics to the built-in Scala collections `map` function.
    *
    * @param  function         Mapping function.
    * @param  batchSize        Batch size to use.
    * @param  numParallelCalls Number elements to process in parallel. If not specified, elements will be processed
    *                          sequentially.
    * @param  dropRemainder    Boolean indicating whether to drop the last batch in the dataset if it's size is less
    *                          than `batchSize`.
    * @tparam R Tensor type for the resulting dataset (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def mapAndBatch[D, S, R: OutputStructure](
      function: T => R,
      batchSize: Long,
      numParallelCalls: Long = 1L,
      dropRemainder: Boolean = false,
      name: String = s"${this.name}/Map"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S]
  ): Dataset[R] = {
    val providedName = name
    new Dataset[R] {
      override val name: String = providedName

      private var instantiatedFunction: Option[InstantiatedFunction[T, R]] = None

      private def initializeInstantiatedFunction(): InstantiatedFunction[T, R] = {
        if (instantiatedFunction.isEmpty)
          instantiatedFunction = Some(
            Function(s"$name/Function", function).instantiate(
              inputDataType = outer.outputDataTypes,
              inputShape = Some(outer.outputShapes),
              appendHashToName = true))
        instantiatedFunction.get
      }

      override def createHandle[RD, RS]()(implicit
          evOutputToDataType: OutputToDataType.Aux[R, RD],
          evOutputToShape: OutputToShape.Aux[R, RS]
      ): Output[Variant] = {
        val instantiatedFunction = this.instantiatedFunction.getOrElse(initializeInstantiatedFunction())
        val bs = Op.nameScope(s"$name/BatchSize")(Basic.constant(batchSize))
        val dr = Op.nameScope(s"$name/DropRemainder")(Basic.constant(dropRemainder))
        Op.Builder[(Output[Variant], Seq[Output[Any]], Output[Long], Output[Long], Output[Boolean]), Output[Variant]](
          opType = "MapAndBatchDatasetV2",
          name = name,
          input = (
              outer.createHandle(),
              instantiatedFunction.extraInputs,
              Basic.constant(batchSize, name = s"$name/BatchSize"),
              Basic.constant(numParallelCalls, name = s"$name/NumParallelCalls"),
              Basic.constant(dropRemainder, name = s"$name/DropRemainder"))
        ).setAttribute("f", instantiatedFunction)
            .setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[RD](implicit ev: OutputToDataType.Aux[R, RD]): RD = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction()).outputDataTypes(ev)
      }

      override def outputShapes[RS](implicit ev: OutputToShape.Aux[R, RS]): RS = {
        val functionOutputShapes = instantiatedFunction.getOrElse(initializeInstantiatedFunction()).outputShapes(ev)
        ev.shapeStructure.decodeShape(
          functionOutputShapes,
          ev.shapeStructure.shapes(functionOutputShapes).map(Shape(-1) ++ _)
        )._1
      }
    }
  }

  /** Creates a new dataset by mapping a function across all elements of this dataset and then flattening the result.
    *
    * The op has similar semantics to the built-in Scala collections `flatMap` function.
    *
    * @param  function Mapping function.
    * @tparam R Tensor type for the resulting dataset elements (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def flatMap[D, S, R: OutputStructure, RD, RS](
      function: T => Dataset[R],
      name: String = s"${this.name}/FlatMap"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S],
      evOutputToDataType: OutputToDataType.Aux[R, RD],
      evOutputToShape: OutputToShape.Aux[R, RS]
  ): Dataset[R] = {
    val providedName = name
    new Dataset[R] {
      override val name: String = providedName

      private var instantiatedFunction: Option[InstantiatedFunction[T, Dataset[R]]] = None

      private def initializeInstantiatedFunction(): InstantiatedFunction[T, Dataset[R]] = {
        if (instantiatedFunction.isEmpty)
          instantiatedFunction = Some(
            Function(s"$name/Function", function).instantiate(
              inputDataType = outer.outputDataTypes,
              inputShape = Some(outer.outputShapes),
              appendHashToName = true))
        instantiatedFunction.get
      }

      override def createHandle[RD, RS]()(implicit
          evOutputToDataType: OutputToDataType.Aux[R, RD],
          evOutputToShape: OutputToShape.Aux[R, RS]
      ): Output[Variant] = {
        val instantiatedFunction = this.instantiatedFunction.getOrElse(initializeInstantiatedFunction())
        Op.Builder[(Output[Variant], Seq[Output[Any]]), Output[Variant]](
          opType = "FlatMapDataset",
          name = name,
          input = (outer.createHandle(), instantiatedFunction.extraInputs)
        ).setAttribute("f", instantiatedFunction)
            .setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[RD](implicit ev: OutputToDataType.Aux[R, RD]): RD = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction())._dummyOutput.outputDataTypes(ev)
      }

      override def outputShapes[RS](implicit ev: OutputToShape.Aux[R, RS]): RS = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction())._dummyOutput.outputShapes(ev)
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
    * @tparam R Tensor type for the resulting dataset elements (i.e., nested structure of outputs).
    * @return Created dataset.
    */
  def interleave[D, S, R: OutputStructure, RD, RS](
      function: T => Dataset[R],
      cycleLength: Long,
      blockLength: Long = 1L,
      numParallelCalls: Int = 1,
      name: String = "Interleave"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S],
      evOutputToDataType: OutputToDataType.Aux[R, RD],
      evOutputToShape: OutputToShape.Aux[R, RS]
  ): Dataset[R] = {
    val providedName = name
    new Dataset[R] {
      override val name: String = s"${outer.name}/$providedName"

      private var instantiatedFunction: Option[InstantiatedFunction[T, Dataset[R]]] = None

      private def initializeInstantiatedFunction(): InstantiatedFunction[T, Dataset[R]] = {
        if (instantiatedFunction.isEmpty)
          instantiatedFunction = Some(
            Function(s"$name/Function", function).instantiate(
              inputDataType = outer.outputDataTypes,
              inputShape = Some(outer.outputShapes),
              appendHashToName = true)(evOutputToDataTypeT, evOutputToShapeT))
        instantiatedFunction.get
      }

      override def createHandle[RD, RS]()(implicit
          evOutputToDataType: OutputToDataType.Aux[R, RD],
          evOutputToShape: OutputToShape.Aux[R, RS]
      ): Output[Variant] = {
        val instantiatedFunction = this.instantiatedFunction.getOrElse(initializeInstantiatedFunction())
        if (numParallelCalls > 1) {
          Op.Builder[(Output[Variant], Seq[Output[Any]], Output[Long], Output[Long], Output[Long]), Output[Variant]](
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
          Op.Builder[(Output[Variant], Seq[Output[Any]], Output[Long], Output[Long]), Output[Variant]](
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

      override def outputDataTypes[RD](implicit ev: OutputToDataType.Aux[R, RD]): RD = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction())._dummyOutput.outputDataTypes(ev)
      }

      override def outputShapes[RS](implicit ev: OutputToShape.Aux[R, RS]): RS = {
        instantiatedFunction.getOrElse(initializeInstantiatedFunction())._dummyOutput.outputShapes(ev)
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
  def groupByWindow[D, S](
      keyFn: T => Output[Long],
      reduceFn: ((Output[Long], Dataset[T])) => Dataset[T],
      windowSizeFn: Output[Long] => Output[Long],
      name: String = s"${this.name}/GroupByWindow"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S],
      // These implicit helpers is used for Scala 2.11 support.
      evOutputToDataType211Helper: OutputToDataType.Aux[(Output[Long], Dataset[T]), (DataType[Long], DataType[Variant])],
      evOutputToShape211Helper: OutputToShape.Aux[(Output[Long], Dataset[T]), (Shape, Shape)]
  ): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      private var instantiatedKeyFunction       : Option[InstantiatedFunction[T, Output[Long]]]                        = None
      private var instantiatedReduceFunction    : Option[InstantiatedFunction[(Output[Long], Dataset[T]), Dataset[T]]] = None
      private var instantiatedWindowSizeFunction: Option[InstantiatedFunction[Output[Long], Output[Long]]]             = None

      private def initializeInstantiatedKeyFunction(): InstantiatedFunction[T, Output[Long]] = {
        if (instantiatedKeyFunction.isEmpty)
          instantiatedKeyFunction = Some(
            Function(s"$name/KeyFunction", keyFn).instantiate(
              inputDataType = outer.outputDataTypes,
              inputShape = Some(outer.outputShapes),
              appendHashToName = true))
        instantiatedKeyFunction.get
      }

      private def initializeInstantiatedReduceFunction(): InstantiatedFunction[(Output[Long], Dataset[T]), Dataset[T]] = {
        if (instantiatedReduceFunction.isEmpty)
          instantiatedReduceFunction = Some(
            Function(s"$name/ReduceFunction", reduceFn).instantiate(
              inputDataType = (INT64.asInstanceOf[DataType[Long]], VARIANT.asInstanceOf[DataType[Variant]]),
              inputShape = Some((Shape(), Shape())),
              appendHashToName = true))
        instantiatedReduceFunction.get
      }

      private def initializeInstantiatedWindowSizeFunction(): InstantiatedFunction[Output[Long], Output[Long]] = {
        if (instantiatedWindowSizeFunction.isEmpty)
          instantiatedWindowSizeFunction = Some(
            Function(s"$name/WindowSizeFunction", windowSizeFn).instantiate(
              inputDataType = INT64,
              inputShape = Some(Shape()),
              appendHashToName = true))
        instantiatedWindowSizeFunction.get
      }

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val instantiatedKeyFunction = this.instantiatedKeyFunction.getOrElse(initializeInstantiatedKeyFunction())
        val instantiatedReduceFunction = this.instantiatedReduceFunction.getOrElse(initializeInstantiatedReduceFunction())
        val instantiatedWindowSizeFunction = this.instantiatedWindowSizeFunction.getOrElse(initializeInstantiatedWindowSizeFunction())
        Op.Builder[(Output[Variant], Seq[Output[Any]], Seq[Output[Any]], Seq[Output[Any]]), Output[Variant]](
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

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        instantiatedReduceFunction.getOrElse(initializeInstantiatedReduceFunction())._dummyOutput.outputDataTypes(ev)
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        instantiatedReduceFunction.getOrElse(initializeInstantiatedReduceFunction())._dummyOutput.outputShapes(ev)
      }
    }
  }

  /** Creates a new dataset that combines consecutive elements of this dataset into batches.
    *
    * @param  batchSize     Batch size.
    * @param  dropRemainder Boolean indicating whether to drop the last batch in the dataset if it's size is less than
    *                       `batchSize`.
    * @return Created dataset.
    */
  def batch[D, S](
      batchSize: Long,
      dropRemainder: Boolean = false
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S],
      evDataTypeToShape: DataTypeToShape.Aux[D, S]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Batch"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val bs = Op.nameScope(name)(Basic.constant(batchSize))
        Op.Builder[(Output[Variant], Output[Long], Output[Boolean]), Output[Variant]](
          opType = "BatchDatasetV2",
          name = name,
          input = (outer.createHandle(), bs, dropRemainder)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit evOutputToDataType: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit evOutputToShape: OutputToShape.Aux[T, S]): S = {
        evDataTypeToShape.decodeShape(
          outputDataTypes(evOutputToDataType),
          outer.flatOutputShapes.map(Shape(-1) ++ _)
        )._1.asInstanceOf[S]
      }
    }
  }

  /** Creates a new dataset that combines consecutive elements of this dataset into batches.
    *
    * @param  batchSize     Batch size.
    * @param  dropRemainder Boolean indicating whether to drop the last batch in the dataset if it's size is less than
    *                       `batchSize`.
    * @return Created dataset.
    */
  def dynamicBatch[D, S](
      batchSize: Output[Long],
      dropRemainder: Output[Boolean] = false
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S],
      evDataTypeToShape: DataTypeToShape.Aux[D, S]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/Batch"

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[(Output[Variant], Output[Long], Output[Boolean]), Output[Variant]](
          opType = "BatchDatasetV2",
          name = name,
          input = (outer.createHandle(), batchSize, dropRemainder)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit evOutputToDataType: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit evOutputToShape: OutputToShape.Aux[T, S]): S = {
        evDataTypeToShape.decodeShape(
          outputDataTypes(evOutputToDataType),
          outer.flatOutputShapes.map(Shape(-1) ++ _)
        )._1.asInstanceOf[S]
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
    * along that dimension.
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
    * @param  paddingValues Scalar tensor structure representing the padding values to use for the respective
    *                       components. Defaults to zero for numeric types and the empty string for string types.
    * @param  name          Name for this dataset.
    * @return Created dataset.
    */
  def paddedBatch[D, S, V](
      batchSize: Long,
      paddedShapes: S,
      paddingValues: Option[V] = None,
      name: String = s"${this.name}/PaddedBatch"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S],
      evOutputToTensor: OutputToTensor.Aux[T, V]
  ): Dataset[T] = {
    new Dataset[T] {
      override val name: String = s"${outer.name}/PaddedBatch"

      private def flatPaddedShapes: Seq[Output[Long]] = {
        evOutputToShape.shapeStructure.shapes(paddedShapes).map(_.toOutput)
      }

      private def flatPaddingValues: Seq[Output[Any]] = {
        paddingValues match {
          case Some(values) => evOutputToTensor.tensorStructure.tensors(values).map(Basic.constant(_))
          case None => flatOutputDataTypes.map(Basic.zeros[Any](_, Tensor.empty[Long]))
        }
      }

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[(Output[Variant], Output[Long], Seq[Output[Long]], Seq[Output[Any]]), Output[Variant]](
          opType = "PaddedBatchDataset",
          name = name,
          input = (
              outer.createHandle(),
              batchSize,
              Op.nameScope(s"$name/PaddedShapes")(flatPaddedShapes),
              Op.nameScope(s"$name/PaddingValues")(flatPaddingValues))
        ).setAttribute("Toutput_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit evOutputToDataType: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit evOutputToShape: OutputToShape.Aux[T, S]): S = {
        evOutputToShape.shapeStructure.decodeShape(
          outer.outputShapes,
          outer.flatOutputShapes.map(Shape(-1) ++ _)
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

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val bs = Op.nameScope(name)(Basic.constant(bufferSize))
        Op.Builder[(Output[Variant], Output[Long]), Output[Variant]](
          opType = "PrefetchDataset",
          name = name,
          input = (outer.createHandle(), bs)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        val d = Op.nameScope(name)(Basic.constant(directory))
        Op.Builder[(Output[Variant], Output[String]), Output[Variant]](
          opType = "CacheDataset",
          name = name,
          input = (outer.createHandle(), d)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[(Output[Variant], Output[String]), Output[Variant]](
          opType = "CacheDataset",
          name = name,
          input = (outer.createHandle(), directory)
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
  def concatenateWith[D, S](
      other: Dataset[T],
      name: String = s"${this.name}/Concatenated"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): Dataset[T] = {
    val providedName = name
    new Dataset[T] {
      override val name: String = providedName

      private var mostSpecificFlattenedShapes: Option[Seq[Shape]] = None

      private def initializeMostSpecificFlattenedShapes(): Seq[Shape] = {
        if (mostSpecificFlattenedShapes.isEmpty) {
          mostSpecificFlattenedShapes = Some(
            outer.flatOutputShapes.zip(other.flatOutputShapes).map(p => {
              Shape.fromSeq(p._1.asArray.zip(p._2.asArray).map {
                case (d1, d2) if d1 == d2 => d1
                case (d1, d2) if d1 == -1 => d2
                case (d1, d2) if d2 == -1 => d1
                case _ => -1
              })
            }))
        }
        mostSpecificFlattenedShapes.get
      }

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[(Output[Variant], Output[Variant]), Output[Variant]](
          opType = "CacheDataset",
          name = name,
          input = (outer.createHandle(), other.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit evOutputToDataType: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit evOutputToShape: OutputToShape.Aux[T, S]): S = {
        evOutputToShape.shapeStructure.decodeShape(
          outer.outputShapes,
          initializeMostSpecificFlattenedShapes()
        )._1
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
  def zip[D, S, R: OutputStructure, RD, RS](
      other: Dataset[R],
      name: String = s"${this.name}/Zip"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S],
      evOutputToDataTypeR: OutputToDataType.Aux[R, RD],
      evOutputToShapeR: OutputToShape.Aux[R, RS]
  ): Dataset[(T, R)] = {
    val providedName = name
    new Dataset[(T, R)] {
      override val name: String = providedName

      override def createHandle[DRD, SRS]()(implicit
          evOutputToDataType: OutputToDataType.Aux[(T, R), DRD],
          evOutputToShape: OutputToShape.Aux[(T, R), SRS]
      ): Output[Variant] = {
        Op.Builder[Seq[Output[Variant]], Output[Variant]](
          opType = "ZipDataset",
          name = name,
          input = Seq(outer.createHandle(), other.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[DRD](implicit ev: OutputToDataType.Aux[(T, R), DRD]): DRD = {
        (outer.outputDataTypes, other.outputDataTypes).asInstanceOf[DRD]
      }

      override def outputShapes[SRS](implicit evOutputToShape: OutputToShape.Aux[(T, R), SRS]): SRS = {
        (outer.outputShapes, other.outputShapes).asInstanceOf[SRS]
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
  def zip3[D, S, R1: OutputStructure, RD1, RS1, R2: OutputStructure, RD2, RS2](
      other1: Dataset[R1],
      other2: Dataset[R2],
      name: String = s"${this.name}/Zip"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S],
      evOutputToDataTypeR1: OutputToDataType.Aux[R1, RD1],
      evOutputToShapeR1: OutputToShape.Aux[R1, RS1],
      evOutputToDataTypeR2: OutputToDataType.Aux[R2, RD2],
      evOutputToShapeR2: OutputToShape.Aux[R2, RS2]
  ): Dataset[(T, R1, R2)] = {
    val providedName = name
    new Dataset[(T, R1, R2)] {
      override val name: String = providedName

      override def createHandle[DRD1RD1, SRS1RS2]()(implicit
          evOutputToDataType: OutputToDataType.Aux[(T, R1, R2), DRD1RD1],
          evOutputToShape: OutputToShape.Aux[(T, R1, R2), SRS1RS2]
      ): Output[Variant] = {
        Op.Builder[Seq[Output[Variant]], Output[Variant]](
          opType = "ZipDataset",
          name = name,
          input = Seq(outer.createHandle(), other1.createHandle(), other2.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[DRD1RD1](implicit evOutputToDataType: OutputToDataType.Aux[(T, R1, R2), DRD1RD1]): DRD1RD1 = {
        (outer.outputDataTypes, other1.outputDataTypes, other2.outputDataTypes).asInstanceOf[DRD1RD1]
      }

      override def outputShapes[SRS1RS2](implicit evOutputToShape: OutputToShape.Aux[(T, R1, R2), SRS1RS2]): SRS1RS2 = {
        (outer.outputShapes, other1.outputShapes, other2.outputShapes).asInstanceOf[SRS1RS2]
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
  def zipMultiple[D, S](
      others: Seq[Dataset[T]],
      name: String = s"${this.name}/Zip"
  )(implicit
      evOutputToDataTypeT: OutputToDataType.Aux[T, D],
      evOutputToShapeT: OutputToShape.Aux[T, S]
  ): Dataset[Seq[T]] = {
    val providedName = name
    new Dataset[Seq[T]] {
      override val name: String = providedName

      override def createHandle[DD, SS]()(implicit
          evOutputToDataType: OutputToDataType.Aux[Seq[T], DD],
          evOutputToShape: OutputToShape.Aux[Seq[T], SS]
      ): Output[Variant] = {
        Op.Builder[Seq[Output[Variant]], Output[Variant]](
          opType = "ZipDataset",
          name = name,
          input = outer.createHandle() +: others.map(_.createHandle())
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[DD](implicit ev: OutputToDataType.Aux[Seq[T], DD]): DD = {
        (outer.outputDataTypes +: others.map(_.outputDataTypes)).asInstanceOf[DD]
      }

      override def outputShapes[SS](implicit ev: OutputToShape.Aux[Seq[T], SS]): SS = {
        (outer.outputShapes +: others.map(_.outputShapes)).asInstanceOf[SS]
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

      override def createHandle[D, S]()(implicit
          evOutputToDataType: OutputToDataType.Aux[T, D],
          evOutputToShape: OutputToShape.Aux[T, S]
      ): Output[Variant] = {
        Op.Builder[Output[Variant], Output[Variant]](
          opType = "IgnoreErrorsDataset",
          name = name,
          input = outer.createHandle()
        ).setAttribute("output_types", flatOutputDataTypes.toArray)
            .setAttribute("output_shapes", flatOutputShapes.toArray)
            .build().output
      }

      override def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
        outer.outputDataTypes
      }

      override def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
        outer.outputShapes
      }
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
    * from a set of TensorFlow record files, shard before converting the dataset to input samples. This avoids
    * reading every file on every worker. The following is an example of an efficient sharding strategy within a
    * complete pipeline:
    * {{{
    *       tf.data.listFiles(pattern)
    *         .shard(numWorkers, workerIndex)
    *         .repeat(numEpochs)
    *         .shuffle(shuffleBufferSize)
    *         .repeat()
    *         .interleave(tf.data.TFRecordDataset, cycleLength = numReaders, blockLength = 1)
    *         .map(parserFn, numParallelCalls)
    * }}}
    *
    * @param  numShards  Number of shards to use.
    * @param  shardIndex Index of the shard to obtain.
    * @return Created (sharded) dataset.
    * @throws InvalidArgumentException If `shardIndex >= numShards`.
    */
  @throws[InvalidArgumentException]
  def shard[D, S](
      numShards: Long,
      shardIndex: Long
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S],
      // These implicit helpers is used for Scala 2.11 support.
      evOutputToDataType211Helper: OutputToDataType.Aux[(T, Output[Long]), (D, DataType[Long])],
      evOutputToShape211Helper: OutputToShape.Aux[(T, Output[Long]), (S, Shape)]
  ): Dataset[T] = {
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
      evR: OutputStructure[R]
  ): Dataset[R] = {
    transformFn(this)
  }

  override def toString: String = {
    s"Dataset[$name]"
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
