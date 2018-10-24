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
import org.platanios.tensorflow.api.core.types.{DataType, Resource, Variant}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.implicits.helpers._

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

/** A simple iterator that does contains an initializer and can thus not be used until an initializer is created for
  * it, using its `createInitializer` method.
  *
  * An iterator represents the state of iterating through a dataset.
  *
  * @param  handle           Handle of the iterator.
  * @param  _outputDataTypes Data types corresponding to each element of the iterator.
  * @param  _outputShapes    Shapes corresponding to each element of the iterator.
  * @param  name             Name for this iterator.
  *
  * @author Emmanouil Antonios Platanios
  */
class DatasetIterator[T] protected[data](
    protected val handle: Output[Resource]
)(
    protected val _outputDataTypes: Any,
    protected val _outputShapes: Any,
    val name: String = "DatasetIterator"
) {
  /** Returns the data types corresponding to each element of this dataset, matching the structure of the elements. */
  def outputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): D = {
    _outputDataTypes.asInstanceOf[D]
  }

  /** Returns the shapes corresponding to each element of this dataset, matching the structure of the elements. */
  def outputShapes[S](implicit ev: OutputToShape.Aux[T, S]): S = {
    _outputShapes.asInstanceOf[S]
  }

  private var nextCallCount: Int = 0

  /** Returns a sequence of data types that correspond to the flattened data types of the nested outputs structure
    * of the elements of this iterator. */
  private[data] def flatOutputDataTypes[D](implicit ev: OutputToDataType.Aux[T, D]): Seq[DataType[Any]] = {
    ev.dataTypeStructure.dataTypes(outputDataTypes)
  }

  /** Returns a sequence of shapes that correspond to the flattened shapes of the nested outputs structure of the
    * elements of this iterator. */
  private[data] def flatOutputShapes[S](implicit ev: OutputToShape.Aux[T, S]): Seq[Shape] = {
    ev.shapeStructure.shapes(outputShapes)
  }

  /** Returns an op that initializes this iterator using the provided dataset.
    *
    * @param  dataset Dataset to initialize this iterator with. The output data types of this iterator must match the
    *                 output data types of the dataset, and its output shapes must be compatible with the output shapes
    *                 of the dataset.
    * @param  name    Name for the created op.
    * @return Created op.
    * @throws IllegalArgumentException If any of the output shapes of this iterator is not compatible with
    *                                  the corresponding output shapes of the provided dataset.
    */
  @throws[IllegalArgumentException]
  def createInitializer[D, S](
      dataset: Dataset[T],
      name: String = s"$name/Initializer"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): Op[(Output[Variant], Output[Resource]), Unit] = {
    createInitializerFromHandle(dataset.createHandle(), name)
  }

  /** Returns an op that initializes this iterator using the provided dataset handle.
    *
    * '''NOTE:''' It is advisable not to use this method for initializing iterators as it does not support compile-time
    * checking for whether the provided dataset handle is compatible with this iterator.
    *
    * @param  datasetHandle Dataset handle to initialize this iterator with. The output data types of this iterator must
    *                       match the output data types of the corresponding dataset, and its output shapes must be
    *                       compatible with the output shapes of that dataset.
    * @param  name          Name for the created op.
    * @return Created op.
    */
  def createInitializerFromHandle(
      datasetHandle: Output[Variant],
      name: String = s"$name/Initializer"
  ): Op[(Output[Variant], Output[Resource]), Unit] = {
    Op.nameScope(name) {
      Op.colocateWith(Set(handle.op), ignoreExisting = true) {
        DatasetIterator.makeIterator(
          datasetHandle = datasetHandle,
          iteratorHandle = handle)
      }
    }
  }

  /** Creates an op that obtains the next element of this iterator and returns a nested structure of outputs
    * (according to the structures supported by the `Data` type trait) that corresponds to that element.
    *
    * @param  name Name for the created op.
    * @return Created op outputs in a nested structure according to the data type of this initializer.
    */
  def next[D, S](name: String = s"$name/Next")(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): T = {
    nextCallCount += 1
    if (nextCallCount > DatasetIterator.NEXT_CALL_WARNING_THRESHOLD)
      DatasetIterator.logger.warn(DatasetIterator.NEXT_CALL_WARNING_MESSAGE)
    val flattenedNext = DatasetIterator.iteratorGetNext(
      iteratorHandle = handle,
      outputDataTypes = flatOutputDataTypes,
      outputShapes = flatOutputShapes,
      name = name)
    evOutputToDataType.decodeOutput(outputDataTypes, flattenedNext)._1
  }

  // TODO: [MEMORY] Add automatic disposal of iterators if necessary.

  /** Creates an op that destroys this iterator.
    *
    * The returned op may be used to release any resources consumed by this iterator, without closing the session.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def dispose(name: String = s"$name/Dispose"): Op[Output[Resource], Unit] = {
    DatasetIterator.iteratorDispose(handle, name)
  }

  /** Creates an op that converts the provided resource handle representing an iterator to a string.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def toStringHandle(name: String = s"$name/ToStringHandle"): Output[String] = {
    DatasetIterator.iteratorToStringHandle(handle, name)
  }
}

/** A dataset iterator that contains an initializer.
  *
  * An iterator represents the state of iterating through a dataset.
  *
  * @param  handle           Handle of the iterator.
  * @param  initializer      Iterator initializer op.
  * @param  _outputDataTypes Data types corresponding to each element of the iterator.
  * @param  _outputShapes    Shapes corresponding to each element of the iterator.
  * @param  name             Name for this iterator.
  */
class InitializableDatasetIterator[T] private[data](
    override protected val handle: Output[Resource]
)(
    val initializer: UntypedOp,
    override protected val _outputDataTypes: Any,
    override protected val _outputShapes: Any,
    override val name: String = "InitializableDatasetIterator"
) extends DatasetIterator[T](handle)(_outputDataTypes, _outputShapes, name)

/** Contains helper functions for creating iterator-related ops, as well as the iterator API trait. */
object DatasetIterator {
  private[data] val logger = Logger(LoggerFactory.getLogger("Data / Iterator"))

  private[data] trait API {
    def iteratorFromDataset[T, D, S](
        dataset: Dataset[T],
        sharedName: String = "",
        name: String = "InitializableIterator"
    )(implicit
        evOutputToDataType: OutputToDataType.Aux[T, D],
        evOutputToShape: OutputToShape.Aux[T, S]
    ): InitializableDatasetIterator[T] = {
      fromDataset(dataset, sharedName, name)
    }

    def iteratorFromStructure[T, D, S](
        outputDataTypes: D,
        outputShapes: S,
        sharedName: String = "",
        name: String = "Iterator"
    )(implicit
        evDataTypeToOutput: DataTypeToOutput.Aux[D, T],
        evOutputToDataType: OutputToDataType.Aux[T, D],
        evOutputToShape: OutputToShape.Aux[T, S]
    ): DatasetIterator[T] = {
      fromStructure(outputDataTypes, outputShapes, sharedName, name)
    }

    def iteratorFromStringHandle[T, D, S](
        stringHandle: Output[String],
        outputDataTypes: D,
        outputShapes: S,
        name: String = "IteratorFromStringHandle"
    )(implicit
        evDataTypeToOutput: DataTypeToOutput.Aux[D, T],
        evOutputToShape: OutputToShape.Aux[T, S]
    ): DatasetIterator[T] = {
      fromStringHandle(stringHandle, outputDataTypes, outputShapes, name)
    }
  }

  /** Note: It is legitimate to call `Iterator.next()` multiple times, e.g. when you are distributing different elements
    * to multiple devices in a single step. However, a common pitfall arises when users call `Iterator.next()` in each
    * iteration of their training loop. `Iterator.next()` adds ops to the graph, and executing each op allocates
    * resources (including threads); as a consequence, invoking it in every iteration of a training loop causes slowdown
    * and eventual resource exhaustion. To guard against this outcome, we log a warning when the number of uses crosses
    * a threshold of suspicion. */
  private[data] val NEXT_CALL_WARNING_THRESHOLD: Int    = 32
  private[data] val NEXT_CALL_WARNING_MESSAGE  : String =
    """An unusually high number of `Iterator.next()` calls was detected. This often indicates that `Iterator.next()`
      |is being called inside a training loop, which will cause gradual slowdown and eventual resource exhaustion. If
      |this is the case, restructure your code to call `nextElement = iterator.next() once outside the loop, and use
      |`nextElement` inside the loop.
    """.stripMargin

  /** Creates a new, uninitialized iterator from the provided dataset.
    *
    * To initialize this iterator, you must run its `initializer`:
    * {{{
    *   val iterator = Iterator.fromDataset(dataset)
    *   // ...
    *   session.run(targets = iterator.initializer)
    * }}}
    *
    * @param  dataset    A dataset for which to create an iterator.
    * @param  sharedName If non-empty, then the constructed iterator will be shared under the the provided name across
    *                    multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name       Name to use for the created ops.
    * @return Created iterator.
    */
  private[api] def fromDataset[T, D, S](
      dataset: Dataset[T],
      sharedName: String = "",
      name: String = "InitializableIterator"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): InitializableDatasetIterator[T] = {
    val (handle, initializer) = Op.nameScope(name) {
      val handle = createIterator(
        sharedName = sharedName,
        outputDataTypes = dataset.flatOutputDataTypes,
        outputShapes = dataset.flatOutputShapes)
      val initializer = makeIterator(
        datasetHandle = dataset.createHandle(),
        iteratorHandle = handle)
      (handle, initializer)
    }
    new InitializableDatasetIterator[T](handle)(
      initializer = initializer,
      _outputDataTypes = dataset.outputDataTypes,
      _outputShapes = dataset.outputShapes,
      name = name)
  }

  /** Creates a new, uninitialized iterator with the provided structure.
    *
    * This iterator-constructing function can be used to create an iterator that is reusable with many different
    * datasets. The returned iterator is not bound to a particular dataset and thus it has no initializer. To
    * initialize the iterator, the user has to run the op returned by `Iterator.createInitializer()`.
    *
    * For example:
    * {{{
    *   val iterator = tf.iteratorFromStructure(tf.INT64, tf.shape())
    *
    *   val rangeDataset = tf.rangeDataset(10)
    *   val rangeInitializer = iterator.createInitializer(rangeDataset)
    *
    *   val evensDataset = rangeDataset.filter(_ % 2 == 0)
    *   val evensInitializer = iterator.createInitializer(evenDataset)
    *
    *   // Define a model based on the iterator. In this example, 'modelFunction' is expected to take INT64 scalar
    *   // tensors as input (see the definition of 'iterator' above).
    *   val (prediction, loss) = modelFunction(iterator.next())
    *
    *   // Train for 'numEpochs', where for each epoch, we first iterate over 'rangeDataset', and then iterate over
    *   // 'evensDataset'.
    *   (0 until numEpochs).foreach(i => {
    *     // Initialize the iterator to 'rangeDataset'.
    *     session.run(targets = rangeInitializer)
    *     var exhaustedIterator = false
    *     while (!exhaustedIterator) {
    *       try {
    *         val (p, l) = session.run(fetches = (prediction, loss))
    *       } catch {
    *         case _: OutOfRangeException => exhaustedIterator = true
    *       }
    *     }
    *
    *     // Initialize the iterator to 'evensDataset'.
    *     session.run(targets = evensInitializer)
    *     var exhaustedIterator = false
    *     while (!exhaustedIterator) {
    *       try {
    *         val (p, l) = session.run(fetches = (prediction, loss))
    *       } catch {
    *         case _: OutOfRangeException => exhaustedIterator = true
    *       }
    *     }
    *   })
    * }}}
    *
    * @param  outputDataTypes Output data types corresponding to each element of the iterator.
    * @param  outputShapes    Output shapes corresponding to each element of the iterator.
    * @param  sharedName      If non-empty, then the constructed iterator will be shared under the the provided name
    *                         across multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  name            Name to use for the created ops.
    * @return Created iterator.
    */
  private[api] def fromStructure[T, D, S](
      outputDataTypes: D,
      outputShapes: S,
      sharedName: String = "",
      name: String = "Iterator"
  )(implicit
      evOutputToDataType: OutputToDataType.Aux[T, D],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): DatasetIterator[T] = {
    // TODO: [DATASETS] Allow for shapes to not be provided.
    val flatOutputDataTypes = evOutputToDataType.dataTypeStructure.dataTypes(outputDataTypes)
    val flatOutputShapes = evOutputToShape.shapeStructure.shapes(outputShapes)
    val handle = createIterator(
      sharedName = sharedName,
      outputDataTypes = flatOutputDataTypes,
      outputShapes = flatOutputShapes)
    new DatasetIterator[T](handle)(outputDataTypes, outputShapes, name)
  }

  /** Creates a new, uninitialized iterator from the provided `STRING` scalar tensor representing a handle of an
    * existing iterator.
    *
    * This method allows you to define a "feedable" iterator where you can choose between concrete iterators by
    * feeding a value in a [[org.platanios.tensorflow.api.core.client.Session.run]] call. In that case,
    * `stringHandle` would be a `tf.placeholder`, and you would feed it with the value of an existing iterator's
    * `Iterator.toStringHandle()` in each step.
    *
    * For example, if you had two iterators that marked the current position in a training dataset and a test dataset,
    * you could choose which one to use in each step, as follows:
    * {{{
    *   val trainIterator = tf.dataset(...).createOneShotIterator()
    *   val trainIteratorHandle = session.run(fetches = trainIterator.toStringHandle())
    *
    *   val testIterator = tf.dataset(...).createOneShotIterator()
    *   val testIteratorHandle = session.run(fetches = testIterator.toStringHandle())
    *
    *   val handle = tf.placeholder(tf.STRING, shape = Shape.scalar)
    *   val iterator = tf.iteratorFromStringHandle(handle, trainIterator.outputDataTypes)
    *
    *   val nextElement = iterator.next()
    *   val loss = f(nextElement)
    *
    *   val trainLoss = session.run(feeds = Map(handle -> trainIteratorHandle), fetches = loss)
    *   val testLoss = session.run(feeds = Map(handle -> testIteratorHandle), fetches = loss)
    * }}}
    *
    * @param  stringHandle    Scalar tensor containing the string representation of a handle of an iterator.
    * @param  outputDataTypes Output data types corresponding to each element of the iterator.
    * @param  outputShapes    Output shapes corresponding to each element of the iterator.
    * @param  name            Name to use for the created op.
    * @return Created iterator.
    */
  private[api] def fromStringHandle[T, D, S](
      stringHandle: Output[String],
      outputDataTypes: D,
      outputShapes: S,
      name: String = "IteratorFromStringHandle"
  )(implicit
      evDataTypeToOutput: DataTypeToOutput.Aux[D, T],
      evOutputToShape: OutputToShape.Aux[T, S]
  ): DatasetIterator[T] = {
    // TODO: [DATASETS] Allow for shapes to not be provided.
    val flatOutputDataTypes = evDataTypeToOutput.dataTypeStructure.dataTypes(outputDataTypes)
    val flatOutputShapes = evOutputToShape.shapeStructure.shapes(outputShapes)
    val handle = iteratorFromStringHandle(
      stringHandle = stringHandle,
      outputDataTypes = flatOutputDataTypes,
      outputShapes = flatOutputShapes,
      name = name)
    new DatasetIterator[T](handle)(outputDataTypes, outputShapes, name)
  }

  //region Low Level Ops

  /** Creates an op that is a container for an iterator resource.
    *
    * @param  container       If non-empty, then the constructed iterator is placed in the provided container.
    *                         Otherwise, a default container is used.
    * @param  sharedName      If non-empty, then the constructed iterator will be shared under the the provided name
    *                         across multiple sessions that share the same devices (e.g., when using a remote server).
    * @param  outputDataTypes Output data types of the created iterator.
    * @param  outputShapes    Output shapes of the created iterator.
    * @param  name            Name for the created op.
    * @return Created op output, which is a handle to the constructed iterator.
    */
  private[data] def createIterator(
      container: String = "",
      sharedName: String = "",
      outputDataTypes: Seq[DataType[Any]],
      outputShapes: Seq[Shape],
      name: String = "Iterator"
  ): Output[Resource] = {
    Op.Builder[Unit, Output[Resource]](
      opType = "Iterator",
      name = name,
      input = ()
    ).setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().output
  }

  /** Creates an op that makes a new iterator for the provided dataset and stores it in the container pointed to by the
    * provided iterator handle.
    *
    * **Note:** The created op may be executed multiple times. Each execution will reset the iterator in `iterator` to
    * the first element of `dataset`.
    *
    * @param  datasetHandle  Handle of the dataset.
    * @param  iteratorHandle Handle of the iterator.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  private[data] def makeIterator(
      datasetHandle: Output[Variant],
      iteratorHandle: Output[Resource],
      name: String = "MakeIterator"
  ): Op[(Output[Variant], Output[Resource]), Unit] = {
    Op.Builder[(Output[Variant], Output[Resource]), Unit](
      opType = "MakeIterator",
      name = name,
      input = (datasetHandle, iteratorHandle)
    ).build()
  }

  // TODO: [DATASETS] [FUNCTIONS] "oneShotIterator".

  /** Creates an op that gets the next output from the provided iterator.
    *
    * @param  iteratorHandle  Handle of the iterator.
    * @param  outputDataTypes Output data types of the iterator.
    * @param  outputShapes    Output shapes of the iterator.
    * @param  name            Name for the created op.
    * @return Created op outputs, which correspond to the iterator outputs.
    */
  private[data] def iteratorGetNext(
      iteratorHandle: Output[Resource],
      outputDataTypes: Seq[DataType[Any]],
      outputShapes: Seq[Shape],
      name: String = "IteratorGetNext"
  ): Seq[Output[Any]] = {
    Op.Builder[Output[Resource], Seq[Output[Any]]](
      opType = "IteratorGetNext",
      name = name,
      input = iteratorHandle
    ).setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().output
  }

  /** Creates an op that releases any resources used by the provided iterator.
    *
    * @param  iteratorHandle Handle of the iterator.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  private[data] def iteratorDispose(
      iteratorHandle: Output[Resource],
      name: String = "IteratorDispose"
  ): Op[Output[Resource], Unit] = {
    Op.Builder[Output[Resource], Unit](
      opType = "IteratorDispose",
      name = name,
      input = iteratorHandle
    ).build()
  }

  /** Creates an op that converts the provided resource handle representing an iterator to a string.
    *
    * @param  iteratorHandle Handle of the iterator.
    * @param  name           Name for the created op.
    * @return Created op output, which is a scalar tensor containing the string handle.
    */
  private[data] def iteratorToStringHandle(
      iteratorHandle: Output[Resource],
      name: String = "IteratorToStringHandle"
  ): Output[String] = {
    Op.Builder[Output[Resource], Output[String]](
      opType = "IteratorToStringHandle",
      name = name,
      input = iteratorHandle
    ).build().output
  }

  /** Creates an op that converts the provided string representing a handle to an iterator to the corresponding iterator
    * handle.
    *
    * @param  stringHandle Scalar tensor containing the string representation of a handle of an iterator.
    * @param  name         Name for the created op.
    * @return Created op output, which is a scalar tensor containing the iterator handle.
    */
  private[data] def iteratorFromStringHandle(
      stringHandle: Output[String],
      outputDataTypes: Seq[DataType[Any]],
      outputShapes: Seq[Shape],
      name: String = "IteratorFromStringHandle"
  ): Output[Resource] = {
    Op.Builder[Output[String], Output[Resource]](
      opType = "IteratorFromStringHandleV2",
      name = name,
      input = stringHandle
    ).setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().output
  }

  //endregion Low Level Ops
}
