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
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.ops.io.data
import org.platanios.tensorflow.api.types.DataType

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

// TODO: Rename to "DatasetIterator".

/** A simple iterator that does contains an initializer and can thus not be used until an initializer is created for
  * it, using its `createInitializer` method.
  *
  * An iterator represents the state of iterating through a dataset.
  *
  * @param  handle          Handle of the iterator.
  * @param  outputDataTypes Data types corresponding to each element of the iterator.
  * @param  outputShapes    Shapes corresponding to each element of the iterator.
  *
  * @author Emmanouil Antonios Platanios
  */
class Iterator[T, O, D, S] private[io](
    val handle: Output,
    val outputDataTypes: D,
    val outputShapes: S,
    val name: String = "Iterator"
)(implicit
    ev: Data.Aux[T, O, D, S]
) {
  private[this] var nextCallCount: Int = 0

  /** Returns an op that initializes this iterator using the provided dataset.
    *
    * @param  dataset Dataset to initialize this iterator with. The output data types of this iterator must match the
    *                 output data types of the dataset, and its output shapes must be compatible with the output shapes
    *                 of the dataset.
    * @param  name    Name for the created op.
    * @return Created op.
    * @throws IllegalArgumentException If any of the output data types or shapes of this iterator is not compatible with
    *                                  the corresponding output data type or shape of the provided dataset.
    */
  @throws[IllegalArgumentException]
  def createInitializer(dataset: Dataset[T, O, D, S], name: String = s"$name/Initializer"): Op = {
    if (flattenedOutputDataTypes.zip(dataset.flattenedOutputDataTypes).exists(t => t._1 != t._2))
      throw new IllegalArgumentException(
        s"Expected output data types '$outputDataTypes', " +
            s"but got dataset with output data types '${dataset.outputDataTypes}'.")
    if (flattenedOutputShapes.zip(dataset.flattenedOutputShapes).exists(s => !s._1.isCompatibleWith(s._2)))
      throw new IllegalArgumentException(
        s"Expected output shapes compatible with '$outputShapes', " +
            s"but got dataset with output shapes '${dataset.outputShapes}'.")
    Op.colocateWith(Set(handle.op), ignoreExisting = true) {
      Iterator.makeIterator(datasetHandle = dataset.createHandle(), iteratorHandle = handle)
    }
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
  def createInitializerFromHandle(datasetHandle: Output, name: String = s"$name/Initializer"): Op = {
    Op.colocateWith(Set(handle.op), ignoreExisting = true) {
      Iterator.makeIterator(datasetHandle = datasetHandle, iteratorHandle = handle)
    }
  }

  /** Creates an op that obtains the next element of this iterator and returns a nested structure of outputs
    * (according to the structures supported by the `Data` type trait) that corresponds to that element.
    *
    * @param  name Name for the created op.
    * @return Created op outputs in a nested structure according to the data type of this initializer.
    */
  def next(name: String = s"$name/Next"): O = {
    nextCallCount += 1
    if (nextCallCount > Iterator.NEXT_CALL_WARNING_THRESHOLD)
      Iterator.logger.warn(Iterator.NEXT_CALL_WARNING_MESSAGE)
    val flattenedNext = Iterator.iteratorGetNext(
      iteratorHandle = handle,
      outputDataTypes = flattenedOutputDataTypes,
      outputShapes = flattenedOutputShapes,
      name = name)
    ev.unflattenOutputs(outputDataTypes, flattenedNext)
  }

  // TODO: Add automatic disposal of iterators if necessary.

  /** Creates an op that destroys this iterator.
    *
    * The returned op may be used to release any resources consumed by this iterator, without closing the session.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def dispose(name: String = s"$name/Dispose"): Op = {
    Iterator.iteratorDispose(iteratorHandle = handle, name = name)
  }

  /** Creates an op that converts the provided resource handle representing an iterator to a string.
    *
    * @param  name Name for the created op.
    * @return Created op.
    */
  def toStringHandle(name: String = s"$name/ToStringHandle"): Output = {
    Iterator.iteratorToStringHandle(iteratorHandle = handle, name = name)
  }

  /** Returns a sequence of data types that correspond to the flattened data types of the nested outputs structure
    * of the elements of this iterator. */
  private[this] def flattenedOutputDataTypes: Seq[DataType[_]] = ev.flattenedDataTypes(outputDataTypes)

  /** Returns a sequence of shapes that correspond to the flattened shapes of the nested outputs structure of the
    * elements of this iterator. */
  private[this] def flattenedOutputShapes: Seq[Shape] = ev.flattenedShapes(outputShapes)
}

/** An iterator that contains an initializer.
  *
  * An iterator represents the state of iterating through a dataset.
  *
  * @param  handle          Handle of the iterator.
  * @param  initializer     Iterator initializer op.
  * @param  outputDataTypes Output data types corresponding to each element of the iterator.
  * @param  outputShapes    Output shapes corresponding to each element of the iterator.
  */
class InitializableIterator[T, O, D, S] private[io](
    override val handle: Output,
    val initializer: Op,
    override val outputDataTypes: D,
    override val outputShapes: S,
    override val name: String = "InitializableIterator"
)(implicit
    ev: Data.Aux[T, O, D, S]
) extends Iterator[T, O, D, S](handle, outputDataTypes, outputShapes, name)(ev)

/** Contains helper functions for creating iterator-related ops, as well as the iterator API trait. */
object Iterator {
  private[data] val logger = Logger(LoggerFactory.getLogger("Data / Iterator"))

  type Iterator[T, O, D, S] = data.Iterator[T, O, D, S]

  private[io] trait API {
    def iteratorFromDataset[T, O, D, S](
        dataset: Dataset[T, O, D, S],
        sharedName: String = "",
        name: String = "InitializableIterator"
    )(implicit ev: Data.Aux[T, O, D, S]): InitializableIterator[T, O, D, S] = {
      fromDataset(dataset, sharedName, name)(ev)
    }

    def iteratorFromStructure[T, O, D, S](
        outputDataTypes: D,
        outputShapes: S,
        sharedName: String = "",
        name: String = "Iterator"
    )(
        implicit ev: Data.Aux[T, O, D, S]): Iterator[T, O, D, S] = {
      fromStructure(outputDataTypes, outputShapes, sharedName, name)(ev)
    }

    def iteratorFromStringHandle[T, O, D, S](
        stringHandle: Output,
        outputDataTypes: D,
        outputShapes: S,
        name: String = "IteratorFromStringHandle"
    )(implicit ev: Data.Aux[T, O, D, S]): Iterator[T, O, D, S] = {
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
  private[api] def fromDataset[T, O, D, S](
      dataset: Dataset[T, O, D, S],
      sharedName: String = "",
      name: String = "InitializableIterator"
  )(implicit ev: Data.Aux[T, O, D, S]): InitializableIterator[T, O, D, S] = {
    val (handle, initializer) = Op.createWithNameScope(name) {
      val handle = createIterator(
        sharedName = sharedName,
        outputDataTypes = dataset.flattenedOutputDataTypes,
        outputShapes = dataset.flattenedOutputShapes)
      val initializer = makeIterator(datasetHandle = dataset.createHandle(), iteratorHandle = handle)
      (handle, initializer)
    }
    new InitializableIterator[T, O, D, S](
      handle = handle,
      initializer = initializer,
      outputDataTypes = dataset.outputDataTypes,
      outputShapes = dataset.outputShapes,
      name = name)(dataset.evData)
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
  private[api] def fromStructure[T, O, D, S](
      outputDataTypes: D,
      outputShapes: S,
      sharedName: String = "",
      name: String = "Iterator"
  )(implicit ev: Data.Aux[T, O, D, S]): Iterator[T, O, D, S] = {
    // TODO: [DATASETS] Allow for shapes to not be provided.
    val flattenedOutputDataTypes = ev.flattenedDataTypes(outputDataTypes)
    val flattenedShapes = ev.flattenedShapes(outputShapes)
    val handle = createIterator(
      sharedName = sharedName,
      outputDataTypes = flattenedOutputDataTypes,
      outputShapes = flattenedShapes)
    new Iterator[T, O, D, S](
      handle = handle,
      outputDataTypes = outputDataTypes,
      outputShapes = outputShapes,
      name = name)
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
    * @param  stringHandle    `STRING` scalar tensor containing the string representation of a handle of an iterator.
    * @param  outputDataTypes Output data types corresponding to each element of the iterator.
    * @param  outputShapes    Output shapes corresponding to each element of the iterator.
    * @param  name            Name to use for the created op.
    * @return Created iterator.
    */
  private[api] def fromStringHandle[T, O, D, S](
      stringHandle: Output,
      outputDataTypes: D,
      outputShapes: S,
      name: String = "IteratorFromStringHandle"
  )(implicit ev: Data.Aux[T, O, D, S]): Iterator[T, O, D, S] = {
    // TODO: [DATASETS] Allow for shapes to not be provided.
    val handle = Iterator.iteratorFromStringHandle(
      stringHandle = stringHandle,
      outputDataTypes = ev.flattenedDataTypes(outputDataTypes),
      outputShapes = ev.flattenedShapes(outputShapes),
      name = name)
    new Iterator[T, O, D, S](
      handle = handle,
      outputDataTypes = outputDataTypes,
      outputShapes = outputShapes,
      name = name)
  }

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
  private[io] def createIterator(
      container: String = "",
      sharedName: String = "",
      outputDataTypes: Seq[DataType[_]],
      outputShapes: Seq[Shape],
      name: String = "Iterator"
  ): Output = {
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
    * @param  datasetHandle  Handle of the dataset.
    * @param  iteratorHandle Handle of the iterator.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  private[io] def makeIterator(datasetHandle: Output, iteratorHandle: Output, name: String = "MakeIterator"): Op = {
    Op.Builder(opType = "MakeIterator", name = name)
        .addInput(datasetHandle)
        .addInput(iteratorHandle)
        .build()
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
  private[io] def iteratorGetNext(
      iteratorHandle: Output,
      outputDataTypes: Seq[DataType[_]],
      outputShapes: Seq[Shape],
      name: String = "IteratorGetNext"
  ): Seq[Output] = {
    Op.Builder(opType = "IteratorGetNext", name = name)
        .addInput(iteratorHandle)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs.toSeq
  }

  /** Creates an op that releases any resources used by the provided iterator.
    *
    * @param  iteratorHandle Handle of the iterator.
    * @param  name           Name for the created op.
    * @return Created op.
    */
  private[io] def iteratorDispose(iteratorHandle: Output, name: String = "IteratorDispose"): Op = {
    Op.Builder(opType = "IteratorDispose", name = name)
        .addInput(iteratorHandle)
        .build()
  }

  /** Creates an op that converts the provided resource handle representing an iterator to a string.
    *
    * @param  iteratorHandle Handle of the iterator.
    * @param  name           Name for the created op.
    * @return Created op output, which is a `STRING` scalar tensor containing the string handle.
    */
  private[io] def iteratorToStringHandle(iteratorHandle: Output, name: String = "IteratorToStringHandle"): Output = {
    Op.Builder(opType = "IteratorToStringHandle", name = name)
        .addInput(iteratorHandle)
        .build().outputs(0)
  }

  /** Creates an op that converts the provided string representing a handle to an iterator to the corresponding iterator
    * handle.
    *
    * @param  stringHandle `STRING` scalar tensor containing the string representation of a handle of an iterator.
    * @param  name         Name for the created op.
    * @return Created op output, which is a `VARIANT` scalar tensor containing the iterator handle.
    */
  private[io] def iteratorFromStringHandle(
      stringHandle: Output,
      outputDataTypes: Seq[DataType[_]],
      outputShapes: Seq[Shape],
      name: String = "IteratorFromStringHandle"
  ): Output = {
    Op.Builder(opType = "IteratorFromStringHandle", name = name)
        .addInput(stringHandle)
        .setAttribute("output_types", outputDataTypes.toArray)
        .setAttribute("output_shapes", outputShapes.toArray)
        .build().outputs(0)
  }

  private[ops] object Gradients {
    GradientsRegistry.registerNonDifferentiable("Iterator")
    GradientsRegistry.registerNonDifferentiable("MakeIterator")
    GradientsRegistry.registerNonDifferentiable("OneShotIterator")
    GradientsRegistry.registerNonDifferentiable("IteratorGetNext")
    GradientsRegistry.registerNonDifferentiable("IteratorDispose")
    GradientsRegistry.registerNonDifferentiable("IteratorToStringHandle")
    GradientsRegistry.registerNonDifferentiable("IteratorFromStringHandle")
  }
}
