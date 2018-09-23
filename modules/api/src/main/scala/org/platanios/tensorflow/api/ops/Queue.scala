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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.types.{DataType, INT64}

import java.nio.ByteBuffer
import java.security.MessageDigest

import scala.language.postfixOps

/** Class that supports all TensorFlow queue implementations.
  *
  * A queue is a TensorFlow data structure that stores tensors across multiple steps, and exposes operations that
  * enqueue and dequeue tensors.
  *
  * Each queue element is a sequence/tuple of one or more tensors, where each tuple component has a static data type,
  * and may have a statically-known shape. The queue implementations support versions of enqueue and dequeue that handle
  * single elements and versions that support enqueuing and dequeuing many elements at once.
  *
  * @param  handle    Handle to the underlying TensorFlow queue.
  * @param  dataTypes Sequence of data types for each element in the queue (an element is a tuple of tensors).
  * @param  shapes    Sequence of shapes for each element in the queue (an element is a tuple of tensors).
  *
  * @author Emmanouil Antonios Platanios
  */
class Queue private[Queue](
    val handle: Output,
    val dataTypes: Seq[DataType[_]]
)(
    val shapes: Seq[Shape] = Seq.fill(dataTypes.size)(Shape.unknown())
) {
  /** Name of this queue. */
  protected val name: String = handle.op.name.split("/").last

  if (dataTypes.isEmpty)
    throw new IllegalArgumentException("Cannot create a queue when no data types are provided.")
  if (shapes.size != dataTypes.size)
    throw new IllegalArgumentException(s"The provided data types and shapes must have the same length.")

  /** Validates the provided `values` and casts them to the expected data type for this queue. If `checkNames` is
    * `true`, an [[IllegalArgumentException]] is thrown if the queue has names and thus expects a `Map[String, Output]`
    * as argument to `enqueue`. */
  @throws[IllegalArgumentException]
  private[this] def processEnqueueValues(values: Seq[Output]): Seq[Output] = {
    values.zip(dataTypes).map(p => Cast.cast(p._1, p._2))
  }

  /** Creates an op that enqueues `values` as a single element to this queue.
    *
    * If the queue is full when the created op executes, it will block until the element has been enqueued (or until
    * `timeout` s elapses, if specified).
    *
    * At runtime, the op may raise an error if the queue is closed before or during its execution. If the queue is
    * closed before the op runs, a cancellation error will be thrown. If the op is blocked and either: (i) the queue is
    * closed by a close operation with `cancelPendingEnqueues = true`, or (ii) the session is closed, then a
    * cancellation error will also be thrown.
    *
    * @param  values  Values to enqueue, corresponding to a single queue element.
    * @param  timeout If the queue is full, the created op will block for up to `timeout` s, if provided.
    * @param  name    Name for the created op.
    * @return Created op.
    */
  def enqueue(values: Seq[Output], timeout: Option[Double] = None, name: String = s"$name/Enqueue"): Op = {
    Op.createWithNameScope(name, values.map(_.op).toSet) {
      val processedValues = processEnqueueValues(values)
      processedValues.zip(shapes).foreach(p => p._1.shape.assertIsCompatibleWith(p._2))
      Queue.queueEnqueue(handle, processedValues, timeout.map(t => (t * 1000).toInt))
    }
  }

  /** Creates an op that enqueues zero or more elements to this queue.
    *
    * The op slices each component tensor along the `0`th axis to make multiple queue elements. All of the tensors in
    * `values` must have the same size in the `0`th axis.
    *
    * If the queue is full when the created op executes, it will block until the element has been enqueued (or until
    * `timeout` s elapses, if specified).
    *
    * At runtime, the op may raise an error if the queue is closed before or during its execution. If the queue is
    * closed before the op runs, a cancellation error will be thrown. If the op is blocked and either: (i) the queue is
    * closed by a close operation with `cancelPendingEnqueues = true`, or (ii) the session is closed, then a
    * cancellation error will also be thrown.
    *
    * @param  values  Values to enqueue, corresponding to a single queue element.
    * @param  timeout If the queue is full, the created op will block for up to `timeout` s, if provided.
    * @param  name    Name for the created op.
    * @return Created op.
    */
  def enqueueMany(values: Seq[Output], timeout: Option[Double] = None, name: String = s"$name/EnqueueMany"): Op = {
    Op.createWithNameScope(name, values.map(_.op).toSet) {
      val processedValues = processEnqueueValues(values)
      processedValues.zip(shapes).foreach(p => p._1.shape(1 ::).assertIsCompatibleWith(p._2))
      Queue.queueEnqueueMany(handle, processedValues, timeout.map(t => (t * 1000).toInt))
    }
  }

  /** Creates an op that dequeues a single element from this queue.
    *
    * If the queue is empty when the created op executes, it will block until there is an element to dequeue (or until
    * `timeout` s elapses, if specified).
    *
    * At runtime, the op may raise an error if the queue is closed before or during its execution. If the queue is
    * closed, it is empty, and there are no pending enqueue operations that can fulfill this request, then an out of
    * range error will be thrown. If the session is closed, then a cancellation error will be thrown.
    *
    * @param  timeout If the queue is empty, the created op will block for up to `timeout` s, if provided.
    * @param  name    Name for the created op.
    * @return Created op outputs.
    */
  def dequeue(timeout: Option[Double] = None, name: String = s"$name/Dequeue"): Seq[Output] = {
    val dequeued = Queue.queueDequeue(handle, timeout.map(t => (t * 1000).toInt), name)
    dequeued.head.op.outputs.zip(shapes).foreach(p => p._1.setShape(p._2))
    dequeued
  }

  /** Creates an op that dequeues and concatenates `n` elements from this queue.
    *
    * The op concatenates queue-element component tensors along the `0`th axis to make a single component tensor. All of
    * the components in the dequeued tuple will have size `n` in the `0`th axis.
    *
    * If the queue is closed and there are less than `n` elements left, then an out of range error is thrown.
    *
    * At runtime, the op may raise an error if the queue is closed before or during its execution. If the queue is
    * closed, it is empty, and there are no pending enqueue operations that can fulfill this request, then an out of
    * range error will be thrown. If the session is closed, then a cancellation error will be thrown.
    *
    * @param  timeout If the queue contains less than `n` elements, the created op will block for up to `timeout` s, if
    *                 provided.
    * @param  name    Name for the created op.
    * @return Created op outputs.
    */
  def dequeueMany(n: Output, timeout: Option[Double] = None, name: String = s"$name/DequeueMany"): Seq[Output] = {
    val dequeued = Queue.queueDequeueMany(handle, n, timeout.map(t => (t * 1000).toInt), name)
    val batchAxis = Output.constantValue(dequeued.head.op.inputs(1)).get.scalar.asInstanceOf[Int]
    dequeued.head.op.outputs.zip(shapes).foreach(p => p._1.setShape(Shape(batchAxis) ++ p._2))
    dequeued
  }

  /** Creates an op that dequeues and concatenates up to `n` elements from this queue.
    *
    * The op is not supported by all queues. If a queue does not support it, then an unimplemented error is thrown.
    *
    * The op concatenates queue-element component tensors along the `0`th axis to make a single component tensor. All of
    * the components in the dequeued tuple will have size `n` in the `0`th axis.
    *
    * If the queue is closed and there are more than `0` but fewer than `n` elements remaining, then instead of raising
    * an out of range error like [[dequeueMany]], less than `n` elements are returned immediately. If the queue is
    * closed and there are `0` elements left in the queue, then an out of range error is thrown just like in
    * [[dequeueMany]]. Otherwise, the behavior is identical to [[dequeueMany]].
    *
    * @param  timeout If the queue contains less than `n` elements, the created op will block for up to `timeout` s, if
    *                 provided.
    * @param  name    Name for the created op.
    * @return Created op outputs.
    */
  def dequeueUpTo(n: Output, timeout: Option[Double] = None, name: String = s"$name/DequeueUpTo"): Seq[Output] = {
    val dequeued = Queue.queueDequeueUpTo(handle, n, timeout.map(t => (t * 1000).toInt), name)
    val batchAxis = Output.constantValue(dequeued.head.op.inputs(1)).get.scalar.asInstanceOf[Int]
    dequeued.head.op.outputs.zip(shapes).foreach(p => p._1.setShape(Shape(batchAxis) ++ p._2))
    dequeued
  }

  /** Creates an op that closes this queue.
    *
    * The op signals that no more elements will be enqueued in the queue. Subsequent enqueue operations will fail.
    * Subsequent dequeue operations will continue to succeed if sufficient elements remain in the queue. Subsequent
    * dequeue operations that would block will fail immediately.
    *
    * @param  cancelPendingEnqueues If `true`, all pending enqueue requests that are blocked on the queue will be
    *                               canceled.
    * @param  name                  Name for the created op.
    * @return Created op.
    */
  def close(cancelPendingEnqueues: Boolean = false, name: String = s"$name/Close"): Op = {
    Queue.queueClose(handle, cancelPendingEnqueues, name)
  }

  /** Creates an op that checks if this queue is closed.
    *
    * The op returns `true`-valued scalar tensor if the queue is closed and a `false`-valued one if the queue is open.
    *
    * @param  name Name for the created op.
    * @return Created op output, which is a `BOOLEAN` tensor.
    */
  def isClosed(name: String = s"$name/IsClosed"): Output = Queue.queueIsClosed(handle, name)

  /** Creates an op that computes the number of elements in this queue.
    *
    * @param  name        Name for the created op.
    * @return Created op output, which is an `INT32` scalar tensor.
    */
  def size(name: String = s"$name/Size"): Output = Queue.queueSize(handle, name)
}

private[api] object Queue {
  private[ops] trait API {
    /** Creates a FIFO queue.
      *
      * A FIFO queue is a queue that produces elements in first-in first-out order.
      *
      * A FIFO queue has bounded capacity; it supports multiple concurrent producers and consumers; and it provides
      * exactly-once delivery. It holds a list of up to `capacity` elements. Each element is a fixed-length tuple of
      * tensors whose data types are described by `componentTypes`, and whose shapes are optionally described by the
      * `componentShapes` argument. If the `componentShapes` argument is specified, each component of a queue element
      * must have the respective fixed shape. If it is unspecified, different queue elements may have different shapes,
      * but the use of [[Queue.dequeueMany]] is disallowed.
      *
      * @param  componentTypes  The data type of each component in a value.
      * @param  componentShapes The shape of each component in a value. The length of this sequence must be either `0`,
      *                         or the same as the length of `componentTypes`. If the length of this sequence is `0`,
      *                         the shapes of the queue elements are not constrained, and only one element may be
      *                         dequeued at a time.
      * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
      * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the queue.
      * @return Constructed queue.
      */
    def fifoQueue(
        componentTypes: Seq[DataType[_]],
        componentShapes: Seq[Shape] = Seq.empty,
        capacity: Int = -1,
        sharedName: String = "",
        name: String = "FIFOQueue"
    ): Queue = {
      if (componentTypes.isEmpty)
        throw new IllegalArgumentException("Cannot create a queue when no data types are provided.")
      val handle = createFifoQueue(componentTypes, componentShapes, capacity, sharedName = sharedName, name = name)
      new Queue(handle, componentTypes)(componentShapes)
    }

    /** Creates a padding FIFO queue.
      *
      * A padding FIFO queue is a queue that produces elements in first-in first-out order. It also allows variable-size
      * shapes, by setting the corresponding shape axes to `-1` in `componentShapes`. In this case,
      * [[Queue.dequeueMany]] will pad up to the maximum size of any given element in the dequeued batch.
      *
      * A FIFO queue has bounded capacity; it holds a list of up to `capacity` elements. Each element is a fixed-length
      * tuple of tensors whose data types are described by `componentTypes`, and whose shapes are described by the
      * `componentShapes` argument.
      *
      * In contrast to [[fifoQueue]], the `componentShapes` argument must be specified; each component of a queue
      * element must have the respective shape. Shapes of fixed rank but variable size are allowed by setting any shape
      * axis size to `-1`. In this case, the inputs' shape may vary along the given dimension, and [[Queue.dequeueMany]]
      * will pad the given dimension with zeros up to the maximum shape of all elements in the given batch.
      *
      * @param  componentTypes  The data type of each component in a value.
      * @param  componentShapes The shape of each component in a value. The length of this sequence must be the same as
      *                         the length of `componentTypes`. Shapes of fixed rank but variable size are allowed by
      *                         setting any shape dimension to `-1`. In this case, the inputs' shape may vary along the
      *                         given axis, and [[queueDequeueMany]] will pad the given axis with zeros up to the
      *                         maximum shape of all elements in the dequeued batch.
      * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
      * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the queue.
      * @return Constructed queue.
      */
    def paddingFifoQueue(
        componentTypes: Seq[DataType[_]],
        componentShapes: Seq[Shape] = Seq.empty,
        capacity: Int = -1,
        sharedName: String = "",
        name: String = "PaddingFIFOQueue"
    ): Queue = {
      if (componentTypes.isEmpty)
        throw new IllegalArgumentException("Cannot create a queue when no data types are provided.")
      if (componentTypes.size != componentShapes.size)
        throw new IllegalArgumentException(
          "Shapes must be provided for all components of a padding FIFO queue. Received: " +
              s"${componentTypes.size} data types and ${componentShapes.size} shapes.")
      val handle = createPaddingFifoQueue(
        componentTypes, componentShapes, capacity, sharedName = sharedName, name = name)
      new Queue(handle, componentTypes)(componentShapes)
    }

    /** Creates a priority queue.
      *
      * A priority queue is a queue that produces elements sorted by the first component value.
      *
      * A priority queue has bounded capacity; it supports multiple concurrent producers and consumers; and it provides
      * exactly-once delivery. It holds a list of up to `capacity` elements. Each element is a fixed-length tuple of
      * tensors whose data types are described by `componentTypes`, and whose shapes are optionally described by the
      * `componentShapes` argument. If the `componentShapes` argument is specified, each component of a queue element
      * must have the respective fixed shape. If it is unspecified, different queue elements may have different shapes,
      * but the use of [[Queue.dequeueMany]] is disallowed.
      *
      * Note that the priority queue requires the first component of any element to be a scalar `INT64` tensor, in
      * addition to the other elements declared by `componentTypes`. Therefore calls to [[Queue.enqueue]] and
      * [[Queue.enqueueMany]] (and respectively to [[Queue.dequeue]] and [[Queue.dequeueMany]] on a priority queue will
      * all require (and respectively output) one extra entry in their input (and respectively output) sequences.
      *
      * @param  componentTypes  The data type of each component in a value.
      * @param  componentShapes The shape of each component in a value. The length of this sequence must be either `0`,
      *                         or the same as the length of `componentTypes`. If the length of this sequence is `0`,
      *                         the shapes of the queue elements are not constrained, and only one element may be
      *                         dequeued at a time.
      * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
      * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the queue.
      * @return Constructed queue.
      */
    def priorityQueue(
        componentTypes: Seq[DataType[_]],
        componentShapes: Seq[Shape] = Seq.empty,
        capacity: Int = -1,
        sharedName: String = "",
        name: String = "PriorityQueue"
    ): Queue = {
      if (componentTypes.isEmpty)
        throw new IllegalArgumentException("Cannot create a queue when no data types are provided.")
      val handle = createPriorityQueue(componentTypes, componentShapes, capacity, sharedName = sharedName, name = name)
      val dataTypes = INT64 +: componentTypes
      val shapes = if (componentShapes.nonEmpty) Shape() +: componentShapes else componentShapes
      new Queue(handle, dataTypes)(shapes)
    }

    /** Creates a random shuffling queue.
      *
      * A random shuffling queue is a queue that randomizes the order of the elements.
      *
      * A random shuffling queue has bounded capacity; it supports multiple concurrent producers and consumers; and it
      * provides exactly-once delivery. It holds a list of up to `capacity` elements. Each element is a fixed-length
      * tuple of tensors whose data types are described by `componentTypes`, and whose shapes are optionally described
      * by the `componentShapes` argument. If the `componentShapes` argument is specified, each component of a queue
      * element must have the respective fixed shape. If it is unspecified, different queue elements may have different
      * shapes, but the use of [[Queue.dequeueMany]] is disallowed.
      *
      * The `minAfterDequeue` argument allows the caller to specify a minimum number of elements that will remain in the
      * queue after a [[Queue.dequeue]] or [[Queue.dequeueMany]] operation completes, in order to ensure a minimum level
      * of mixing of elements. This invariant is maintained by blocking those operations until a sufficient number of
      * elements have been enqueued. The `minAfterDequeue` argument is ignored after the queue has been closed.
      *
      * @param  componentTypes  The data type of each component in a value.
      * @param  componentShapes The shape of each component in a value. The length of this sequence must be either `0`,
      *                         or the same as the length of `componentTypes`. If the length of this sequence is `0`,
      *                         the shapes of the queue elements are not constrained, and only one element may be
      *                         dequeued at a time.
      * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
      * @param  minAfterDequeue If specified, this argument allows the caller to specify a minimum number of elements
      *                         that will remain in the queue after a [[Queue.dequeue]] or [[Queue.dequeueMany]]
      *                         operation completes, in order to ensure a minimum level of mixing of elements. This
      *                         invariant is maintained by blocking those operations until a sufficient number of
      *                         elements have been enqueued. The argument is ignored after the queue has been closed.
      * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name
      *                         across multiple sessions.
      * @param  name            Name for the queue.
      * @return Constructed queue.
      */
    def randomShuffleQueue(
        componentTypes: Seq[DataType[_]],
        componentShapes: Seq[Shape] = Seq.empty,
        capacity: Int = -1,
        minAfterDequeue: Int = 0,
        seed: Option[Int] = None,
        sharedName: String = "",
        name: String = "RandomShuffleQueue"
    ): Queue = {
      val (graphSeed, opSeed) = Op.currentGraphRandomSeed(seed)
      val (seed1: Int, seed2: Int) = {
        if (graphSeed.isEmpty && opSeed.isEmpty) {
          (0, 0)
        } else if (seed.isEmpty && sharedName != "") {
          // This means that a graph-level seed is provided but no op-level seed is provided. If sharedName is also
          // provided, we make seed2 depend only on the graph-level seed and sharedName (seed2 from
          // Op.currentGraphRandomSeed() is generally dependent on the current number of ops in the graph).
          val s1 = graphSeed.getOrElse(0)
          val s2 = ByteBuffer.wrap(MessageDigest.getInstance("MD5").digest(s"$s1".getBytes())).getInt & 0x7FFFFFFF
          (s1, s2)
        } else {
          (graphSeed.getOrElse(0), opSeed.getOrElse(0))
        }
      }
      val handle = createRandomShuffleQueue(
        componentTypes, componentShapes, capacity, minAfterDequeue, seed1, seed2, sharedName = sharedName, name = name)
      new Queue(handle, componentTypes)(componentShapes)
    }
  }

  /** Creates a queue using the queue reference from `queues(index)`.
    *
    * @param  index  Integer scalar tensor that determines the input that gets selected.
    * @param  queues Sequence of queues with the same component data types and names.
    * @return Created queue.
    * @throws IllegalArgumentException If the provided queues do not have matching component data types or names.
    */
  @throws[IllegalArgumentException]
  def fromSeq(index: Output, queues: Seq[Queue]): Queue = {
    val dataTypes = queues.head.dataTypes
    if (queues.tail.exists(_.dataTypes != dataTypes))
      throw new IllegalArgumentException("The provided queues do not have matching component data types.")
    val shapes = dataTypes.indices.map(i => queues.map(_.shapes(i)).reduce(commonShape))
    val handle = Basic.gather(Basic.stack(queues.map(_.handle)), index)
    new Queue(handle, dataTypes)(shapes)
  }

  /** Returns the greatest lower bound (ordered by specificity) shape, between `shape1` and `shape2`. */
  private[this] def commonShape(shape1: Shape, shape2: Shape): Shape = {
    if (shape1.rank == -1 || shape2.rank == -1 || shape1.rank != shape2.rank)
      Shape.unknown()
    else
      Shape.fromSeq(shape1.asArray.zip(shape2.asArray).map({ case (s1, s2) => if (s1 != -1 && s1 == s2) s1 else -1 }))
  }

  /** Creates an op that constructs a FIFO queue.
    *
    * A FIFO queue is a queue that produces elements in first-in first-out order.
    *
    * @param  componentTypes  The data type of each component in a value.
    * @param  componentShapes The shape of each component in a value. The length of this sequence must be either `0`, or
    *                         the same as the length of `componentTypes`. If the length of this sequence is `0`, the
    *                         shapes of the queue elements are not constrained, and only one element may be dequeued at
    *                         a time.
    * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
    * @param  container       If non-empty, then the constructed queue is placed in the provided container. Otherwise, a
    *                         default container is used.
    * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name across
    *                         multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is the handle to constructed queue.
    */
  private[ops] def createFifoQueue(
      componentTypes: Seq[DataType[_]],
      componentShapes: Seq[Shape] = Seq.empty,
      capacity: Int = -1,
      container: String = "",
      sharedName: String = "",
      name: String = "FIFOQueue"
  ): Output = {
    Op.Builder(opType = "FIFOQueueV2", name = name)
        .setAttribute("component_types", componentTypes.toArray)
        .setAttribute("shapes", componentShapes.toArray)
        .setAttribute("capacity", capacity)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that constructs a padding FIFO queue.
    *
    * A padding FIFO queue is a queue that produces elements in first-in first-out order. It also allows variable-size
    * shapes, by setting the corresponding shape axes to `-1` in `componentShapes`. In this case, [[queueDequeueMany]]
    * will pad up to the maximum size of any given element in the dequeued batch.
    *
    * @param  componentTypes  The data type of each component in a value.
    * @param  componentShapes The shape of each component in a value. The length of this sequence must be either `0`, or
    *                         the same as the length of `componentTypes`. Shapes of fixed rank but variable size are
    *                         allowed by setting any shape dimension to `-1`. In this case, the inputs' shape may vary
    *                         along the given axis, and [[queueDequeueMany]] will pad the given axis with zeros up to
    *                         the maximum shape of all elements in the dequeued batch. If the length of this sequence is
    *                         `0`, the shapes of the queue elements are not constrained, and only one element may be
    *                         dequeued at a time.
    * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
    * @param  container       If non-empty, then the constructed queue is placed in the provided container. Otherwise, a
    *                         default container is used.
    * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name across
    *                         multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is the handle to constructed queue.
    */
  private[ops] def createPaddingFifoQueue(
      componentTypes: Seq[DataType[_]],
      componentShapes: Seq[Shape] = Seq.empty,
      capacity: Int = -1,
      container: String = "",
      sharedName: String = "",
      name: String = "PaddingFIFOQueue"
  ): Output = {
    Op.Builder(opType = "PaddingFIFOQueueV2", name = name)
        .setAttribute("component_types", componentTypes.toArray)
        .setAttribute("shapes", componentShapes.toArray)
        .setAttribute("capacity", capacity)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that constructs a priority queue.
    *
    * A priority queue is a queue that produces elements sorted by the first component value.
    *
    * Note that the priority queue requires the first component of any element to be a scalar `INT64` tensor, in
    * addition to the other elements declared by `componentTypes`. Therefore calls to [[queueEnqueue]] and
    * [[queueEnqueueMany]] (and respectively to [[queueDequeue]] and [[queueDequeueMany]] on a priority queue will all
    * require (and respectively output) one extra entry in their input (and respectively output) sequences.
    *
    * @param  componentTypes  The data type of each component in a value.
    * @param  componentShapes The shape of each component in a value. The length of this sequence must be the same as
    *                         the length of `componentTypes`. Shapes of fixed rank but variable size are allowed by
    *                         setting any shape dimension to `-1`. In this case, the inputs' shape may vary along the
    *                         given axis, and [[queueDequeueMany]] will pad the given axis with zeros up to the maximum
    *                         shape of all elements in the dequeued batch.
    * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
    * @param  container       If non-empty, then the constructed queue is placed in the provided container. Otherwise, a
    *                         default container is used.
    * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name across
    *                         multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is the handle to constructed queue.
    */
  private[ops] def createPriorityQueue(
      componentTypes: Seq[DataType[_]] = Seq.empty,
      componentShapes: Seq[Shape] = Seq.empty,
      capacity: Int = -1,
      container: String = "",
      sharedName: String = "",
      name: String = "PriorityQueue"
  ): Output = {
    Op.Builder(opType = "PriorityQueueV2", name = name)
        .setAttribute("component_types", componentTypes.toArray)
        .setAttribute("shapes", componentShapes.toArray)
        .setAttribute("capacity", capacity)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that constructs a random shuffling queue.
    *
    * A random shuffling queue is a queue that randomizes the order of the elements.
    *
    * @param  componentTypes  The data type of each component in a value.
    * @param  componentShapes The shape of each component in a value. The length of this sequence must be either `0`, or
    *                         the same as the length of `componentTypes`. If the length of this sequence is `0`, the
    *                         shapes of the queue elements are not constrained, and only one element may be dequeued at
    *                         a time.
    * @param  capacity        Upper bound on the number of elements in this queue. Negative numbers imply no bounds.
    * @param  minAfterDequeue If specified, this argument allows the caller to specify a minimum number of elements that
    *                         will remain in the queue after a [[queueDequeue]] or [[queueDequeueMany]] operation
    *                         completes, in order to ensure a minimum level of mixing of elements. This invariant is
    *                         maintained by blocking those operations until a sufficient number of elements have been
    *                         enqueued. The argument is ignored after the queue has been closed.
    * @param  seed1           If either `seed1` or `seed2` is set to be non-zero, the random number generator is seeded
    *                         by the provided seed. Otherwise, a random seed is used.
    * @param  seed2           If either `seed1` or `seed2` is set to be non-zero, the random number generator is seeded
    *                         by the provided seed. Otherwise, a random seed is used.
    * @param  container       If non-empty, then the constructed queue is placed in the provided container. Otherwise, a
    *                         default container is used.
    * @param  sharedName      If non-empty, then the constructed queue will be shared under the the provided name across
    *                         multiple sessions.
    * @param  name            Name for the created op.
    * @return Created op output, which is the handle to constructed queue.
    */
  private[ops] def createRandomShuffleQueue(
      componentTypes: Seq[DataType[_]],
      componentShapes: Seq[Shape] = Seq.empty,
      capacity: Int = -1,
      minAfterDequeue: Int = 0,
      seed1: Int = 0,
      seed2: Int = 0,
      container: String = "",
      sharedName: String = "",
      name: String = "RandomShuffleQueue"
  ): Output = {
    Op.Builder(opType = "RandomShuffleQueueV2", name = name)
        .setAttribute("component_types", componentTypes.toArray)
        .setAttribute("shapes", componentShapes.toArray)
        .setAttribute("capacity", capacity)
        .setAttribute("min_after_dequeue", minAfterDequeue)
        .setAttribute("seed", seed1)
        .setAttribute("seed2", seed2)
        .setAttribute("container", container)
        .setAttribute("shared_name", sharedName)
        .build().outputs(0)
  }

  /** Creates an op that enqueues a sequence of one or more tensors in a queue.
    *
    * The `values` sequence has `k` elements, which correspond to the components of tuples stored in the provided queue.
    *
    * **Note:** If the queue is full, this operation will block until the provided element has been enqueued (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  values      Sequence of one or more tensors from which the enqueued tensors should be taken.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[ops] def queueEnqueue(
      queueHandle: Output,
      values: Seq[Output],
      timeout: Option[Int] = None,
      name: String = "QueueEnqueue"
  ): Op = {
    Op.Builder(opType = "QueueEnqueueV2", name = name)
        .addInput(queueHandle)
        .addInputList(values)
        .setAttribute("timeout_ms", timeout.getOrElse(-1))
        .build()
  }

  /** Creates an op that enqueues zero or more tuples of a sequence of one or more tensors in a queue.
    *
    * The op slices each component tensor along the `0`th axis to make multiple queue elements. All of the sequence
    * components must have the same size in the `0`th axis.
    *
    * The `values` sequence has `k` elements, which correspond to the components of tuples stored in the provided queue.
    *
    * **Note:** If the queue is full, this operation will block until the provided element has been enqueued (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  values      Sequence of one or more tensors from which the enqueued tensors should be taken.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op.
    */
  private[ops] def queueEnqueueMany(
      queueHandle: Output,
      values: Seq[Output],
      timeout: Option[Int] = None,
      name: String = "QueueEnqueueMany"
  ): Op = {
    Op.Builder(opType = "QueueEnqueueManyV2", name = name)
        .addInput(queueHandle)
        .addInputList(values)
        .setAttribute("timeout_ms", timeout.getOrElse(-1))
        .build()
  }

  /** Creates an op that dequeues a sequence of one or more tensors from a queue.
    *
    * The op has `k` outputs, where `k` is the number of components in the tuples stored in the provided queue, and
    * output `i` is the `i`th component of the dequeued tuple.
    *
    * **Note:** If the queue is empty, this operation will block until there is an element to dequeue (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    */
  private[ops] def queueDequeue(
      queueHandle: Output,
      timeout: Option[Int] = None,
      name: String = "QueueDequeue"
  ): Seq[Output] = {
    Op.Builder(opType = "QueueDequeueV2", name = name)
        .addInput(queueHandle)
        .setAttribute("timeout_ms", timeout.getOrElse(-1))
        .build().outputs.toSeq
  }

  /** Creates an op that dequeues `n` tuples of a sequence of one or more tensors from a queue.
    *
    * If the queue is closed and there are fewer than `n` elements, then an `OutOfRange` error is returned.
    *
    * The op concatenates queue-element component tensors along the `0`th dimension to make a single component tensor.
    * All of the components in the dequeued tuple will have size `n` in the `0`th dimension.
    *
    * The op has `k` outputs, where `k` is the number of components in the tuples stored in the provided queue, and
    * output `i` is the `i`th component of the dequeued tuple.
    *
    * **Note:** If the queue is empty, this operation will block until there is an element to dequeue (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  n           Number of tuples to dequeue.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    */
  private[ops] def queueDequeueMany(
      queueHandle: Output,
      n: Output,
      timeout: Option[Int] = None,
      name: String = "QueueDequeueMany"
  ): Seq[Output] = {
    Op.Builder(opType = "QueueDequeueManyV2", name = name)
        .addInput(queueHandle)
        .addInput(n)
        .setAttribute("timeout_ms", timeout.getOrElse(-1))
        .build().outputs.toSeq
  }

  /** Creates an op that dequeues up to `n` tuples of a sequence of one or more tensors from a queue.
    *
    * The op is not supported by all queues. If a queue does not support it, then an `Unimplemented` error is returned.
    *
    * If the queue is closed and there are more than `0` but less than `n` elements remaining, then instead of returning
    * an `OutOfRange` error like [[queueDequeueMany]], less than `n` elements are returned immediately. If the queue is
    * closed and there are `0` elements left in the queue, then an `OutOfRange` error is returned just like in
    * [[queueDequeueMany]]. Otherwise, the behavior is identical to [[queueDequeueMany]].
    *
    * The op concatenates queue-element component tensors along the `0`th dimension to make a single component tensor.
    * All of the components in the dequeued tuple will have size `n` in the `0`th dimension.
    *
    * The op has `k` outputs, where `k` is the number of components in the tuples stored in the provided queue, and
    * output `i` is the `i`th component of the dequeued tuple.
    *
    * **Note:** If the queue is empty, this operation will block until there is an element to dequeue (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  n           Number of tuples to dequeue.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    */
  private[ops] def queueDequeueUpTo(
      queueHandle: Output,
      n: Output,
      timeout: Option[Int] = None,
      name: String = "QueueDequeueUpTo"
  ): Seq[Output] = {
    Op.Builder(opType = "QueueDequeueUpToV2", name = name)
        .addInput(queueHandle)
        .addInput(n)
        .setAttribute("timeout_ms", timeout.getOrElse(-1))
        .build().outputs.toSeq
  }

  /** Creates an op that closes the provided queue.
    *
    * The op signals that no more elements will be enqueued in the provided queue. Subsequent enqueue operations will
    * fail. Subsequent dequeue operations will continue to succeed if sufficient elements remain in the queue.
    * Subsequent dequeue operations that would block will fail immediately.
    *
    * @param  queueHandle           Handle to a queue.
    * @param  cancelPendingEnqueues If `true`, all pending enqueue requests that are blocked on the provided queue will
    *                               be canceled.
    * @param  name                  Name for the created op.
    * @return Created op.
    */
  private[ops] def queueClose(
      queueHandle: Output,
      cancelPendingEnqueues: Boolean = false,
      name: String = "QueueClose"
  ): Op = {
    Op.Builder(opType = "QueueCloseV2", name = name)
        .addInput(queueHandle)
        .setAttribute("cancel_pending_enqueues", cancelPendingEnqueues)
        .build()
  }

  /** Creates an op that checks if the provided queue is closed.
    *
    * The op returns `true`-valued scalar tensor if the queue is closed and a `false`-valued one if the queue is open.
    *
    * @param  queueHandle Handle to a queue.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  private[ops] def queueIsClosed(queueHandle: Output, name: String = "QueueIsClosed"): Output = {
    Op.Builder(opType = "QueueIsClosedV2", name = name)
        .addInput(queueHandle)
        .build().outputs(0)
  }

  /** Creates an op that computes the number of elements in the provided queue.
    *
    * The op returns an `INT32` scalar tensor.
    *
    * @param  queueHandle Handle to a queue.
    * @param  name        Name for the created op.
    * @return Created op output.
    */
  private[ops] def queueSize(queueHandle: Output, name: String = "QueueSize"): Output = {
    Op.Builder(opType = "QueueSizeV2", name = name)
        .addInput(queueHandle)
        .build().outputs(0)
  }
}
