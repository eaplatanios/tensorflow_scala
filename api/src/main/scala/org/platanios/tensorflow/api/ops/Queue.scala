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
import org.platanios.tensorflow.api.ops.Gradients.{Registry => GradientsRegistry}
import org.platanios.tensorflow.api.types.DataType

/** Contains functions for constructing ops related to queues.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Queue {
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
  private[api] def fifoQueue(
      componentTypes: Seq[DataType], componentShapes: Seq[Shape] = Seq.empty, capacity: Int = -1,
      container: String = "", sharedName: String = "", name: String = "FIFOQueue"): Output = {
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
  private[api] def paddingFifoQueue(
      componentTypes: Seq[DataType], componentShapes: Seq[Shape] = Seq.empty, capacity: Int = -1,
      container: String = "", sharedName: String = "", name: String = "PaddingFIFOQueue"): Output = {
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
  private[api] def priorityQueue(
      componentTypes: Seq[DataType] = Seq.empty, componentShapes: Seq[Shape] = Seq.empty, capacity: Int = -1,
      container: String = "", sharedName: String = "", name: String = "PriorityQueue"): Output = {
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
  private[api] def randomShuffleQueue(
      componentTypes: Seq[DataType], componentShapes: Seq[Shape] = Seq.empty, capacity: Int = -1, seed1: Int = 0,
      seed2: Int = 0, container: String = "", sharedName: String = "", name: String = "RandomShuffleQueue"): Output = {
    Op.Builder(opType = "RandomShuffleQueueV2", name = name)
        .setAttribute("component_types", componentTypes.toArray)
        .setAttribute("shapes", componentShapes.toArray)
        .setAttribute("capacity", capacity)
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
  private[api] def queueEnqueue(
      queueHandle: Output, values: Seq[Output], timeout: Option[Int] = None, name: String = "QueueEnqueue"): Op = {
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
  private[api] def queueEnqueueMany(
      queueHandle: Output, values: Seq[Output], timeout: Option[Int] = None, name: String = "QueueEnqueueMany"): Op = {
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
    * **Note:** If the queue is full, this operation will block until the provided element has been enqueued (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    */
  private[api] def queueDequeue(
      queueHandle: Output, timeout: Option[Int] = None, name: String = "QueueDequeue"): Seq[Output] = {
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
    * **Note:** If the queue is full, this operation will block until the provided element has been enqueued (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  n           Number of tuples to dequeue.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    */
  private[api] def queueDequeueMany(
      queueHandle: Output, n: Output, timeout: Option[Int] = None, name: String = "QueueDequeueMany"): Seq[Output] = {
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
    * **Note:** If the queue is full, this operation will block until the provided element has been enqueued (or until
    * `timeout` ms elapses, if specified).
    *
    * @param  queueHandle Handle to a queue.
    * @param  n           Number of tuples to dequeue.
    * @param  timeout     If the queue is full, the created op will block for up to `timeout` ms, if provided.
    * @param  name        Name for the created op.
    * @return Created op outputs.
    */
  private[api] def queueDequeueUpTo(
      queueHandle: Output, n: Output, timeout: Option[Int] = None, name: String = "QueueDequeueUpTo"): Seq[Output] = {
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
  private[api] def queueClose(
      queueHandle: Output, cancelPendingEnqueues: Boolean = false, name: String = "QueueClose"): Op = {
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
  private[api] def queueIsClosed(queueHandle: Output, name: String = "QueueIsClosed"): Output = {
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
  private[api] def queueSize(queueHandle: Output, name: String = "QueueSize"): Output = {
    Op.Builder(opType = "QueueSizeV2", name = name)
        .addInput(queueHandle)
        .build().outputs(0)
  }
}

object Queue extends Queue {
  private[api] object Gradients {
    GradientsRegistry.registerNonDifferentiable("FIFOQueue")
    GradientsRegistry.registerNonDifferentiable("FIFOQueueV2")
    GradientsRegistry.registerNonDifferentiable("PaddingFIFOQueue")
    GradientsRegistry.registerNonDifferentiable("PaddingFIFOQueueV2")
    GradientsRegistry.registerNonDifferentiable("PriorityQueue")
    GradientsRegistry.registerNonDifferentiable("PriorityQueueV2")
    GradientsRegistry.registerNonDifferentiable("RandomShuffleQueue")
    GradientsRegistry.registerNonDifferentiable("RandomShuffleQueueV2")
    GradientsRegistry.registerNonDifferentiable("FakeQueue")
    GradientsRegistry.registerNonDifferentiable("QueueEnqueue")
    GradientsRegistry.registerNonDifferentiable("QueueEnqueueV2")
    GradientsRegistry.registerNonDifferentiable("QueueEnqueueMany")
    GradientsRegistry.registerNonDifferentiable("QueueEnqueueManyV2")
    GradientsRegistry.registerNonDifferentiable("QueueDequeue")
    GradientsRegistry.registerNonDifferentiable("QueueDequeueV2")
    GradientsRegistry.registerNonDifferentiable("QueueDequeueMany")
    GradientsRegistry.registerNonDifferentiable("QueueDequeueManyV2")
    GradientsRegistry.registerNonDifferentiable("QueueDequeueUpTo")
    GradientsRegistry.registerNonDifferentiable("QueueDequeueUpToV2")
    GradientsRegistry.registerNonDifferentiable("QueueClose")
    GradientsRegistry.registerNonDifferentiable("QueueCloseV2")
    GradientsRegistry.registerNonDifferentiable("QueueIsClosed")
    GradientsRegistry.registerNonDifferentiable("QueueIsClosedV2")
    GradientsRegistry.registerNonDifferentiable("QueueSize")
    GradientsRegistry.registerNonDifferentiable("QueueSizeV2")
  }
}
