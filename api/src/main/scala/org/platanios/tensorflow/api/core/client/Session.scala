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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.core.{Graph, client}
import org.platanios.tensorflow.api.ops.{Op, OpCreationContext, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.{Closeable, Disposer}
import org.platanios.tensorflow.jni.{Session => NativeSession, Tensor => NativeTensor}

import org.tensorflow.framework.{RunMetadata, RunOptions}

import scala.util.DynamicVariable

/** Sessions provide the client interface for interacting with TensorFlow computations.
  *
  * The [[Session]] class enables incremental graph building with inline execution of ops and evaluation of tensors.
  *
  * @author Emmanouil Antonios Platanios
  */
final case class Session private (
    private val graphReference: Graph#Reference, private var nativeHandle: Long) extends Closeable {
  val graph: Graph = graphReference.graph

  private[this] object NativeHandleLock
  private[this] var referenceCount: Int = 0

  // Keep track of references in the Scala side and notify the native library when the session is not referenced
  // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
  // potential memory leak.
  Disposer.add(this, () => this.close())

  /** Runs ops and evaluates tensors in `fetches`, and returns the values of the evaluated tensors.
    *
    * This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every
    * `Op` and evaluate every `Output` in `fetches`, substituting the values in `feeds` for the corresponding input
    * values.
    *
    * @param  feeds   Optional feed map. This argument can be used to feed values into a TensorFlow session. It is a map
    *                 from graph nodes to their corresponding [[Tensor]] values. More specifically, for all allowed
    *                 types for this argument, please refer to the documentation of the [[Feedable]] type class.
    * @param  fetches Optional argument specifying which values to fetch from the TensorFlow session. This argument can
    *                 have various forms and its type defines the return type of this function. Please refer to the
    *                 documentation of the [[Fetchable]] type class for details on the allowed types.
    * @param  targets Optional argument specifying which ops to execute in the TensorFlow graph, without returning their
    *                 value. Please refer to the documentation of the [[Executable]] type class for details on the
    *                 allowed types of `targets`.
    * @param  options Optional [[RunOptions]] protocol buffer that allows controlling the behavior of this particular
    *                 run (e.g., turning tracing on).
    * @return The evaluated tensors using the structure of `fetches`. For more details on the return type, please refer
    *         to the documentation of the [[Fetchable]] type class.
    * @throws IllegalStateException If this session has already been closed.
    */
  @throws[IllegalStateException]
  def run[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output],
      targets: E = Traversable.empty[Op], options: RunOptions = null)
      (implicit executable: Executable[E], fetchable: Fetchable.Aux[F, R]): R = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, options = options)._1
  }

  /** Runs ops and evaluates tensors in `fetches`, and returns the values of the evaluated tensors, along with any run
    * metadata that may have been collected.
    *
    * This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every
    * [[Op]] and evaluate every [[Output]] in `fetches`, substituting the values in `feeds` for the corresponding input
    * values.
    *
    * When appropriate (e.g., when users turn on tracing in `options`), the run metadata output of this run will be
    * collected in a [[RunMetadata]] protocol buffer and returned by this function.
    *
    * @param  feeds   Optional feed map. This argument can be used to feed values into a TensorFlow session. It is a map
    *                 from graph nodes to their corresponding [[Tensor]] values. More specifically, for all allowed
    *                 types for this argument, please refer to the documentation of the [[Feedable]] type class.
    * @param  fetches Optional argument specifying which values to fetch from the TensorFlow session. This argument can
    *                 have various forms and its type defines the return type of this function. Please refer to the
    *                 documentation of the [[Fetchable]] type class for details on the allowed types.
    * @param  targets Optional argument specifying which ops to execute in the TensorFlow graph, without returning their
    *                 value. Please refer to the documentation of the [[Executable]] type class for details on the
    *                 allowed types of `targets`.
    * @param  options Optional [[RunOptions]] protocol buffer that allows controlling the behavior of this particular
    *                 run (e.g., turning tracing on).
    * @return A tuple containing two elements:
    *         - The evaluated tensors using the structure of `fetches`. For more details on the return type, please
    *           refer to the documentation of the [[Fetchable]] type class.
    *         - A [[RunMetadata]] protocol buffer option containing the collected run metadata, if any.
    * @throws IllegalStateException If the session has already been closed.
    */
  @throws[IllegalStateException]
  def runWithMetadata[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null)
      (implicit executable: Executable[E], fetchable: Fetchable.Aux[F, R]): (R, Option[RunMetadata]) = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, options = options, wantMetadata = true)
  }

  /** Helper method for [[run]] and [[runWithMetadata]]. */
  @throws[IllegalStateException]
  private[this] def runHelper[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Output], targets: E = Traversable.empty[Op],
      options: RunOptions = null, wantMetadata: Boolean = false)
      (implicit executable: Executable[E], fetchable: Fetchable.Aux[F, R]): (R, Option[RunMetadata]) = {
    if (nativeHandle == 0)
      throw new IllegalStateException("This session has already been closed.")
    val (inputs, inputTensors) = feeds.values.toSeq.unzip
    val inputTensorHandles: Array[Long] = inputTensors.map(_.resolve()).toArray
    val inputOpHandles: Array[Long] = inputs.map(_.op.nativeHandle).toArray
    val inputOpIndices: Array[Int] = inputs.map(_.index).toArray
    val (uniqueFetches, resultsBuilder) = Fetchable.process(fetches)(fetchable)
    val outputOpHandles: Array[Long] = uniqueFetches.map(_.op.nativeHandle).toArray
    val outputOpIndices: Array[Int] = uniqueFetches.map(_.index).toArray
    val outputTensorHandles: Array[Long] = Array.ofDim[Long](uniqueFetches.length)
    val targetOpHandles: Array[Long] = executable.ops(targets).map(_.nativeHandle).toArray
    NativeHandleLock.synchronized {
      if (nativeHandle == 0)
        throw new IllegalStateException("close() has been called on the session.")
      referenceCount += 1
    }
    val metadata: Array[Byte] = NativeSession.run(
      handle = nativeHandle,
      runOptions = if (options != null) options.toByteArray else Array.empty[Byte],
      inputTensorHandles = inputTensorHandles,
      inputOpHandles = inputOpHandles,
      inputOpIndices = inputOpIndices,
      outputOpHandles = outputOpHandles,
      outputOpIndices = outputOpIndices,
      targetOpHandles = targetOpHandles,
      wantRunMetadata = wantMetadata,
      outputTensorHandles = outputTensorHandles)
    val outputs: R = resultsBuilder(outputTensorHandles.map(handle => {
      val tensor = Tensor.fromHostNativeHandle(handle)
      NativeTensor.delete(handle)
      tensor
    }))
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        referenceCount -= 1
        if (referenceCount == 0)
          NativeHandleLock.notifyAll()
      }
    }
    inputTensorHandles.foreach(NativeTensor.delete)
    (outputs, Option(metadata).map(RunMetadata.parseFrom))
  }

  /** Closes this [[Session]] and releases any resources associated with it. Note that a [[Session]] is not usable after
    * it has been closed. */
  override def close(): Unit = {
    graphReference.close()
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        while (referenceCount > 0) {
          try {
            NativeHandleLock.wait()
          } catch {
            case _: InterruptedException =>
              Thread.currentThread().interrupt()
              // TODO: [CLIENT] Possible leak of the session and graph in this case?
              return
          }
        }
        NativeSession.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

/** Contains helper functions for managing [[Session]] instances. */
object Session {
  private[client] trait API {
    type Session = client.Session
    val Session: client.Session.type = client.Session
  }

  def apply()(implicit context: DynamicVariable[OpCreationContext]): Session = {
    val graph = context.graph
    val graphReference = graph.reference
    Session(graphReference, NativeSession.allocate(graphReference.nativeHandle))
  }

  def apply(graph: Graph): Session = {
    val graphReference = graph.reference
    Session(graphReference, NativeSession.allocate(graphReference.nativeHandle))
  }
}
