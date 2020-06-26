/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.client.SessionConfig.{L1GraphOptimizerGlobalJIT, L2GraphOptimizerGlobalJIT}
import org.platanios.tensorflow.api.implicits.helpers._
import org.platanios.tensorflow.api.ops.{Op, Output, UntypedOp}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.{Closeable, DefaultsTo, Disposer, NativeHandleWrapper}
import org.platanios.tensorflow.jni.{TensorFlow, Session => NativeSession, Tensor => NativeTensor}
import org.platanios.tensorflow.proto.{RunMetadata, RunOptions}

import scala.collection.compat.immutable.ArraySeq
import scala.collection.mutable

/** Sessions provide the client interface for interacting with TensorFlow computations.
  *
  * The [[Session]] class enables incremental graph building with inline execution of ops and evaluation of tensors.
  *
  * @param  graphReference      Reference to the underlying graph of this session.
  * @param  target              Process to which this session will connect.
  * @param  nativeHandleWrapper Wrapper around the pointer to the native server object.
  * @param  closeFn             Function used to delete the native server object (i.e., free relevant memory).
  *
  * @author Emmanouil Antonios Platanios
  */
class Session private[api](
    private[api] val graphReference: Graph#Reference,
    val target: String = "",
    private[api] val nativeHandleWrapper: NativeHandleWrapper,
    override protected val closeFn: () => Unit
) extends Closeable {
  val graph: Graph = graphReference.graph

  /** Lock for the native handle. */
  private[Session] def NativeHandleLock = nativeHandleWrapper.Lock

  /** Native handle of this tensor. */
  private[api] def nativeHandle: Long = nativeHandleWrapper.handle

  /** Runs ops and evaluates tensors in `fetches`, and returns the values of the evaluated tensors.
    *
    * This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every
    * `Op` and evaluate every `Output` in `fetches`, substituting the values in `feeds` for the corresponding input
    * values.
    *
    * @param  feeds   Optional feed map. This argument can be used to feed values into a TensorFlow session. It is a map
    *                 from graph nodes to their corresponding [[Tensor]] values. More specifically, for all allowed
    *                 types for this argument, please refer to the documentation of the nested structure type class.
    * @param  fetches Optional argument specifying which values to fetch from the TensorFlow session. This argument can
    *                 have various forms and its type defines the return type of this function. Please refer to the
    *                 documentation of the nested structure type class for details on the allowed types.
    * @param  targets Optional argument specifying which ops to execute in the TensorFlow graph, without returning their
    *                 value.
    * @param  options Optional [[RunOptions]] protocol buffer that allows controlling the behavior of this particular
    *                 run (e.g., turning tracing on).
    * @return The evaluated tensors using the structure of `fetches`. For more details on the return type, please refer
    *         to the documentation of the nested structure type class.
    * @throws IllegalStateException If this session has already been closed.
    */
  @throws[IllegalStateException]
  def run[F: Session.DefaultFetches : OutputStructure, V, E: Session.DefaultTargets : OpStructure](
      feeds: FeedMap = FeedMap.empty,
      fetches: F = Seq.empty[Output[Any]],
      targets: E = Set.empty[UntypedOp],
      options: Option[RunOptions] = None
  )(implicit
      evOutputToTensor: OutputToTensor.Aux[F, V]
  ): V = {
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
    *                 documentation of the nested structure type class for details on the allowed types.
    * @param  targets Optional argument specifying which ops to execute in the TensorFlow graph, without returning their
    *                 value. Please refer to the documentation of the [[NestedStructureOps]] type class for details on
    *                 the allowed types of `targets`.
    * @param  options Optional [[RunOptions]] protocol buffer that allows controlling the behavior of this particular
    *                 run (e.g., turning tracing on).
    * @return   A tuple containing two elements:
    *           - The evaluated tensors using the structure of `fetches`. For more details on the return type, please
    *           refer to the documentation of the nested structure type class.
    *           - A [[RunMetadata]] protocol buffer option containing the collected run metadata, if any.
    * @throws IllegalStateException If the session has already been closed.
    */
  @throws[IllegalStateException]
  def runWithMetadata[F: Session.DefaultFetches : OutputStructure, V, E: Session.DefaultTargets : OpStructure](
      feeds: FeedMap = FeedMap.empty,
      fetches: F = Seq.empty[Output[Any]],
      targets: E = Set.empty[UntypedOp],
      options: Option[RunOptions] = None
  )(implicit
      evOutputToTensor: OutputToTensor.Aux[F, V]
  ): (V, Option[RunMetadata]) = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, options = options, wantMetadata = true)
  }

  /** Helper method for [[run]] and [[runWithMetadata]]. */
  @throws[IllegalStateException]
  private[api] def runHelper[F: Session.DefaultFetches : OutputStructure, V, E: Session.DefaultTargets : OpStructure](
      feeds: FeedMap = FeedMap.empty,
      fetches: F = Seq.empty[Output[Any]],
      targets: E = Set.empty[UntypedOp],
      options: Option[RunOptions] = None,
      wantMetadata: Boolean = false
  )(implicit
      evOutputToTensor: OutputToTensor.Aux[F, V]
  ): (V, Option[RunMetadata]) = {
    if (nativeHandle == 0)
      throw new IllegalStateException("This session has already been closed.")
    // TODO: !!! [JNI] Add a call to 'extend' once some JNI issues are resolved.
    val (inputs, inputTensors) = feeds.values.toSeq.unzip
    val inputTensorHandles: Array[Long] = inputTensors.map(_.resolve()).toArray
    val inputOpHandles: Array[Long] = inputs.map(_.op.nativeHandle).toArray
    val inputOpIndices: Array[Int] = inputs.map(_.index).toArray
    val (uniqueFetches, resultsBuilder) = Session.processFetches(fetches)
    val outputOpHandles: Array[Long] = uniqueFetches.map(_.op.nativeHandle).toArray
    val outputOpIndices: Array[Int] = uniqueFetches.map(_.index).toArray
    val outputTensorHandles: Array[Long] = Array.ofDim[Long](uniqueFetches.length)
    val targetOpHandles: Array[Long] = Option(targets).map(OpStructure[E].ops)
        .getOrElse(Set.empty[UntypedOp]).map(_.nativeHandle).toArray
    NativeHandleLock.synchronized {
      if (nativeHandle == 0)
        throw new IllegalStateException("close() has been called on the session.")
      nativeHandleWrapper.referenceCount += 1
    }
    try {
      val metadata: Array[Byte] = NativeSession.run(
        handle = nativeHandle,
        runOptions = options.map(_.toByteArray).getOrElse(Array.empty[Byte]),
        inputTensorHandles = inputTensorHandles,
        inputOpHandles = inputOpHandles,
        inputOpIndices = inputOpIndices,
        outputOpHandles = outputOpHandles,
        outputOpIndices = outputOpIndices,
        targetOpHandles = targetOpHandles,
        wantRunMetadata = wantMetadata,
        outputTensorHandles = outputTensorHandles)
      val outputs: V = resultsBuilder(ArraySeq.unsafeWrapArray(outputTensorHandles.map(handle => {
        val tensor = Tensor.fromHostNativeHandle[Any](handle)
        NativeTensor.delete(handle)
        tensor
      })))
      inputTensorHandles.foreach(NativeTensor.delete)
      (outputs, Option(metadata).map(RunMetadata.parseFrom))
    } catch {
      case t: Throwable =>
        NativeHandleLock.synchronized {
          if (nativeHandle != 0) {
            nativeHandleWrapper.referenceCount -= 1
            if (nativeHandleWrapper.referenceCount == 0)
              NativeHandleLock.notifyAll()
          }
        }
        throw t
    } finally {
      NativeHandleLock.synchronized {
        if (nativeHandle != 0) {
          nativeHandleWrapper.referenceCount -= 1
          if (nativeHandleWrapper.referenceCount == 0)
            NativeHandleLock.notifyAll()
        }
      }
    }
  }

  /** Returns a boolean flag indicating whether this session has been closed. */
  def closed: Boolean = {
    nativeHandle == 0
  }
}

/** Contains helper functions for managing sessions. */
object Session {
  type DefaultFetches[F] = DefaultsTo[F, Seq[Output[Any]]]
  type DefaultTargets[T] = DefaultsTo[T, Set[UntypedOp]]

  def apply(
      graph: Graph = Op.currentGraph,
      target: String = null,
      sessionConfig: Option[SessionConfig] = None
  ): Session = {
    // Enable XLA support, if needed.
    sessionConfig.foreach { config =>
      config.optGlobalJITLevel match {
        case Some(L1GraphOptimizerGlobalJIT) | Some(L2GraphOptimizerGlobalJIT) => TensorFlow.enableXLA()
        case _ => ()
      }
    }
    val graphReference = graph.reference
    val nativeHandle = NativeSession.allocate(
      graphReference.nativeHandle,
      target,
      sessionConfig.map(_.configProto.toByteArray).orNull)
    val nativeHandleWrapper = NativeHandleWrapper(nativeHandle)
    val closeFn = () => {
      var done = false
      graphReference.close()
      nativeHandleWrapper.Lock.synchronized {
        if (nativeHandleWrapper.handle != 0) {
          while (!done && nativeHandleWrapper.referenceCount > 0) {
            try {
              nativeHandleWrapper.Lock.wait()
            } catch {
              case _: InterruptedException =>
                Thread.currentThread().interrupt()
                // TODO: [CLIENT] Possible leak of the session and graph in this case?
                done = true
            }
          }
          if (!done) {
            NativeSession.delete(nativeHandleWrapper.handle)
            nativeHandleWrapper.handle = 0
          }
        }
      }
    }
    graph.nativeHandleWrapper.addPreCleanupFunction(closeFn)
    val session = new Session(graphReference, target, nativeHandleWrapper, closeFn)
    // Keep track of references in the Scala side and notify the native library when the session is not referenced
    // anymore anywhere in the Scala side. This will let the native library free the allocated resources and prevent a
    // potential memory leak.
    Disposer.add(session, closeFn)
    session
  }

  private[core] def processFetches[T, V](
      fetchable: T
  )(implicit
      evOutputStructure: OutputStructure[T],
      evOutputToTensor: OutputToTensor.Aux[T, V]
  ): (Seq[Output[Any]], Seq[Tensor[Any]] => V) = {
    val fetches = evOutputStructure.outputs(fetchable)
    val (uniqueFetches, indices) = uniquifyFetches(fetches)
    val resultsBuilder = (values: Seq[Tensor[Any]]) => {
      evOutputToTensor.decodeTensor(fetchable, indices.map(values(_)))._1
    }
    (uniqueFetches, resultsBuilder)
  }

  private[Session] def uniquifyFetches(
      fetches: Seq[Output[Any]]
  ): (Seq[Output[Any]], Seq[Int]) = {
    val uniqueFetches = mutable.ArrayBuffer.empty[Output[Any]]
    val seenFetches = mutable.Map.empty[Output[_], Int]
    val indices = fetches.map(f => seenFetches.getOrElseUpdate(f, {
      uniqueFetches += f
      uniqueFetches.length - 1
    }))
    (uniqueFetches.toSeq, indices)
  }
}
