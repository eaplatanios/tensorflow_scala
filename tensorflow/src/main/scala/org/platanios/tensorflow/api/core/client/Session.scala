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

import org.platanios.tensorflow.api.Closeable
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.{Op, OpCreationContext}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.jni.{Session => NativeSession}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
final case class Session private (
    graph: Graph, private val graphReference: Graph#Reference, private var nativeHandle: Long) extends Closeable {
  private[this] object NativeHandleLock
  private[this] var referenceCount: Int = 0

  def run[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Op.Output],
      targets: E = Traversable.empty[Op], runOptions: Option[Array[Byte]] = None)
      (implicit executable: Executable[E], fetchable: Fetchable.Aux[F, R]): R = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, runOptions = runOptions)._1
  }

  def runWithMetadata[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Op.Output],
      targets: E = Traversable.empty[Op], runOptions: Option[Array[Byte]] = None)
      (implicit executable: Executable[E], fetchable: Fetchable.Aux[F, R]): (R, Array[Byte]) = {
    runHelper(feeds = feeds, fetches = fetches, targets = targets, runOptions = runOptions, wantMetadata = true)
  }

  private def runHelper[F, E, R](
      feeds: FeedMap = FeedMap.empty, fetches: F = Seq.empty[Op.Output],
      targets: E = Traversable.empty[Op], runOptions: Option[Array[Byte]] = None, wantMetadata: Boolean = false)
      (implicit executable: Executable[E], fetchable: Fetchable.Aux[F, R]): (R, Array[Byte]) = {
    val (inputs, inputTensors) = feeds.values.toSeq.unzip
    val inputTensorNativeViews = inputTensors.map(_.nativeView)
    val inputTensorHandles: Array[Long] = inputTensorNativeViews.map(_.nativeHandle).toArray
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
    // It's okay to use Operation.getUnsafeNativeHandle() here since the safety depends on the
    // validity of the Graph and the graph reference ensures that.
    val metadata: Array[Byte] = NativeSession.run(
      handle = nativeHandle,
      runOptions = runOptions.getOrElse(Array.empty[Byte]),
      inputTensorHandles = inputTensorHandles,
      inputOpHandles = inputOpHandles,
      inputOpIndices = inputOpIndices,
      outputOpHandles = outputOpHandles,
      outputOpIndices = outputOpIndices,
      targetOpHandles = targetOpHandles,
      wantRunMetadata = wantMetadata,
      outputTensorHandles = outputTensorHandles)
    val outputs: R = resultsBuilder(outputTensorHandles.map(Tensor.fromTFNativeHandle))
    NativeHandleLock.synchronized {
      if (nativeHandle != 0) {
        referenceCount -= 1
        if (referenceCount == 0)
          NativeHandleLock.notifyAll()
      }
    }
    inputTensorNativeViews.foreach(_.close())
    (outputs, metadata)
  }

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
              // Possible leak of the session and graph in this case?
              return
          }
        }
        NativeSession.delete(nativeHandle)
        nativeHandle = 0
      }
    }
  }
}

object Session {
  def apply(graph: Graph): Session = {
    val graphReference = graph.reference
    Session(graph, graphReference, NativeSession.allocate(graphReference.nativeHandle))
  }

  def apply()(implicit context: DynamicVariable[OpCreationContext]): Session = {
    val graph = context.graph
    val graphReference = graph.reference
    Session(graph, graphReference, NativeSession.allocate(graphReference.nativeHandle))
  }
}
