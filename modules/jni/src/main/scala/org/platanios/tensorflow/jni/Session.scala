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

package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
object Session {
  TensorFlow.load()

  @native def allocate(graphHandle: Long, target: String, configProto: Array[Byte]): Long
  @native def delete(handle: Long): Unit

  // TODO: [SESSION] "listDevices".
  // TODO: [SESSION] Add TPU support using the experimental C API.

  /** Executes a computation in a session.
    *
    * @param handle              Handle to the native TensorFlow session object.
    * @param runOptions          Serialized representation of a `RunOptions` protocol buffer, or `null`.
    * @param inputOpHandles      See `inputOpIndices`.
    * @param inputOpIndices      See `inputTensorHandles`.
    * @param inputTensorHandles  Together with `inputOpHandles` and `inputOpIndices` this array specifies the values
    *                            that are being "fed" (do not need to be computed) during graph execution.
    *                            `inputTensorHandles(i)` (which corresponds to a `Tensor.nativeHandle`) is considered to
    *                            be the `inputOpIndices(i)`-th output of the operation `inputOpHandles(i)`. Thus, it is
    *                            required that
    *                            `inputOpHandles.length == inputOpIndices.length == inputTensorHandles.length`.
    * @param outputOpHandles     (see outputOpIndices)
    * @param outputOpIndices     together with outputOpHandles identifies the set of values that should
    *                            be computed. The `outputOpIndices(i)`-th output of the operation `outputOpHandles(i)`.
    *                            It is required that `outputOpHandles.length == outputOpIndices.length`.
    * @param targetOpHandles     Set of operations in the graph that are to be executed but whose output will not be
    *                            returned.
    * @param wantRunMetadata     Boolean variable that indicates whether metadata about this execution should be
    *                            returned.
    * @param outputTensorHandles Array that will be filled in with handles to the outputs requested. It is required that
    *                            `outputTensorHandles.length == outputOpHandles.length`.
    * @return Serialized representation of the `RunMetadata` protocol buffer, or `null` if `wantRunMetadata` is `false`.
    */
  @native def run(
      handle: Long,
      runOptions: Array[Byte],
      inputTensorHandles: Array[Long],
      inputOpHandles: Array[Long],
      inputOpIndices: Array[Int],
      outputOpHandles: Array[Long],
      outputOpIndices: Array[Int],
      targetOpHandles: Array[Long],
      wantRunMetadata: Boolean,
      outputTensorHandles: Array[Long]): Array[Byte]

  @native def extend(handle: Long): Unit

  @native def deviceList(configProto: Array[Byte]): Array[Array[Byte]]
}
