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

package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
object Session {
  TensorFlow.load()

  @native def allocate(graphHandle: Long, target: String, configProto: Array[Byte]): Long
  @native def delete(handle: Long): Unit
  // TODO: [SESSION] "listDevices".

  /** Executes a computation in a session.
    *
    * The author apologizes for the ugliness of the long argument list of this method. However,
    * take solace in the fact that this is a private method meant to cross the JNI boundary.
    *
    * @param handle              to the C API TF_Session object (Session.nativeHandle)
    * @param runOptions          serialized representation of a RunOptions protocol buffer, or null
    * @param inputOpHandles      (see inputOpIndices)
    * @param inputOpIndices      (see inputTensorHandles)
    * @param inputTensorHandles  together with inputOpHandles and inputOpIndices specifies the values
    *                            that are being "fed" (do not need to be computed) during graph execution.
    *                            inputTensorHandles[i] (which correponds to a Tensor.nativeHandle) is considered to be the
    *                            inputOpIndices[i]-th output of the Operation inputOpHandles[i]. Thus, it is required that
    *     inputOpHandles.length == inputOpIndices.length == inputTensorHandles.length.
    * @param outputOpHandles     (see outputOpIndices)
    * @param outputOpIndices     together with outputOpHandles identifies the set of values that should
    *                            be computed. The outputOpIndices[i]-th output of the Operation outputOpHandles[i], It is
    *                            required that outputOpHandles.length == outputOpIndices.length.
    * @param targetOpHandles     is the set of Operations in the graph that are to be executed but whose
    *                            output will not be returned
    * @param wantRunMetadata     indicates whether metadata about this execution should be returned.
    * @param outputTensorHandles will be filled in with handles to the outputs requested. It is
    *                            required that outputTensorHandles.length == outputOpHandles.length.
    * @return if wantRunMetadata is true, serialized representation of the RunMetadata protocol
    *         buffer, false otherwise.
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
}
