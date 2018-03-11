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

package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
object Function {
  TensorFlow.load()

  @native def graphToFunction(
      fnBodyGraphHandle: Long, fnName: String, appendHashToFnName: Boolean, opHandles: Array[Long],
      inputOpHandles: Array[Long], inputOpIndices: Array[Int],
      outputOpHandles: Array[Long], outputOpIndices: Array[Int],
      outputNames: Array[String]): Long
  @native def copyToGraph(graphHandle: Long, functionHandle: Long, gradientHandle: Long): Unit
  @native def toFunctionDef(handle: Long): Array[Byte]
  @native def delete(handle: Long): Unit
}
