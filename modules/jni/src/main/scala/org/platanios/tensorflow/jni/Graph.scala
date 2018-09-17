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
object Graph {
  TensorFlow.load()

  @native def allocate(): Long
  @native def delete(handle: Long): Unit
  @native def findOp(handle: Long, name: String): Long
  @native def ops(handle: Long): Array[Long]

  @native def addGradients(
      handle: Long,
      y: Array[Output],
      x: Array[Output],
      dx: Array[Output]
  ): Array[Output]

  @native def importGraphDef(
      handle: Long,
      graphDef: Array[Byte],
      prefix: String,
      inputsMapSourceOpNames: Array[String],
      inputsMapSourceOutputIndices: Array[Int],
      inputsMapDestinationOpHandles: Array[Long],
      inputsMapDestinationOutputIndices: Array[Int],
      controlDependenciesMapSourceOpNames: Array[String],
      controlDependenciesMapDestinationOpHandles: Array[Long],
      controlDependenciesOpHandles: Array[Long]): Unit

  @native def toGraphDef(handle: Long): Array[Byte]
}
