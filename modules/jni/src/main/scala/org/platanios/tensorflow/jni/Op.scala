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
case class Output(opHandle: Long, outputIndex: Int)

object Op {
  TensorFlow.load()

  //region Operation

  @native def name(handle: Long): String
  @native def opType(handle: Long): String
  @native def device(handle: Long): String
  @native def numInputs(handle: Long): Int
  @native def numControlInputs(handle: Long): Int
  @native def numOutputs(handle: Long): Int
  @native def numControlOutputs(handle: Long): Int
  @native def numConsumers(handle: Long, output: Int): Int
  @native def input(handle: Long, inputIndex: Int): Output
  @native def inputs(handle: Long): Array[Output]
  @native def controlInputs(handle: Long): Array[Long]
  @native def controlOutputs(handle: Long): Array[Long]
  @native def consumers(handle: Long, outputIndex: Int): Array[Output]
  @native def inputDataType(graphHandle: Long, opHandle: Long, inputIndex: Int): Int
  @native def outputDataType(graphHandle: Long, opHandle: Long, outputIndex: Int): Int
  @native def shape(graphHandle: Long, opHandle: Long, output: Int): Array[Long]
  @native def setShape(graphHandle: Long, opHandle: Long, output: Int, shape: Array[Long], rank: Int): Unit
  @native def getAttrString(handle: Long, name: String): String
  @native def getAttrStringList(handle: Long, name: String): Array[String]
  @native def getAttrInt(handle: Long, name: String): Long
  @native def getAttrIntList(handle: Long, name: String): Array[Long]
  @native def getAttrFloat(handle: Long, name: String): Float
  @native def getAttrFloatList(handle: Long, name: String): Array[Float]
  @native def getAttrBool(handle: Long, name: String): Boolean
  @native def getAttrBoolList(handle: Long, name: String): Array[Boolean]
  @native def getAttrType(handle: Long, name: String): Int
  @native def getAttrTypeList(handle: Long, name: String): Array[Int]
  @native def getAttrTensor(handle: Long, name: String): Long
  @native def getAttrShape(handle: Long, name: String): Array[Long]
  @native def toNodeDef(handle: Long): Array[Byte]
  @native def allOps: Array[Byte]
  @native def allRegisteredKernels: Array[Byte]
  @native def registeredKernelsForOp(opName: String): Array[Byte]
  @native def tryEvaluateConstant(graphHandle: Long, opHandle: Long, outputIndex: Int): Long

  //endregion Operation

  //region Operation Builder

  @native def toOpDef(graphHandle: Long, opType: String): Array[Byte]
  @native def allocate(graphHandle: Long, opType: String, name: String): Long
  @native def finish(handle: Long): Long
  @native def addInput(handle: Long, operationHandle: Long, index: Int): Unit
  @native def addInputList(handle: Long, operationHandles: Array[Long], indices: Array[Int]): Unit
  @native def addControlInput(handle: Long, inputOpHandle: Long): Unit
  @native def setDevice(handle: Long, device: String): Unit
  @native def colocateWith(handle: Long, colocationOpHandle: Long): Unit
  @native def setAttrString(handle: Long, name: String, value: Array[Byte]): Unit
  @native def setAttrStringList(handle: Long, name: String, value: Array[Array[Byte]]): Unit
  @native def setAttrInt(handle: Long, name: String, value: Long): Unit
  @native def setAttrIntList(handle: Long, name: String, value: Array[Long]): Unit
  @native def setAttrFloat(handle: Long, name: String, value: Float): Unit
  @native def setAttrFloatList(handle: Long, name: String, value: Array[Float]): Unit
  @native def setAttrBool(handle: Long, name: String, value: Boolean): Unit
  @native def setAttrBoolList(handle: Long, name: String, value: Array[Boolean]): Unit
  @native def setAttrType(handle: Long, name: String, dataType: Int): Unit
  @native def setAttrTypeList(handle: Long, name: String, dataType: Array[Int]): Unit
  @native def setAttrTensor(handle: Long, name: String, tensorHandle: Long): Unit
  @native def setAttrTensorList(handle: Long, name: String, tensorHandle: Array[Long]): Unit
  @native def setAttrShape(handle: Long, name: String, shape: Array[Long], numDims: Int): Unit
  @native def setAttrShapeList(handle: Long, name: String, shapes: Array[Array[Long]], numDims: Array[Int], numShapes: Int): Unit
  @native def setAttrFuncName(handle: Long, name: String, value: Array[Byte]): Unit
  @native def setAttrProto(handle: Long, name: String, value: Array[Byte]): Unit

  //endregion Operation Builder
}
