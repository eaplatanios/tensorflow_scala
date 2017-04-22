package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
case class OpOutput(opHandle: Long, outputIndex: Int)

object Op {
  TensorFlow.load()

  // Operation
  @native def name(handle: Long): String
  @native def opType(handle: Long): String
  @native def device(handle: Long): String
  @native def numInputs(handle: Long): Int
  @native def numControlInputs(handle: Long): Int
  @native def numOutputs(handle: Long): Int
  @native def numControlOutputs(handle: Long): Int
  @native def numConsumers(handle: Long, output: Int): Int
  @native def input(handle: Long, inputIndex: Int): OpOutput
  @native def controlInputs(handle: Long): Array[Long]
  @native def controlOutputs(handle: Long): Array[Long]
  @native def consumers(handle: Long, outputIndex: Int): Array[OpOutput]
  @native def inputDataType(graphHandle: Long, opHandle: Long, inputIndex: Int): Int
  @native def outputDataType(graphHandle: Long, opHandle: Long, outputIndex: Int): Int
  @native def shape(graphHandle: Long, opHandle: Long, output: Int): Array[Long]
  @native def setShape(graphHandle: Long, opHandle: Long, output: Int, shape: Array[Long], rank: Int): Unit
  @native def getAttrString(handle: Long, name: String): String
  @native def getAttrStringList(handle: Long, name: String): Array[String]
  @native def getAttrType(handle: Long, name: String): Int
  @native def getAttrShape(handle: Long, name: String): Array[Long]
  @native def allOps: Array[Byte]

  // Operation Builder
  @native def allocate(graphHandle: Long, opType: String, name: String): Long
  @native def finish(handle: Long): Long
  @native def addInput(handle: Long, operationHandle: Long, index: Int): Unit
  @native def addInputList(handle: Long, operationHandles: Array[Long], indices: Array[Int]): Unit
  @native def addControlInput(handle: Long, inputOpHandle: Long): Unit
  @native def setDevice(handle: Long, device: String): Unit
  @native def colocateWith(handle: Long, colocationOpHandle: Long): Unit

  // The names of all the setAttr* family functions below correspond to the C library types, not the
  // Java library types. Roughly, setAttrFoo calls the TensorFlow C library function: TF_SetAttrFoo.
  //
  // TODO:
  // - setAttrShapeList: Which would take in a long[][]
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
}
