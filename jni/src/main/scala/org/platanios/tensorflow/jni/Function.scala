package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
object Function {
  TensorFlow.load()

  @native def graphToFunction(
      fnBodyGraphHandle: Long, fnName: String, opHandles: Array[Long],
      inputOpHandles: Array[Long], inputOpIndices: Array[Int],
      outputOpHandles: Array[Long], outputOpIndices: Array[Int],
      outputNames: Array[String]): Long
  @native def addToGraph(graphHandle: Long, functionHandle: Long): Unit
  @native def toFunctionDef(handle: Long): Array[Byte]
  @native def delete(handle: Long): Unit
}
