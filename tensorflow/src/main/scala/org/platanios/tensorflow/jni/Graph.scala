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
  @native def addGradients(handle: Long, y: Array[OpOutput], x: Array[OpOutput], dx: Array[OpOutput]): Array[OpOutput]
  @throws[IllegalArgumentException]
  @native def importGraphDef(
      handle: Long, graphDef: Array[Byte], prefix: String, inputsMapSourceOpNames: Array[String],
      inputsMapSourceOpOutputIndices: Array[Int], inputsMapDestinationOpHandles: Array[Long],
      inputsMapDestinationOpOutputIndices: Array[Int], controlDependenciesMapSourceOpNames: Array[String],
      controlDependenciesMapDestinationOpHandles: Array[Long], controlDependenciesOpHandles: Array[Long]): Unit
  @native def toGraphDef(handle: Long): Array[Byte]
}
