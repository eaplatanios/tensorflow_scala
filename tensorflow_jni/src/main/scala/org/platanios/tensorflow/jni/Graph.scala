package org.platanios.tensorflow.jni

import ch.jodersky.jni.nativeLoader

/**
  * @author Emmanouil Antonios Platanios
  */
@nativeLoader("tensorflow_jni")
object Graph {
  @native def allocate(): Long
  @native def delete(handle: Long): Unit
  @native def operation(handle: Long, name: String): Long
  @throws[IllegalArgumentException]
  @native def importGraphDef(handle: Long, graphDef: Array[Byte], prefix: String): Unit
  @native def toGraphDef(handle: Long): Array[Byte]
}
