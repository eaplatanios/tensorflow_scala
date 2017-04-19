package org.platanios.tensorflow.jni

import java.nio.ByteBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
object Tensor {
  TensorFlow.load()

  @native def fromBuffer(buffer: ByteBuffer, dataType: Int, shape: Array[Long], byteSize: Long): Long
  @native def dataType(handle: Long): Int
  @native def shape(handle: Long): Array[Long]
  @native def buffer(handle: Long): ByteBuffer
  @native def delete(handle: Long): Unit

  //  @native def allocate(dataType: Int, shape: Array[Long], byteSize: Long): Long
  //  @native def allocateScalarBytes(value: Array[Byte]): Long
  //  @native def setValue(handle: Long, value: Any): Unit
  //  @native def scalarFloat(handle: Long): Float
  //  @native def scalarDouble(handle: Long): Double
  //  @native def scalarInt(handle: Long): Int
  //  @native def scalarLong(handle: Long): Long
  //  @native def scalarBoolean(handle: Long): Boolean
  //  @native def scalarBytes(handle: Long): Array[Byte]
  //  @native def readNDArray(handle: Long, value: Any): Unit
}
