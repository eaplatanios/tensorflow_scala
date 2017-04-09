package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

/**
  * @author Emmanouil Antonios Platanios
  */
sealed case class DataType private (cValue: Int, byteSize: Int)

object DataType {
  val float: DataType = DataType(cValue = 1, byteSize = 4)
  val double: DataType = DataType(cValue = 2, byteSize = 8)
  val int32: DataType = DataType(cValue = 3, byteSize = 4)
  val uint8: DataType = DataType(cValue = 4, byteSize = 1)
  val string: DataType = DataType(cValue = 7, byteSize = -1)
  val int64: DataType = DataType(cValue = 9, byteSize = 8)
  val boolean: DataType = DataType(cValue = 10, byteSize = 1)

  private[api] def fromCValue(cValue: Int): DataType = {
    cValue match {
      case float.cValue => float
      case double.cValue => double
      case int32.cValue => int32
      case uint8.cValue => uint8
      case string.cValue => string
      case int64.cValue => int64
      case boolean.cValue => boolean
      case value => throw new IllegalArgumentException(
        s"Data type $value is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
    }
  }
}
