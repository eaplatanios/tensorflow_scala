package org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
sealed case class DataType[T](cValue: Int, byteSize: Int)

object DataType {
  val float: DataType[Float] = DataType[Float](cValue = 1, byteSize = 4)
  val double: DataType[Double] = DataType[Double](cValue = 2, byteSize = 8)
  val int32: DataType[Int] = DataType[Int](cValue = 3, byteSize = 4)
  val uint8: DataType[Byte] = DataType[Byte](cValue = 4, byteSize = 1)
  val string: DataType[String] = DataType[String](cValue = 7, byteSize = -1)
  val int64: DataType[Long] = DataType[Long](cValue = 9, byteSize = 8)
  val boolean: DataType[Boolean] = DataType[Boolean](cValue = 10, byteSize = 1)

  def fromCValue(cValue: Int): DataType[_] = {
    cValue match {
      case float.cValue => float
      case double.cValue => double
      case int32.cValue => int32
      case uint8.cValue => uint8
      case string.cValue => string
      case int64.cValue => int64
      case boolean.cValue => boolean
      case value => throw new IllegalArgumentException(
        s"Data type $value is not recognized in Scala (TensorFlow version ${TensorFlow.version}).")
    }
  }
}
