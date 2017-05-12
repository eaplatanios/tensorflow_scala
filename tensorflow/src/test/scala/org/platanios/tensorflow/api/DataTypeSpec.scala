package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.tf.DataType

import java.nio.ByteBuffer

import org.scalatest._

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
class DataTypeSpec extends FlatSpec with Matchers {
  "'DataType.dataTypeOf'" must "work correctly when valid values are provided" in {
    assert(DataType.dataTypeOf(1.0f) === tf.FLOAT32)
    assert(DataType.dataTypeOf(1.0) === tf.FLOAT64)
    assert(DataType.dataTypeOf(1.asInstanceOf[Byte]) === tf.INT8)
    assert(DataType.dataTypeOf(1.asInstanceOf[Short]) === tf.INT16)
    assert(DataType.dataTypeOf(1) === tf.INT32)
    assert(DataType.dataTypeOf(1L) === tf.INT64)
    assert(DataType.dataTypeOf(UByte(1)) === tf.UINT8)
    assert(DataType.dataTypeOf(UShort(1)) === tf.UINT16)
    // assert(DataType.dataTypeOf('a') === tf.UINT16)
    assert(DataType.dataTypeOf(true) === tf.BOOLEAN)
    // assert(DataType.dataTypeOf("foo") === tf.STRING)
    // assert(DataType.dataTypeOf(Tensor(1, 2)) === tf.INT32)
  }

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(Array(1))")
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(1.asInstanceOf[Any])")
  }

  "'DataType.fromCValue'" must "work correctly when valid C values are provided" in {
    assert(DataType.fromCValue(1) === tf.FLOAT32)
    assert(DataType.fromCValue(2) === tf.FLOAT64)
    assert(DataType.fromCValue(3) === tf.INT32)
    assert(DataType.fromCValue(4) === tf.UINT8)
    assert(DataType.fromCValue(5) === tf.INT16)
    assert(DataType.fromCValue(6) === tf.INT8)
    assert(DataType.fromCValue(7) === tf.STRING)
    // assert(DataType.fromCValue(8) === tf.COMPLEX64)
    assert(DataType.fromCValue(9) === tf.INT64)
    assert(DataType.fromCValue(10) === tf.BOOLEAN)
    assert(DataType.fromCValue(11) === tf.QINT8)
    assert(DataType.fromCValue(12) === tf.QUINT8)
    assert(DataType.fromCValue(13) === tf.QINT32)
    // assert(DataType.fromCValue(14) === tf.BFLOAT16)
    assert(DataType.fromCValue(15) === tf.QINT16)
    assert(DataType.fromCValue(16) === tf.QUINT16)
    assert(DataType.fromCValue(17) === tf.UINT16)
    // assert(DataType.fromCValue(18) === tf.COMPLEX128)
    // assert(DataType.fromCValue(19) === tf.FLOAT16)
    assert(DataType.fromCValue(20) === tf.RESOURCE)
  }

  it must "throw an 'IllegalArgumentException' when invalid C values are provided" in {
    assertThrows[IllegalArgumentException](DataType.fromCValue(-10))
    assertThrows[IllegalArgumentException](DataType.fromCValue(-1))
    assertThrows[IllegalArgumentException](DataType.fromCValue(0))
    assertThrows[IllegalArgumentException](DataType.fromCValue(21))
    assertThrows[IllegalArgumentException](DataType.fromCValue(54))
    assertThrows[IllegalArgumentException](DataType.fromCValue(167))
  }

  "'DataType.fromName'" must "work correctly when valid data type names are provided" in {
    assert(DataType.fromName("FLOAT32") === tf.FLOAT32)
    assert(DataType.fromName("FLOAT64") === tf.FLOAT64)
    assert(DataType.fromName("INT32") === tf.INT32)
    assert(DataType.fromName("UINT8") === tf.UINT8)
    assert(DataType.fromName("INT16") === tf.INT16)
    assert(DataType.fromName("INT8") === tf.INT8)
    assert(DataType.fromName("STRING") === tf.STRING)
    // assert(DataType.fromName("Complex64") === tf.COMPLEX64)
    assert(DataType.fromName("INT64") === tf.INT64)
    assert(DataType.fromName("BOOLEAN") === tf.BOOLEAN)
    assert(DataType.fromName("QINT8") === tf.QINT8)
    assert(DataType.fromName("QUINT8") === tf.QUINT8)
    assert(DataType.fromName("QINT32") === tf.QINT32)
    // assert(DataType.fromName("BFLOAT16") === tf.BFLOAT16)
    assert(DataType.fromName("QINT16") === tf.QINT16)
    assert(DataType.fromName("QUINT16") === tf.QUINT16)
    assert(DataType.fromName("UINT16") === tf.UINT16)
    // assert(DataType.fromName("COMPLEX128") === tf.COMPLEX128))
    // assert(DataType.fromName("FLOAT16") === tf.FLOAT16)
    assert(DataType.fromName("RESOURCE") === tf.RESOURCE)
  }

  it must "throw an 'IllegalArgumentException' when invalid data type names are provided" in {
    assertThrows[IllegalArgumentException](DataType.fromName("foo"))
    assertThrows[IllegalArgumentException](DataType.fromName("bar"))
    assertThrows[IllegalArgumentException](DataType.fromName(""))
    assertThrows[IllegalArgumentException](DataType.fromName(null))
  }

  "'DataType.size'" must "give the correct result" in {
    assert(tf.STRING.byteSize === -1)
    assert(tf.BOOLEAN.byteSize === 1)
    // assert(tf.FLOAT16.byteSize === 2)
    assert(tf.FLOAT32.byteSize === 4)
    assert(tf.FLOAT64.byteSize === 8)
    // assert(tf.BFLOAT16.byteSize === 2)
    // assert(tf.COMPLEX64.byteSize === 8)
    // assert(tf.COMPLEX128.byteSize === 16)
    assert(tf.INT8.byteSize === 1)
    assert(tf.INT16.byteSize === 2)
    assert(tf.INT32.byteSize === 4)
    assert(tf.INT64.byteSize === 8)
    assert(tf.UINT8.byteSize === 1)
    assert(tf.UINT16.byteSize === 2)
    assert(tf.QINT8.byteSize === 1)
    assert(tf.QINT16.byteSize === 2)
    assert(tf.QINT32.byteSize === 4)
    assert(tf.QUINT8.byteSize === 1)
    assert(tf.QUINT16.byteSize === 2)
    assert(tf.RESOURCE.byteSize === -1)
  }

  // TODO: Add checks for data type priorities.

  "'DataType.isBoolean'" must "always work correctly" in {
    assert(tf.BOOLEAN.isBoolean === true)
    assert(tf.STRING.isBoolean === false)
    // assert(tf.FLOAT16.isBoolean === false)
    assert(tf.FLOAT32.isBoolean === false)
    assert(tf.FLOAT64.isBoolean === false)
    // assert(tf.BFLOAT16.isBoolean === false)
    // assert(tf.Complex64.isBoolean === false)
    // assert(tf.COMPLEX128.isBoolean === false)
    assert(tf.INT8.isBoolean === false)
    assert(tf.INT16.isBoolean === false)
    assert(tf.INT32.isBoolean === false)
    assert(tf.INT64.isBoolean === false)
    assert(tf.UINT8.isBoolean === false)
    assert(tf.UINT16.isBoolean === false)
    assert(tf.QINT8.isBoolean === false)
    assert(tf.QINT16.isBoolean === false)
    assert(tf.QINT32.isBoolean === false)
    assert(tf.QUINT8.isBoolean === false)
    assert(tf.QUINT16.isBoolean === false)
    assert(tf.RESOURCE.isBoolean === false)
  }

  "'DataType.isFloatingPoint'" must "always work correctly" in {
    assert(tf.BOOLEAN.isFloatingPoint === false)
    assert(tf.STRING.isFloatingPoint === false)
    // assert(tf.FLOAT16.isFloatingPoint === true)
    assert(tf.FLOAT32.isFloatingPoint === true)
    assert(tf.FLOAT64.isFloatingPoint === true)
    // assert(tf.BFLOAT16.isFloatingPoint === false)
    // assert(tf.Complex64.isFloatingPoint === false)
    // assert(tf.COMPLEX128.isFloatingPoint === false)
    assert(tf.INT8.isFloatingPoint === false)
    assert(tf.INT16.isFloatingPoint === false)
    assert(tf.INT32.isFloatingPoint === false)
    assert(tf.INT64.isFloatingPoint === false)
    assert(tf.UINT8.isFloatingPoint === false)
    assert(tf.UINT16.isFloatingPoint === false)
    assert(tf.QINT8.isFloatingPoint === false)
    assert(tf.QINT16.isFloatingPoint === false)
    assert(tf.QINT32.isFloatingPoint === false)
    assert(tf.QUINT8.isFloatingPoint === false)
    assert(tf.QUINT16.isFloatingPoint === false)
    assert(tf.RESOURCE.isFloatingPoint === false)
  }

  "'DataType.isInteger'" must "always work correctly" in {
    assert(tf.BOOLEAN.isInteger === false)
    assert(tf.STRING.isInteger === false)
    // assert(tf.FLOAT16.isInteger === false)
    assert(tf.FLOAT32.isInteger === false)
    assert(tf.FLOAT64.isInteger === false)
    // assert(tf.BFLOAT16.isInteger === false)
    // assert(tf.Complex64.isInteger === false)
    // assert(tf.COMPLEX128.isInteger === false)
    assert(tf.INT8.isInteger === true)
    assert(tf.INT16.isInteger === true)
    assert(tf.INT32.isInteger === true)
    assert(tf.INT64.isInteger === true)
    assert(tf.UINT8.isInteger === true)
    assert(tf.UINT16.isInteger === true)
    assert(tf.QINT8.isInteger === false)
    assert(tf.QINT16.isInteger === false)
    assert(tf.QINT32.isInteger === false)
    assert(tf.QUINT8.isInteger === false)
    assert(tf.QUINT16.isInteger === false)
    assert(tf.RESOURCE.isInteger === false)
  }

  "'DataType.isComplex'" must "always work correctly" in {
    assert(tf.BOOLEAN.isComplex === false)
    assert(tf.STRING.isComplex === false)
    // assert(tf.FLOAT16.isComplex === false)
    assert(tf.FLOAT32.isComplex === false)
    assert(tf.FLOAT64.isComplex === false)
    // assert(tf.BFLOAT16.isComplex === false)
    // assert(tf.Complex64.isComplex === true)
    // assert(tf.COMPLEX128.isComplex === true)
    assert(tf.INT8.isComplex === false)
    assert(tf.INT16.isComplex === false)
    assert(tf.INT32.isComplex === false)
    assert(tf.INT64.isComplex === false)
    assert(tf.UINT8.isComplex === false)
    assert(tf.UINT16.isComplex === false)
    assert(tf.QINT8.isComplex === false)
    assert(tf.QINT16.isComplex === false)
    assert(tf.QINT32.isComplex === false)
    assert(tf.QUINT8.isComplex === false)
    assert(tf.QUINT16.isComplex === false)
    assert(tf.RESOURCE.isComplex === false)
  }

  "'DataType.isQuantized'" must "always work correctly" in {
    assert(tf.BOOLEAN.isQuantized === false)
    assert(tf.STRING.isQuantized === false)
    // assert(tf.FLOAT16.isQuantized === false)
    assert(tf.FLOAT32.isQuantized === false)
    assert(tf.FLOAT64.isQuantized === false)
    // assert(tf.BFLOAT16.isQuantized === true)
    // assert(tf.Complex64.isQuantized === false)
    // assert(tf.COMPLEX128.isQuantized === false)
    assert(tf.INT8.isQuantized === false)
    assert(tf.INT16.isQuantized === false)
    assert(tf.INT32.isQuantized === false)
    assert(tf.INT64.isQuantized === false)
    assert(tf.UINT8.isQuantized === false)
    assert(tf.UINT16.isQuantized === false)
    assert(tf.QINT8.isQuantized === true)
    assert(tf.QINT16.isQuantized === true)
    assert(tf.QINT32.isQuantized === true)
    assert(tf.QUINT8.isQuantized === true)
    assert(tf.QUINT16.isQuantized === true)
    assert(tf.RESOURCE.isQuantized === false)
  }

  "'DataType.isUnsigned'" must "always work correctly" in {
    assert(tf.BOOLEAN.isUnsigned === false)
    assert(tf.STRING.isUnsigned === false)
    // assert(tf.FLOAT16.isUnsigned === false)
    assert(tf.FLOAT32.isUnsigned === false)
    assert(tf.FLOAT64.isUnsigned === false)
    // assert(tf.BFLOAT16.isUnsigned === false)
    // assert(tf.Complex64.isUnsigned === false)
    // assert(tf.COMPLEX128.isUnsigned === false)
    assert(tf.INT8.isUnsigned === false)
    assert(tf.INT16.isUnsigned === false)
    assert(tf.INT32.isUnsigned === false)
    assert(tf.INT64.isUnsigned === false)
    assert(tf.UINT8.isUnsigned === true)
    assert(tf.UINT16.isUnsigned === true)
    assert(tf.QINT8.isUnsigned === false)
    assert(tf.QINT16.isUnsigned === false)
    assert(tf.QINT32.isUnsigned === false)
    assert(tf.QUINT8.isUnsigned === false)
    assert(tf.QUINT16.isUnsigned === false)
    assert(tf.RESOURCE.isUnsigned === false)
  }

  // "'DataType.real'" must "always work correctly" in {
  //   assert(tf.Bool.real === tf.BOOLEAN)
  //   assert(tf.Str.real === tf.Strinh)
  //   assert(tf.FLOAT16.real === tf.FLOAT16)
  //   assert(tf.FLOAT32.real === tf.FLOAT32)
  //   assert(tf.FLOAT64.real === tf.FLOAT64)
  //   assert(tf.BFLOAT16.real === tf.BFLOAT16)
  //   assert(tf.Complex64.real === tf.FLOAT32)
  //   assert(tf.COMPLEX128.real === tf.FLOAT64)
  //   assert(tf.INT8.real === tf.INT8)
  //   assert(tf.INT16.real === tf.INT16)
  //   assert(tf.INT32.real === tf.INT32)
  //   assert(tf.INT64.real === tf.INT64)
  //   assert(tf.UINT8.real === tf.UINT8)
  //   assert(tf.UINT16.real === tf.UINT16)
  //   assert(tf.QINT8.real === tf.QINT8)
  //   assert(tf.QINT16.real === tf.QINT16)
  //   assert(tf.QINT32.real === tf.QINT32)
  //   assert(tf.QUINT8.real === tf.QUINT8)
  //   assert(tf.QUINT16.real === tf.QUINT16)
  //   assert(tf.RESOURCE.real === tf.RESOURCE)
  // }

  "'DataType.equals'" must "always work correctly" in {
    assert(tf.FLOAT32 === tf.FLOAT32)
    assert(tf.FLOAT32 !== tf.FLOAT64)
    assert(tf.INT8 !== tf.UINT8)
    assert(tf.INT8 !== tf.QINT8)
  }

  "'DataType.cast'" must "work correctly when provided values of supported types" in {
    assert(tf.BOOLEAN.cast(false) === false)
    assert(tf.BOOLEAN.cast(true) === true)
    assert(tf.STRING.cast("foo") === "foo")
    // assert(tf.STRING.cast(false) === "false")
    // assert(tf.STRING.cast(1.0) === "1.0")
    // assert(tf.STRING.cast(1f) === "1.0")
    // assert(tf.STRING.cast(-2L) === "-2")
    // assert(tf.FLOAT16.cast(-2.0) === -2f)
    // assert(tf.FLOAT16.cast(-2L) === -2f)
    assert(tf.FLOAT32.cast(-2.0) === -2f)
    assert(tf.FLOAT32.cast(2) === 2f)
    assert(tf.FLOAT64.cast(2) === 2.0)
    assert(tf.FLOAT64.cast(-2f) === -2.0)
    assert(tf.FLOAT64.cast(-2L) === -2.0)
    /// assert(tf.BFLOAT16.cast(-2) === -2f)
    // TODO: Add complex data type checks.
    assert(tf.INT8.cast(-2L) === -2.toByte)
    assert(tf.INT8.cast(-2.0) === -2.toByte)
    assert(tf.INT8.cast(UByte(2)) === 2.toByte)
    assert(tf.INT16.cast(-2L) === -2.toShort)
    assert(tf.INT32.cast(-2L) === -2)
    assert(tf.INT64.cast(-2.0) === -2L)
    assert(tf.INT64.cast(UByte(2)) === 2L)
    assert(tf.UINT8.cast(UByte(2)) === UByte(2))
    assert(tf.UINT8.cast(UShort(2)) === UByte(2))
    assert(tf.UINT8.cast(2L) === UByte(2))
    assert(tf.UINT8.cast(-2.0) === UByte(254)) // TODO: Should this throw an error?
    assert(tf.UINT16.cast(-UByte(2)) === UShort(254)) // TODO: Should this throw an error?
    assert(tf.UINT16.cast(UShort(2)) === UShort(2))
    assert(tf.UINT16.cast(2L) === UShort(2))
    assert(tf.UINT16.cast(2.0) === UShort(2))
    assert(tf.QINT8.cast(-2L) === -2.toByte)
    assert(tf.QINT8.cast(-2.0) === -2.toByte)
    assert(tf.QINT8.cast(UByte(2)) === 2.toByte)
    assert(tf.QINT16.cast(-2L) === -2.toShort)
    assert(tf.QINT32.cast(-2L) === -2)
    assert(tf.QUINT8.cast(UByte(2)) === UByte(2))
    assert(tf.QUINT8.cast(UShort(2)) === UByte(2))
    assert(tf.QUINT8.cast(2L) === UByte(2))
    assert(tf.QUINT8.cast(-2.0) === UByte(254)) // TODO: Should this throw an error?
    assert(tf.QUINT16.cast(-UByte(2)) === UShort(254)) // TODO: Should this throw an error?
    assert(tf.QUINT16.cast(UShort(2)) === UShort(2))
    assert(tf.QUINT16.cast(2L) === UShort(2))
    assert(tf.QUINT16.cast(2.0) === UShort(2))
  }

  // TODO: Add 'InvalidCastException' checks.

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("DataType.FLOAT32.cast(Array(1))")
    assertDoesNotCompile("DataType.FLOAT32.cast(1.asInstanceOf[Any])")
  }

  "'DataType.putElementInBuffer'" must "work correctly when provided values of supported types" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(12)
    tf.INT32.putElementInBuffer(buffer, 0, 1)
    tf.INT32.putElementInBuffer(buffer, 4, 16)
    tf.INT32.putElementInBuffer(buffer, 8, 257)
    assert(buffer.get(0) === 0x00.toByte)
    assert(buffer.get(1) === 0x00.toByte)
    assert(buffer.get(2) === 0x00.toByte)
    assert(buffer.get(3) === 0x01.toByte)
    assert(buffer.get(4) === 0x00.toByte)
    assert(buffer.get(5) === 0x00.toByte)
    assert(buffer.get(6) === 0x00.toByte)
    assert(buffer.get(7) === 0x10.toByte)
    assert(buffer.get(8) === 0x00.toByte)
    assert(buffer.get(9) === 0x00.toByte)
    assert(buffer.get(10) === 0x01.toByte)
    assert(buffer.get(11) === 0x01.toByte)
    // TODO: Add checks for other data types.
  }

  it must "throw an 'IndexOutOfBoundsException' exception when writing beyond the buffer boundaries" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(10)
    tf.INT32.putElementInBuffer(buffer, 0, 1)
    tf.INT32.putElementInBuffer(buffer, 4, 16)
    assertThrows[IndexOutOfBoundsException](tf.INT32.putElementInBuffer(buffer, 8, 257))
  }

  "'DataType.getElementFromBuffer'" must "match the behavior of 'DataType.putElementInBuffer'" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(1024)
    // tf.FLOAT16.putElementInBuffer(buffer, 0, 2f)
    tf.FLOAT32.putElementInBuffer(buffer, 2, -4.23f)
    tf.FLOAT64.putElementInBuffer(buffer, 6, 3.45)
    // tf.BFLOAT16.putElementInBuffer(buffer, 14, -1.23f)
    tf.INT8.putElementInBuffer(buffer, 16, 4.toByte)
    tf.INT16.putElementInBuffer(buffer, 17, (-2).toShort)
    tf.INT32.putElementInBuffer(buffer, 19, 54)
    tf.INT64.putElementInBuffer(buffer, 23, -3416L)
    tf.UINT8.putElementInBuffer(buffer, 31, UByte(34))
    tf.UINT16.putElementInBuffer(buffer, 32, UShort(657))
    tf.QINT8.putElementInBuffer(buffer, 34, (-4).toByte)
    tf.QINT16.putElementInBuffer(buffer, 35, 32.toShort)
    tf.QINT32.putElementInBuffer(buffer, 37, -548979)
    tf.QUINT8.putElementInBuffer(buffer, 41, UByte(254))
    tf.QUINT16.putElementInBuffer(buffer, 42, UShort(765))
    tf.BOOLEAN.putElementInBuffer(buffer, 44, true)
    tf.BOOLEAN.putElementInBuffer(buffer, 45, false)
    // TODO: Add checks for the string data type.
    // assert(tf.FLOAT16.getElementFromBuffer(buffer, 0) === 2f)
    assert(tf.FLOAT32.getElementFromBuffer(buffer, 2) === -4.23f)
    assert(tf.FLOAT64.getElementFromBuffer(buffer, 6) === 3.45)
    // assert(tf.BFLOAT16.getElementFromBuffer(buffer, 14) === -1.23f)
    assert(tf.INT8.getElementFromBuffer(buffer, 16) === 4.toByte)
    assert(tf.INT16.getElementFromBuffer(buffer, 17) === (-2).toShort)
    assert(tf.INT32.getElementFromBuffer(buffer, 19) === 54)
    assert(tf.INT64.getElementFromBuffer(buffer, 23) === -3416L)
    assert(tf.UINT8.getElementFromBuffer(buffer, 31) === UByte(34))
    assert(tf.UINT16.getElementFromBuffer(buffer, 32) === UShort(657))
    assert(tf.QINT8.getElementFromBuffer(buffer, 34) === (-4).toByte)
    assert(tf.QINT16.getElementFromBuffer(buffer, 35) === 32.toShort)
    assert(tf.QINT32.getElementFromBuffer(buffer, 37) === -548979)
    assert(tf.QUINT8.getElementFromBuffer(buffer, 41) === UByte(254))
    assert(tf.QUINT16.getElementFromBuffer(buffer, 42) === UShort(765))
    assert(tf.BOOLEAN.getElementFromBuffer(buffer, 44) === true)
    assert(tf.BOOLEAN.getElementFromBuffer(buffer, 45) === false)
  }
}
