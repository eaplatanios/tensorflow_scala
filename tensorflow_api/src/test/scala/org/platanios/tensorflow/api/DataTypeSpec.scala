package org.platanios.tensorflow.api

import java.nio.ByteBuffer

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class DataTypeSpec extends FlatSpec with Matchers {
  "'DataType.dataTypeOf'" must "work correctly when valid values are provided" in {
    assert(DataType.dataTypeOf(1.0f) === DataType.Float32)
    assert(DataType.dataTypeOf(1.0) === DataType.Float64)
    assert(DataType.dataTypeOf(1.asInstanceOf[Byte]) === DataType.Int8)
    assert(DataType.dataTypeOf(1.asInstanceOf[Short]) === DataType.Int16)
    assert(DataType.dataTypeOf(1) === DataType.Int32)
    assert(DataType.dataTypeOf(1L) === DataType.Int64)
    assert(DataType.dataTypeOf(UInt8(1)) === DataType.UInt8)
    assert(DataType.dataTypeOf(UInt16(1)) === DataType.UInt16)
    // assert(DataType.dataTypeOf('a') === DataType.UInt16)
    assert(DataType.dataTypeOf(true) === DataType.Bool)
    // assert(DataType.dataTypeOf("foo") === DataType.String)
    // assert(DataType.dataTypeOf(Tensor(1, 2)) === DataType.Int32)
  }

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(Array(1))")
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(1.asInstanceOf[Any])")
  }

  "'DataType.fromCValue'" must "work correctly when valid C values are provided" in {
    assert(DataType.fromCValue(1) === DataType.Float32)
    assert(DataType.fromCValue(2) === DataType.Float64)
    assert(DataType.fromCValue(3) === DataType.Int32)
    assert(DataType.fromCValue(4) === DataType.UInt8)
    assert(DataType.fromCValue(5) === DataType.Int16)
    assert(DataType.fromCValue(6) === DataType.Int8)
    assert(DataType.fromCValue(7) === DataType.Str)
    // assert(DataType.fromCValue(8) === DataType.Complex64)
    assert(DataType.fromCValue(9) === DataType.Int64)
    assert(DataType.fromCValue(10) === DataType.Bool)
    assert(DataType.fromCValue(11) === DataType.QInt8)
    assert(DataType.fromCValue(12) === DataType.QUInt8)
    assert(DataType.fromCValue(13) === DataType.QInt32)
    // assert(DataType.fromCValue(14) === DataType.BFloat16)
    assert(DataType.fromCValue(15) === DataType.QInt16)
    assert(DataType.fromCValue(16) === DataType.QUInt16)
    assert(DataType.fromCValue(17) === DataType.UInt16)
    // assert(DataType.fromCValue(18) === DataType.Complex128)
    // assert(DataType.fromCValue(19) === DataType.Float16)
    assert(DataType.fromCValue(20) === DataType.Resource)
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
    assert(DataType.fromName("Float32") === DataType.Float32)
    assert(DataType.fromName("Float64") === DataType.Float64)
    assert(DataType.fromName("Int32") === DataType.Int32)
    assert(DataType.fromName("UInt8") === DataType.UInt8)
    assert(DataType.fromName("Int16") === DataType.Int16)
    assert(DataType.fromName("Int8") === DataType.Int8)
    assert(DataType.fromName("String") === DataType.Str)
    // assert(DataType.fromName("Complex64") === DataType.Complex64)
    assert(DataType.fromName("Int64") === DataType.Int64)
    assert(DataType.fromName("Boolean") === DataType.Bool)
    assert(DataType.fromName("QInt8") === DataType.QInt8)
    assert(DataType.fromName("QUInt8") === DataType.QUInt8)
    assert(DataType.fromName("QInt32") === DataType.QInt32)
    // assert(DataType.fromName("BFloat16") === DataType.BFloat16)
    assert(DataType.fromName("QInt16") === DataType.QInt16)
    assert(DataType.fromName("QUInt16") === DataType.QUInt16)
    assert(DataType.fromName("UInt16") === DataType.UInt16)
    // assert(DataType.fromName("Complex128") === DataType.Complex128)
    // assert(DataType.fromName("Float16") === DataType.Float16)
    assert(DataType.fromName("Resource") === DataType.Resource)
  }

  it must "throw an 'IllegalArgumentException' when invalid data type names are provided" in {
    assertThrows[IllegalArgumentException](DataType.fromName("foo"))
    assertThrows[IllegalArgumentException](DataType.fromName("bar"))
    assertThrows[IllegalArgumentException](DataType.fromName(""))
    assertThrows[IllegalArgumentException](DataType.fromName(null))
  }

  "'DataType.size'" must "give the correct result" in {
    assert(DataType.Str.byteSize === -1)
    assert(DataType.Bool.byteSize === 1)
    // assert(DataType.Float16.byteSize === 2)
    assert(DataType.Float32.byteSize === 4)
    assert(DataType.Float64.byteSize === 8)
    // assert(DataType.BFloat16.byteSize === 2)
    // assert(DataType.Complex64.byteSize === 8)
    // assert(DataType.Complex128.byteSize === 16)
    assert(DataType.Int8.byteSize === 1)
    assert(DataType.Int16.byteSize === 2)
    assert(DataType.Int32.byteSize === 4)
    assert(DataType.Int64.byteSize === 8)
    assert(DataType.UInt8.byteSize === 1)
    assert(DataType.UInt16.byteSize === 2)
    assert(DataType.QInt8.byteSize === 1)
    assert(DataType.QInt16.byteSize === 2)
    assert(DataType.QInt32.byteSize === 4)
    assert(DataType.QUInt8.byteSize === 1)
    assert(DataType.QUInt16.byteSize === 2)
    assert(DataType.Resource.byteSize === -1)
  }

  // TODO: Add checks for data type priorities.

  "'DataType.isBoolean'" must "always work correctly" in {
    assert(DataType.Bool.isBoolean === true)
    assert(DataType.Str.isBoolean === false)
    // assert(DataType.Float16.isBoolean === false)
    assert(DataType.Float32.isBoolean === false)
    assert(DataType.Float64.isBoolean === false)
    // assert(DataType.BFloat16.isBoolean === false)
    // assert(DataType.Complex64.isBoolean === false)
    // assert(DataType.Complex128.isBoolean === false)
    assert(DataType.Int8.isBoolean === false)
    assert(DataType.Int16.isBoolean === false)
    assert(DataType.Int32.isBoolean === false)
    assert(DataType.Int64.isBoolean === false)
    assert(DataType.UInt8.isBoolean === false)
    assert(DataType.UInt16.isBoolean === false)
    assert(DataType.QInt8.isBoolean === false)
    assert(DataType.QInt16.isBoolean === false)
    assert(DataType.QInt32.isBoolean === false)
    assert(DataType.QUInt8.isBoolean === false)
    assert(DataType.QUInt16.isBoolean === false)
    assert(DataType.Resource.isBoolean === false)
  }

  "'DataType.isFloatingPoint'" must "always work correctly" in {
    assert(DataType.Bool.isFloatingPoint === false)
    assert(DataType.Str.isFloatingPoint === false)
    // assert(DataType.Float16.isFloatingPoint === true)
    assert(DataType.Float32.isFloatingPoint === true)
    assert(DataType.Float64.isFloatingPoint === true)
    // assert(DataType.BFloat16.isFloatingPoint === false)
    // assert(DataType.Complex64.isFloatingPoint === false)
    // assert(DataType.Complex128.isFloatingPoint === false)
    assert(DataType.Int8.isFloatingPoint === false)
    assert(DataType.Int16.isFloatingPoint === false)
    assert(DataType.Int32.isFloatingPoint === false)
    assert(DataType.Int64.isFloatingPoint === false)
    assert(DataType.UInt8.isFloatingPoint === false)
    assert(DataType.UInt16.isFloatingPoint === false)
    assert(DataType.QInt8.isFloatingPoint === false)
    assert(DataType.QInt16.isFloatingPoint === false)
    assert(DataType.QInt32.isFloatingPoint === false)
    assert(DataType.QUInt8.isFloatingPoint === false)
    assert(DataType.QUInt16.isFloatingPoint === false)
    assert(DataType.Resource.isFloatingPoint === false)
  }

  "'DataType.isInteger'" must "always work correctly" in {
    assert(DataType.Bool.isInteger === false)
    assert(DataType.Str.isInteger === false)
    // assert(DataType.Float16.isInteger === false)
    assert(DataType.Float32.isInteger === false)
    assert(DataType.Float64.isInteger === false)
    // assert(DataType.BFloat16.isInteger === false)
    // assert(DataType.Complex64.isInteger === false)
    // assert(DataType.Complex128.isInteger === false)
    assert(DataType.Int8.isInteger === true)
    assert(DataType.Int16.isInteger === true)
    assert(DataType.Int32.isInteger === true)
    assert(DataType.Int64.isInteger === true)
    assert(DataType.UInt8.isInteger === true)
    assert(DataType.UInt16.isInteger === true)
    assert(DataType.QInt8.isInteger === false)
    assert(DataType.QInt16.isInteger === false)
    assert(DataType.QInt32.isInteger === false)
    assert(DataType.QUInt8.isInteger === false)
    assert(DataType.QUInt16.isInteger === false)
    assert(DataType.Resource.isInteger === false)
  }

  "'DataType.isComplex'" must "always work correctly" in {
    assert(DataType.Bool.isComplex === false)
    assert(DataType.Str.isComplex === false)
    // assert(DataType.Float16.isComplex === false)
    assert(DataType.Float32.isComplex === false)
    assert(DataType.Float64.isComplex === false)
    // assert(DataType.BFloat16.isComplex === false)
    // assert(DataType.Complex64.isComplex === true)
    // assert(DataType.Complex128.isComplex === true)
    assert(DataType.Int8.isComplex === false)
    assert(DataType.Int16.isComplex === false)
    assert(DataType.Int32.isComplex === false)
    assert(DataType.Int64.isComplex === false)
    assert(DataType.UInt8.isComplex === false)
    assert(DataType.UInt16.isComplex === false)
    assert(DataType.QInt8.isComplex === false)
    assert(DataType.QInt16.isComplex === false)
    assert(DataType.QInt32.isComplex === false)
    assert(DataType.QUInt8.isComplex === false)
    assert(DataType.QUInt16.isComplex === false)
    assert(DataType.Resource.isComplex === false)
  }

  "'DataType.isQuantized'" must "always work correctly" in {
    assert(DataType.Bool.isQuantized === false)
    assert(DataType.Str.isQuantized === false)
    // assert(DataType.Float16.isQuantized === false)
    assert(DataType.Float32.isQuantized === false)
    assert(DataType.Float64.isQuantized === false)
    // assert(DataType.BFloat16.isQuantized === true)
    // assert(DataType.Complex64.isQuantized === false)
    // assert(DataType.Complex128.isQuantized === false)
    assert(DataType.Int8.isQuantized === false)
    assert(DataType.Int16.isQuantized === false)
    assert(DataType.Int32.isQuantized === false)
    assert(DataType.Int64.isQuantized === false)
    assert(DataType.UInt8.isQuantized === false)
    assert(DataType.UInt16.isQuantized === false)
    assert(DataType.QInt8.isQuantized === true)
    assert(DataType.QInt16.isQuantized === true)
    assert(DataType.QInt32.isQuantized === true)
    assert(DataType.QUInt8.isQuantized === true)
    assert(DataType.QUInt16.isQuantized === true)
    assert(DataType.Resource.isQuantized === false)
  }

  "'DataType.isUnsigned'" must "always work correctly" in {
    assert(DataType.Bool.isUnsigned === false)
    assert(DataType.Str.isUnsigned === false)
    // assert(DataType.Float16.isUnsigned === false)
    assert(DataType.Float32.isUnsigned === false)
    assert(DataType.Float64.isUnsigned === false)
    // assert(DataType.BFloat16.isUnsigned === false)
    // assert(DataType.Complex64.isUnsigned === false)
    // assert(DataType.Complex128.isUnsigned === false)
    assert(DataType.Int8.isUnsigned === false)
    assert(DataType.Int16.isUnsigned === false)
    assert(DataType.Int32.isUnsigned === false)
    assert(DataType.Int64.isUnsigned === false)
    assert(DataType.UInt8.isUnsigned === true)
    assert(DataType.UInt16.isUnsigned === true)
    assert(DataType.QInt8.isUnsigned === false)
    assert(DataType.QInt16.isUnsigned === false)
    assert(DataType.QInt32.isUnsigned === false)
    assert(DataType.QUInt8.isUnsigned === false)
    assert(DataType.QUInt16.isUnsigned === false)
    assert(DataType.Resource.isUnsigned === false)
  }

  // "'DataType.real'" must "always work correctly" in {
  //   assert(DataType.Bool.real === DataType.Bool)
  //   assert(DataType.Str.real === DataType.Str)
  //   assert(DataType.Float16.real === DataType.Float16)
  //   assert(DataType.Float32.real === DataType.Float32)
  //   assert(DataType.Float64.real === DataType.Float64)
  //   assert(DataType.BFloat16.real === DataType.BFloat16)
  //   assert(DataType.Complex64.real === DataType.Float32)
  //   assert(DataType.Complex128.real === DataType.Float64)
  //   assert(DataType.Int8.real === DataType.Int8)
  //   assert(DataType.Int16.real === DataType.Int16)
  //   assert(DataType.Int32.real === DataType.Int32)
  //   assert(DataType.Int64.real === DataType.Int64)
  //   assert(DataType.UInt8.real === DataType.UInt8)
  //   assert(DataType.UInt16.real === DataType.UInt16)
  //   assert(DataType.QInt8.real === DataType.QInt8)
  //   assert(DataType.QInt16.real === DataType.QInt16)
  //   assert(DataType.QInt32.real === DataType.QInt32)
  //   assert(DataType.QUInt8.real === DataType.QUInt8)
  //   assert(DataType.QUInt16.real === DataType.QUInt16)
  //   assert(DataType.Resource.real === DataType.Resource)
  // }

  "'DataType.equals'" must "always work correctly" in {
    assert(DataType.Float32 === DataType.Float32)
    assert(DataType.Float32 !== DataType.Float64)
    assert(DataType.Int8 !== DataType.UInt8)
    assert(DataType.Int8 !== DataType.QInt8)
  }

  "'DataType.cast'" must "work correctly when provided values of supported types" in {
    assert(DataType.Bool.cast(false) === false)
    assert(DataType.Bool.cast(true) === true)
    assert(DataType.Str.cast("foo") === "foo")
    // assert(DataType.Str.cast(false) === "false")
    // assert(DataType.Str.cast(1.0) === "1.0")
    // assert(DataType.Str.cast(1f) === "1.0")
    // assert(DataType.Str.cast(-2L) === "-2")
    // assert(DataType.Float16.cast(-2.0) === -2f)
    // assert(DataType.Float16.cast(-2L) === -2f)
    assert(DataType.Float32.cast(-2.0) === -2f)
    assert(DataType.Float32.cast(2) === 2f)
    assert(DataType.Float64.cast(2) === 2.0)
    assert(DataType.Float64.cast(-2f) === -2.0)
    assert(DataType.Float64.cast(-2L) === -2.0)
    /// assert(DataType.BFloat16.cast(-2) === -2f)
    // TODO: Add complex data type checks.
    assert(DataType.Int8.cast(-2L) === -2.toByte)
    assert(DataType.Int8.cast(-2.0) === -2.toByte)
    assert(DataType.Int8.cast(UInt8(2)) === 2.toByte)
    assert(DataType.Int16.cast(-2L) === -2.toShort)
    assert(DataType.Int32.cast(-2L) === -2)
    assert(DataType.Int64.cast(-2.0) === -2L)
    assert(DataType.Int64.cast(UInt8(2)) === 2L)
    assert(DataType.UInt8.cast(UInt8(2)) === UInt8(2))
    assert(DataType.UInt8.cast(UInt16(2)) === UInt8(2))
    assert(DataType.UInt8.cast(2L) === UInt8(2))
    assert(DataType.UInt8.cast(-2.0) === UInt8(254)) // TODO: Should this throw an error?
    assert(DataType.UInt16.cast(-UInt8(2)) === UInt16(65534)) // TODO: Should this throw an error?
    assert(DataType.UInt16.cast(UInt16(2)) === UInt16(2))
    assert(DataType.UInt16.cast(2L) === UInt16(2))
    assert(DataType.UInt16.cast(2.0) === UInt16(2))
    assert(DataType.QInt8.cast(-2L) === -2.toByte)
    assert(DataType.QInt8.cast(-2.0) === -2.toByte)
    assert(DataType.QInt8.cast(UInt8(2)) === 2.toByte)
    assert(DataType.QInt16.cast(-2L) === -2.toShort)
    assert(DataType.QInt32.cast(-2L) === -2)
    assert(DataType.QUInt8.cast(UInt8(2)) === UInt8(2))
    assert(DataType.QUInt8.cast(UInt16(2)) === UInt8(2))
    assert(DataType.QUInt8.cast(2L) === UInt8(2))
    assert(DataType.QUInt8.cast(-2.0) === UInt8(254)) // TODO: Should this throw an error?
    assert(DataType.QUInt16.cast(-UInt8(2)) === UInt16(65534)) // TODO: Should this throw an error?
    assert(DataType.QUInt16.cast(UInt16(2)) === UInt16(2))
    assert(DataType.QUInt16.cast(2L) === UInt16(2))
    assert(DataType.QUInt16.cast(2.0) === UInt16(2))
  }

  // TODO: Add 'InvalidCastException' checks.

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("DataType.Float32.cast(Array(1))")
    assertDoesNotCompile("DataType.Float32.cast(1.asInstanceOf[Any])")
  }

  "'DataType.putElementInBuffer'" must "work correctly when provided values of supported types" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(12)
    DataType.Int32.putElementInBuffer(buffer, 0, 1)
    DataType.Int32.putElementInBuffer(buffer, 4, 16)
    DataType.Int32.putElementInBuffer(buffer, 8, 257)
    assert(buffer.get(0) === Int8(0x00))
    assert(buffer.get(1) === Int8(0x00))
    assert(buffer.get(2) === Int8(0x00))
    assert(buffer.get(3) === Int8(0x01))
    assert(buffer.get(4) === Int8(0x00))
    assert(buffer.get(5) === Int8(0x00))
    assert(buffer.get(6) === Int8(0x00))
    assert(buffer.get(7) === Int8(0x10))
    assert(buffer.get(8) === Int8(0x00))
    assert(buffer.get(9) === Int8(0x00))
    assert(buffer.get(10) === Int8(0x01))
    assert(buffer.get(11) === Int8(0x01))
    // TODO: Add checks for other data types.
  }

  it must "throw an 'IndexOutOfBoundsException' exception when writing beyond the buffer boundaries" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(10)
    DataType.Int32.putElementInBuffer(buffer, 0, 1)
    DataType.Int32.putElementInBuffer(buffer, 4, 16)
    assertThrows[IndexOutOfBoundsException](DataType.Int32.putElementInBuffer(buffer, 8, 257))
  }

  "'DataType.getElementFromBuffer'" must "match the behavior of 'DataType.putElementInBuffer'" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(1024)
    // DataType.Float16.putElementInBuffer(buffer, 0, 2f)
    DataType.Float32.putElementInBuffer(buffer, 2, -4.23f)
    DataType.Float64.putElementInBuffer(buffer, 6, 3.45)
    // DataType.BFloat16.putElementInBuffer(buffer, 14, -1.23f)
    DataType.Int8.putElementInBuffer(buffer, 16, 4.toByte)
    DataType.Int16.putElementInBuffer(buffer, 17, (-2).toShort)
    DataType.Int32.putElementInBuffer(buffer, 19, 54)
    DataType.Int64.putElementInBuffer(buffer, 23, -3416L)
    DataType.UInt8.putElementInBuffer(buffer, 31, UInt8(34))
    DataType.UInt16.putElementInBuffer(buffer, 32, UInt16(657))
    DataType.QInt8.putElementInBuffer(buffer, 34, (-4).toByte)
    DataType.QInt16.putElementInBuffer(buffer, 35, 32.toShort)
    DataType.QInt32.putElementInBuffer(buffer, 37, -548979)
    DataType.QUInt8.putElementInBuffer(buffer, 41, UInt8(254))
    DataType.QUInt16.putElementInBuffer(buffer, 42, UInt16(765))
    DataType.Bool.putElementInBuffer(buffer, 44, true)
    DataType.Bool.putElementInBuffer(buffer, 45, false)
    // TODO: Add checks for the string data type.
    // assert(DataType.Float16.getElementFromBuffer(buffer, 0) === 2f)
    assert(DataType.Float32.getElementFromBuffer(buffer, 2) === -4.23f)
    assert(DataType.Float64.getElementFromBuffer(buffer, 6) === 3.45)
    // assert(DataType.BFloat16.getElementFromBuffer(buffer, 14) === -1.23f)
    assert(DataType.Int8.getElementFromBuffer(buffer, 16) === 4.toByte)
    assert(DataType.Int16.getElementFromBuffer(buffer, 17) === (-2).toShort)
    assert(DataType.Int32.getElementFromBuffer(buffer, 19) === 54)
    assert(DataType.Int64.getElementFromBuffer(buffer, 23) === -3416L)
    assert(DataType.UInt8.getElementFromBuffer(buffer, 31) === UInt8(34))
    assert(DataType.UInt16.getElementFromBuffer(buffer, 32) === UInt16(657))
    assert(DataType.QInt8.getElementFromBuffer(buffer, 34) === (-4).toByte)
    assert(DataType.QInt16.getElementFromBuffer(buffer, 35) === 32.toShort)
    assert(DataType.QInt32.getElementFromBuffer(buffer, 37) === -548979)
    assert(DataType.QUInt8.getElementFromBuffer(buffer, 41) === UInt8(254))
    assert(DataType.QUInt16.getElementFromBuffer(buffer, 42) === UInt16(765))
    assert(DataType.Bool.getElementFromBuffer(buffer, 44) === true)
    assert(DataType.Bool.getElementFromBuffer(buffer, 45) === false)
  }
}
