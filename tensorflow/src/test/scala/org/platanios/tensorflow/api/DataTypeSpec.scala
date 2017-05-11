package org.platanios.tensorflow.api

import java.nio.ByteBuffer

import org.scalatest._

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
class DataTypeSpec extends FlatSpec with Matchers {
  "'DataType.dataTypeOf'" must "work correctly when valid values are provided" in {
    assert(DataType.dataTypeOf(1.0f) === TFFloat32)
    assert(DataType.dataTypeOf(1.0) === TFFloat64)
    assert(DataType.dataTypeOf(1.asInstanceOf[Byte]) === TFInt8)
    assert(DataType.dataTypeOf(1.asInstanceOf[Short]) === TFInt16)
    assert(DataType.dataTypeOf(1) === TFInt32)
    assert(DataType.dataTypeOf(1L) === TFInt64)
    assert(DataType.dataTypeOf(UByte(1)) === TFUInt8)
    assert(DataType.dataTypeOf(UShort(1)) === TFUInt16)
    // assert(DataType.dataTypeOf('a') === TFUInt16)
    assert(DataType.dataTypeOf(true) === TFBoolean)
    // assert(DataType.dataTypeOf("foo") === TFString)
    // assert(DataType.dataTypeOf(Tensor(1, 2)) === TFInt32)
  }

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(Array(1))")
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(1.asInstanceOf[Any])")
  }

  "'DataType.fromCValue'" must "work correctly when valid C values are provided" in {
    assert(DataType.fromCValue(1) === TFFloat32)
    assert(DataType.fromCValue(2) === TFFloat64)
    assert(DataType.fromCValue(3) === TFInt32)
    assert(DataType.fromCValue(4) === TFUInt8)
    assert(DataType.fromCValue(5) === TFInt16)
    assert(DataType.fromCValue(6) === TFInt8)
    assert(DataType.fromCValue(7) === TFString)
    // assert(DataType.fromCValue(8) === TFComplex64)
    assert(DataType.fromCValue(9) === TFInt64)
    assert(DataType.fromCValue(10) === TFBoolean)
    assert(DataType.fromCValue(11) === TFQInt8)
    assert(DataType.fromCValue(12) === TFQUInt8)
    assert(DataType.fromCValue(13) === TFQInt32)
    // assert(DataType.fromCValue(14) === TFBFloat16)
    assert(DataType.fromCValue(15) === TFQInt16)
    assert(DataType.fromCValue(16) === TFQUInt16)
    assert(DataType.fromCValue(17) === TFUInt16)
    // assert(DataType.fromCValue(18) === TFComplex128)
    // assert(DataType.fromCValue(19) === TFFloat16)
    assert(DataType.fromCValue(20) === TFResource)
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
    assert(DataType.fromName("TFFloat32") === TFFloat32)
    assert(DataType.fromName("TFFloat64") === TFFloat64)
    assert(DataType.fromName("TFInt32") === TFInt32)
    assert(DataType.fromName("TFUInt8") === TFUInt8)
    assert(DataType.fromName("TFInt16") === TFInt16)
    assert(DataType.fromName("TFInt8") === TFInt8)
    assert(DataType.fromName("TFString") === TFString)
    // assert(DataType.fromName("TFComplex64") === TFComplex64)
    assert(DataType.fromName("TFInt64") === TFInt64)
    assert(DataType.fromName("TFBoolean") === TFBoolean)
    assert(DataType.fromName("TFQInt8") === TFQInt8)
    assert(DataType.fromName("TFQUInt8") === TFQUInt8)
    assert(DataType.fromName("TFQInt32") === TFQInt32)
    // assert(DataType.fromName("TFBFloat16") === TFBFloat16)
    assert(DataType.fromName("TFQInt16") === TFQInt16)
    assert(DataType.fromName("TFQUInt16") === TFQUInt16)
    assert(DataType.fromName("TFUInt16") === TFUInt16)
    // assert(DataType.fromName("TFComplex128") === TFComplex128)
    // assert(DataType.fromName("TFFloat16") === TFFloat16)
    assert(DataType.fromName("TFResource") === TFResource)
  }

  it must "throw an 'IllegalArgumentException' when invalid data type names are provided" in {
    assertThrows[IllegalArgumentException](DataType.fromName("foo"))
    assertThrows[IllegalArgumentException](DataType.fromName("bar"))
    assertThrows[IllegalArgumentException](DataType.fromName(""))
    assertThrows[IllegalArgumentException](DataType.fromName(null))
  }

  "'DataType.size'" must "give the correct result" in {
    assert(TFString.byteSize === -1)
    assert(TFBoolean.byteSize === 1)
    // assert(TFFloat16.byteSize === 2)
    assert(TFFloat32.byteSize === 4)
    assert(TFFloat64.byteSize === 8)
    // assert(TFBFloat16.byteSize === 2)
    // assert(TFComplex64.byteSize === 8)
    // assert(TFComplex128.byteSize === 16)
    assert(TFInt8.byteSize === 1)
    assert(TFInt16.byteSize === 2)
    assert(TFInt32.byteSize === 4)
    assert(TFInt64.byteSize === 8)
    assert(TFUInt8.byteSize === 1)
    assert(TFUInt16.byteSize === 2)
    assert(TFQInt8.byteSize === 1)
    assert(TFQInt16.byteSize === 2)
    assert(TFQInt32.byteSize === 4)
    assert(TFQUInt8.byteSize === 1)
    assert(TFQUInt16.byteSize === 2)
    assert(TFResource.byteSize === -1)
  }

  // TODO: Add checks for data type priorities.

  "'DataType.isBoolean'" must "always work correctly" in {
    assert(TFBoolean.isBoolean === true)
    assert(TFString.isBoolean === false)
    // assert(TFFloat16.isBoolean === false)
    assert(TFFloat32.isBoolean === false)
    assert(TFFloat64.isBoolean === false)
    // assert(TFBFloat16.isBoolean === false)
    // assert(TFComplex64.isBoolean === false)
    // assert(TFComplex128.isBoolean === false)
    assert(TFInt8.isBoolean === false)
    assert(TFInt16.isBoolean === false)
    assert(TFInt32.isBoolean === false)
    assert(TFInt64.isBoolean === false)
    assert(TFUInt8.isBoolean === false)
    assert(TFUInt16.isBoolean === false)
    assert(TFQInt8.isBoolean === false)
    assert(TFQInt16.isBoolean === false)
    assert(TFQInt32.isBoolean === false)
    assert(TFQUInt8.isBoolean === false)
    assert(TFQUInt16.isBoolean === false)
    assert(TFResource.isBoolean === false)
  }

  "'DataType.isFloatingPoint'" must "always work correctly" in {
    assert(TFBoolean.isFloatingPoint === false)
    assert(TFString.isFloatingPoint === false)
    // assert(TFFloat16.isFloatingPoint === true)
    assert(TFFloat32.isFloatingPoint === true)
    assert(TFFloat64.isFloatingPoint === true)
    // assert(TFBFloat16.isFloatingPoint === false)
    // assert(TFComplex64.isFloatingPoint === false)
    // assert(TFComplex128.isFloatingPoint === false)
    assert(TFInt8.isFloatingPoint === false)
    assert(TFInt16.isFloatingPoint === false)
    assert(TFInt32.isFloatingPoint === false)
    assert(TFInt64.isFloatingPoint === false)
    assert(TFUInt8.isFloatingPoint === false)
    assert(TFUInt16.isFloatingPoint === false)
    assert(TFQInt8.isFloatingPoint === false)
    assert(TFQInt16.isFloatingPoint === false)
    assert(TFQInt32.isFloatingPoint === false)
    assert(TFQUInt8.isFloatingPoint === false)
    assert(TFQUInt16.isFloatingPoint === false)
    assert(TFResource.isFloatingPoint === false)
  }

  "'DataType.isInteger'" must "always work correctly" in {
    assert(TFBoolean.isInteger === false)
    assert(TFString.isInteger === false)
    // assert(TFFloat16.isInteger === false)
    assert(TFFloat32.isInteger === false)
    assert(TFFloat64.isInteger === false)
    // assert(TFBFloat16.isInteger === false)
    // assert(TFComplex64.isInteger === false)
    // assert(TFComplex128.isInteger === false)
    assert(TFInt8.isInteger === true)
    assert(TFInt16.isInteger === true)
    assert(TFInt32.isInteger === true)
    assert(TFInt64.isInteger === true)
    assert(TFUInt8.isInteger === true)
    assert(TFUInt16.isInteger === true)
    assert(TFQInt8.isInteger === false)
    assert(TFQInt16.isInteger === false)
    assert(TFQInt32.isInteger === false)
    assert(TFQUInt8.isInteger === false)
    assert(TFQUInt16.isInteger === false)
    assert(TFResource.isInteger === false)
  }

  "'DataType.isComplex'" must "always work correctly" in {
    assert(TFBoolean.isComplex === false)
    assert(TFString.isComplex === false)
    // assert(TFFloat16.isComplex === false)
    assert(TFFloat32.isComplex === false)
    assert(TFFloat64.isComplex === false)
    // assert(TFBFloat16.isComplex === false)
    // assert(TFComplex64.isComplex === true)
    // assert(TFComplex128.isComplex === true)
    assert(TFInt8.isComplex === false)
    assert(TFInt16.isComplex === false)
    assert(TFInt32.isComplex === false)
    assert(TFInt64.isComplex === false)
    assert(TFUInt8.isComplex === false)
    assert(TFUInt16.isComplex === false)
    assert(TFQInt8.isComplex === false)
    assert(TFQInt16.isComplex === false)
    assert(TFQInt32.isComplex === false)
    assert(TFQUInt8.isComplex === false)
    assert(TFQUInt16.isComplex === false)
    assert(TFResource.isComplex === false)
  }

  "'DataType.isQuantized'" must "always work correctly" in {
    assert(TFBoolean.isQuantized === false)
    assert(TFString.isQuantized === false)
    // assert(TFFloat16.isQuantized === false)
    assert(TFFloat32.isQuantized === false)
    assert(TFFloat64.isQuantized === false)
    // assert(TFBFloat16.isQuantized === true)
    // assert(TFComplex64.isQuantized === false)
    // assert(TFComplex128.isQuantized === false)
    assert(TFInt8.isQuantized === false)
    assert(TFInt16.isQuantized === false)
    assert(TFInt32.isQuantized === false)
    assert(TFInt64.isQuantized === false)
    assert(TFUInt8.isQuantized === false)
    assert(TFUInt16.isQuantized === false)
    assert(TFQInt8.isQuantized === true)
    assert(TFQInt16.isQuantized === true)
    assert(TFQInt32.isQuantized === true)
    assert(TFQUInt8.isQuantized === true)
    assert(TFQUInt16.isQuantized === true)
    assert(TFResource.isQuantized === false)
  }

  "'DataType.isUnsigned'" must "always work correctly" in {
    assert(TFBoolean.isUnsigned === false)
    assert(TFString.isUnsigned === false)
    // assert(TFFloat16.isUnsigned === false)
    assert(TFFloat32.isUnsigned === false)
    assert(TFFloat64.isUnsigned === false)
    // assert(TFBFloat16.isUnsigned === false)
    // assert(TFComplex64.isUnsigned === false)
    // assert(TFComplex128.isUnsigned === false)
    assert(TFInt8.isUnsigned === false)
    assert(TFInt16.isUnsigned === false)
    assert(TFInt32.isUnsigned === false)
    assert(TFInt64.isUnsigned === false)
    assert(TFUInt8.isUnsigned === true)
    assert(TFUInt16.isUnsigned === true)
    assert(TFQInt8.isUnsigned === false)
    assert(TFQInt16.isUnsigned === false)
    assert(TFQInt32.isUnsigned === false)
    assert(TFQUInt8.isUnsigned === false)
    assert(TFQUInt16.isUnsigned === false)
    assert(TFResource.isUnsigned === false)
  }

  // "'DataType.real'" must "always work correctly" in {
  //   assert(TFBool.real === TFBoolean)
  //   assert(TFStr.real === TFStrinh)
  //   assert(TFFloat16.real === TFFloat16)
  //   assert(TFFloat32.real === TFFloat32)
  //   assert(TFFloat64.real === TFFloat64)
  //   assert(TFBFloat16.real === TFBFloat16)
  //   assert(TFComplex64.real === TFFloat32)
  //   assert(TFComplex128.real === TFFloat64)
  //   assert(TFInt8.real === TFInt8)
  //   assert(TFInt16.real === TFInt16)
  //   assert(TFInt32.real === TFInt32)
  //   assert(TFInt64.real === TFInt64)
  //   assert(TFUInt8.real === TFUInt8)
  //   assert(TFUInt16.real === TFUInt16)
  //   assert(TFQInt8.real === TFQInt8)
  //   assert(TFQInt16.real === TFQInt16)
  //   assert(TFQInt32.real === TFQInt32)
  //   assert(TFQUInt8.real === TFQUInt8)
  //   assert(TFQUInt16.real === TFQUInt16)
  //   assert(TFResource.real === TFResource)
  // }

  "'DataType.equals'" must "always work correctly" in {
    assert(TFFloat32 === TFFloat32)
    assert(TFFloat32 !== TFFloat64)
    assert(TFInt8 !== TFUInt8)
    assert(TFInt8 !== TFQInt8)
  }

  "'DataType.cast'" must "work correctly when provided values of supported types" in {
    assert(TFBoolean.cast(false) === false)
    assert(TFBoolean.cast(true) === true)
    assert(TFString.cast("foo") === "foo")
    // assert(TFString.cast(false) === "false")
    // assert(TFString.cast(1.0) === "1.0")
    // assert(TFString.cast(1f) === "1.0")
    // assert(TFString.cast(-2L) === "-2")
    // assert(TFFloat16.cast(-2.0) === -2f)
    // assert(TFFloat16.cast(-2L) === -2f)
    assert(TFFloat32.cast(-2.0) === -2f)
    assert(TFFloat32.cast(2) === 2f)
    assert(TFFloat64.cast(2) === 2.0)
    assert(TFFloat64.cast(-2f) === -2.0)
    assert(TFFloat64.cast(-2L) === -2.0)
    /// assert(TFBFloat16.cast(-2) === -2f)
    // TODO: Add complex data type checks.
    assert(TFInt8.cast(-2L) === -2.toByte)
    assert(TFInt8.cast(-2.0) === -2.toByte)
    assert(TFInt8.cast(UByte(2)) === 2.toByte)
    assert(TFInt16.cast(-2L) === -2.toShort)
    assert(TFInt32.cast(-2L) === -2)
    assert(TFInt64.cast(-2.0) === -2L)
    assert(TFInt64.cast(UByte(2)) === 2L)
    assert(TFUInt8.cast(UByte(2)) === UByte(2))
    assert(TFUInt8.cast(UShort(2)) === UByte(2))
    assert(TFUInt8.cast(2L) === UByte(2))
    assert(TFUInt8.cast(-2.0) === UByte(254)) // TODO: Should this throw an error?
    assert(TFUInt16.cast(-UByte(2)) === UShort(254)) // TODO: Should this throw an error?
    assert(TFUInt16.cast(UShort(2)) === UShort(2))
    assert(TFUInt16.cast(2L) === UShort(2))
    assert(TFUInt16.cast(2.0) === UShort(2))
    assert(TFQInt8.cast(-2L) === -2.toByte)
    assert(TFQInt8.cast(-2.0) === -2.toByte)
    assert(TFQInt8.cast(UByte(2)) === 2.toByte)
    assert(TFQInt16.cast(-2L) === -2.toShort)
    assert(TFQInt32.cast(-2L) === -2)
    assert(TFQUInt8.cast(UByte(2)) === UByte(2))
    assert(TFQUInt8.cast(UShort(2)) === UByte(2))
    assert(TFQUInt8.cast(2L) === UByte(2))
    assert(TFQUInt8.cast(-2.0) === UByte(254)) // TODO: Should this throw an error?
    assert(TFQUInt16.cast(-UByte(2)) === UShort(254)) // TODO: Should this throw an error?
    assert(TFQUInt16.cast(UShort(2)) === UShort(2))
    assert(TFQUInt16.cast(2L) === UShort(2))
    assert(TFQUInt16.cast(2.0) === UShort(2))
  }

  // TODO: Add 'InvalidCastException' checks.

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("DataType.Float32.cast(Array(1))")
    assertDoesNotCompile("DataType.Float32.cast(1.asInstanceOf[Any])")
  }

  "'DataType.putElementInBuffer'" must "work correctly when provided values of supported types" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(12)
    TFInt32.putElementInBuffer(buffer, 0, 1)
    TFInt32.putElementInBuffer(buffer, 4, 16)
    TFInt32.putElementInBuffer(buffer, 8, 257)
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
    TFInt32.putElementInBuffer(buffer, 0, 1)
    TFInt32.putElementInBuffer(buffer, 4, 16)
    assertThrows[IndexOutOfBoundsException](TFInt32.putElementInBuffer(buffer, 8, 257))
  }

  "'DataType.getElementFromBuffer'" must "match the behavior of 'DataType.putElementInBuffer'" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(1024)
    // TFFloat16.putElementInBuffer(buffer, 0, 2f)
    TFFloat32.putElementInBuffer(buffer, 2, -4.23f)
    TFFloat64.putElementInBuffer(buffer, 6, 3.45)
    // TFBFloat16.putElementInBuffer(buffer, 14, -1.23f)
    TFInt8.putElementInBuffer(buffer, 16, 4.toByte)
    TFInt16.putElementInBuffer(buffer, 17, (-2).toShort)
    TFInt32.putElementInBuffer(buffer, 19, 54)
    TFInt64.putElementInBuffer(buffer, 23, -3416L)
    TFUInt8.putElementInBuffer(buffer, 31, UByte(34))
    TFUInt16.putElementInBuffer(buffer, 32, UShort(657))
    TFQInt8.putElementInBuffer(buffer, 34, (-4).toByte)
    TFQInt16.putElementInBuffer(buffer, 35, 32.toShort)
    TFQInt32.putElementInBuffer(buffer, 37, -548979)
    TFQUInt8.putElementInBuffer(buffer, 41, UByte(254))
    TFQUInt16.putElementInBuffer(buffer, 42, UShort(765))
    TFBoolean.putElementInBuffer(buffer, 44, true)
    TFBoolean.putElementInBuffer(buffer, 45, false)
    // TODO: Add checks for the string data type.
    // assert(TFFloat16.getElementFromBuffer(buffer, 0) === 2f)
    assert(TFFloat32.getElementFromBuffer(buffer, 2) === -4.23f)
    assert(TFFloat64.getElementFromBuffer(buffer, 6) === 3.45)
    // assert(TFBFloat16.getElementFromBuffer(buffer, 14) === -1.23f)
    assert(TFInt8.getElementFromBuffer(buffer, 16) === 4.toByte)
    assert(TFInt16.getElementFromBuffer(buffer, 17) === (-2).toShort)
    assert(TFInt32.getElementFromBuffer(buffer, 19) === 54)
    assert(TFInt64.getElementFromBuffer(buffer, 23) === -3416L)
    assert(TFUInt8.getElementFromBuffer(buffer, 31) === UByte(34))
    assert(TFUInt16.getElementFromBuffer(buffer, 32) === UShort(657))
    assert(TFQInt8.getElementFromBuffer(buffer, 34) === (-4).toByte)
    assert(TFQInt16.getElementFromBuffer(buffer, 35) === 32.toShort)
    assert(TFQInt32.getElementFromBuffer(buffer, 37) === -548979)
    assert(TFQUInt8.getElementFromBuffer(buffer, 41) === UByte(254))
    assert(TFQUInt16.getElementFromBuffer(buffer, 42) === UShort(765))
    assert(TFBoolean.getElementFromBuffer(buffer, 44) === true)
    assert(TFBoolean.getElementFromBuffer(buffer, 45) === false)
  }
}
