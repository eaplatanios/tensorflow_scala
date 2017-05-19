// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.tf._

import org.scalatest._
import spire.math.{UByte, UShort}

import java.nio.ByteBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
class DataTypeSpec extends FlatSpec with Matchers {
  "'DataType.dataTypeOf'" must "work correctly when valid values are provided" in {
    assert(DataType.dataTypeOf(1.0f) === FLOAT32)
    assert(DataType.dataTypeOf(1.0) === FLOAT64)
    assert(DataType.dataTypeOf(1.asInstanceOf[Byte]) === INT8)
    assert(DataType.dataTypeOf(1.asInstanceOf[Short]) === INT16)
    assert(DataType.dataTypeOf(1) === INT32)
    assert(DataType.dataTypeOf(1L) === INT64)
    assert(DataType.dataTypeOf(UByte(1)) === UINT8)
    assert(DataType.dataTypeOf(UShort(1)) === UINT16)
    // assert(DataType.dataTypeOf('a') === UINT16)
    assert(DataType.dataTypeOf(true) === BOOLEAN)
    // assert(DataType.dataTypeOf("foo") === STRING)
    // assert(DataType.dataTypeOf(Tensor(1, 2)) === INT32)
  }

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(Array(1))")
    assertDoesNotCompile("val d: DataType = DataType.dataTypeOf(1.asInstanceOf[Any])")
  }

  "'DataType.fromCValue'" must "work correctly when valid C values are provided" in {
    assert(DataType.fromCValue(1) === FLOAT32)
    assert(DataType.fromCValue(2) === FLOAT64)
    assert(DataType.fromCValue(3) === INT32)
    assert(DataType.fromCValue(4) === UINT8)
    assert(DataType.fromCValue(5) === INT16)
    assert(DataType.fromCValue(6) === INT8)
    assert(DataType.fromCValue(7) === STRING)
    // assert(DataType.fromCValue(8) === COMPLEX64)
    assert(DataType.fromCValue(9) === INT64)
    assert(DataType.fromCValue(10) === BOOLEAN)
    assert(DataType.fromCValue(11) === QINT8)
    assert(DataType.fromCValue(12) === QUINT8)
    assert(DataType.fromCValue(13) === QINT32)
    // assert(DataType.fromCValue(14) === BFLOAT16)
    assert(DataType.fromCValue(15) === QINT16)
    assert(DataType.fromCValue(16) === QUINT16)
    assert(DataType.fromCValue(17) === UINT16)
    // assert(DataType.fromCValue(18) === COMPLEX128)
    // assert(DataType.fromCValue(19) === FLOAT16)
    assert(DataType.fromCValue(20) === RESOURCE)
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
    assert(DataType.fromName("FLOAT32") === FLOAT32)
    assert(DataType.fromName("FLOAT64") === FLOAT64)
    assert(DataType.fromName("INT32") === INT32)
    assert(DataType.fromName("UINT8") === UINT8)
    assert(DataType.fromName("INT16") === INT16)
    assert(DataType.fromName("INT8") === INT8)
    assert(DataType.fromName("STRING") === STRING)
    // assert(DataType.fromName("Complex64") === COMPLEX64)
    assert(DataType.fromName("INT64") === INT64)
    assert(DataType.fromName("BOOLEAN") === BOOLEAN)
    assert(DataType.fromName("QINT8") === QINT8)
    assert(DataType.fromName("QUINT8") === QUINT8)
    assert(DataType.fromName("QINT32") === QINT32)
    // assert(DataType.fromName("BFLOAT16") === BFLOAT16)
    assert(DataType.fromName("QINT16") === QINT16)
    assert(DataType.fromName("QUINT16") === QUINT16)
    assert(DataType.fromName("UINT16") === UINT16)
    // assert(DataType.fromName("COMPLEX128") === COMPLEX128))
    // assert(DataType.fromName("FLOAT16") === FLOAT16)
    assert(DataType.fromName("RESOURCE") === RESOURCE)
  }

  it must "throw an 'IllegalArgumentException' when invalid data type names are provided" in {
    assertThrows[IllegalArgumentException](DataType.fromName("foo"))
    assertThrows[IllegalArgumentException](DataType.fromName("bar"))
    assertThrows[IllegalArgumentException](DataType.fromName(""))
    assertThrows[IllegalArgumentException](DataType.fromName(null))
  }

  "'DataType.size'" must "give the correct result" in {
    assert(STRING.byteSize === -1)
    assert(BOOLEAN.byteSize === 1)
    // assert(FLOAT16.byteSize === 2)
    assert(FLOAT32.byteSize === 4)
    assert(FLOAT64.byteSize === 8)
    // assert(BFLOAT16.byteSize === 2)
    // assert(COMPLEX64.byteSize === 8)
    // assert(COMPLEX128.byteSize === 16)
    assert(INT8.byteSize === 1)
    assert(INT16.byteSize === 2)
    assert(INT32.byteSize === 4)
    assert(INT64.byteSize === 8)
    assert(UINT8.byteSize === 1)
    assert(UINT16.byteSize === 2)
    assert(QINT8.byteSize === 1)
    assert(QINT16.byteSize === 2)
    assert(QINT32.byteSize === 4)
    assert(QUINT8.byteSize === 1)
    assert(QUINT16.byteSize === 2)
    assert(RESOURCE.byteSize === -1)
  }

  // TODO: Add checks for data type priorities.

  "'DataType.isBoolean'" must "always work correctly" in {
    assert(BOOLEAN.isBoolean === true)
    assert(STRING.isBoolean === false)
    // assert(FLOAT16.isBoolean === false)
    assert(FLOAT32.isBoolean === false)
    assert(FLOAT64.isBoolean === false)
    // assert(BFLOAT16.isBoolean === false)
    // assert(Complex64.isBoolean === false)
    // assert(COMPLEX128.isBoolean === false)
    assert(INT8.isBoolean === false)
    assert(INT16.isBoolean === false)
    assert(INT32.isBoolean === false)
    assert(INT64.isBoolean === false)
    assert(UINT8.isBoolean === false)
    assert(UINT16.isBoolean === false)
    assert(QINT8.isBoolean === false)
    assert(QINT16.isBoolean === false)
    assert(QINT32.isBoolean === false)
    assert(QUINT8.isBoolean === false)
    assert(QUINT16.isBoolean === false)
    assert(RESOURCE.isBoolean === false)
  }

  "'DataType.isFloatingPoint'" must "always work correctly" in {
    assert(BOOLEAN.isFloatingPoint === false)
    assert(STRING.isFloatingPoint === false)
    // assert(FLOAT16.isFloatingPoint === true)
    assert(FLOAT32.isFloatingPoint === true)
    assert(FLOAT64.isFloatingPoint === true)
    // assert(BFLOAT16.isFloatingPoint === false)
    // assert(Complex64.isFloatingPoint === false)
    // assert(COMPLEX128.isFloatingPoint === false)
    assert(INT8.isFloatingPoint === false)
    assert(INT16.isFloatingPoint === false)
    assert(INT32.isFloatingPoint === false)
    assert(INT64.isFloatingPoint === false)
    assert(UINT8.isFloatingPoint === false)
    assert(UINT16.isFloatingPoint === false)
    assert(QINT8.isFloatingPoint === false)
    assert(QINT16.isFloatingPoint === false)
    assert(QINT32.isFloatingPoint === false)
    assert(QUINT8.isFloatingPoint === false)
    assert(QUINT16.isFloatingPoint === false)
    assert(RESOURCE.isFloatingPoint === false)
  }

  "'DataType.isInteger'" must "always work correctly" in {
    assert(BOOLEAN.isInteger === false)
    assert(STRING.isInteger === false)
    // assert(FLOAT16.isInteger === false)
    assert(FLOAT32.isInteger === false)
    assert(FLOAT64.isInteger === false)
    // assert(BFLOAT16.isInteger === false)
    // assert(Complex64.isInteger === false)
    // assert(COMPLEX128.isInteger === false)
    assert(INT8.isInteger === true)
    assert(INT16.isInteger === true)
    assert(INT32.isInteger === true)
    assert(INT64.isInteger === true)
    assert(UINT8.isInteger === true)
    assert(UINT16.isInteger === true)
    assert(QINT8.isInteger === false)
    assert(QINT16.isInteger === false)
    assert(QINT32.isInteger === false)
    assert(QUINT8.isInteger === false)
    assert(QUINT16.isInteger === false)
    assert(RESOURCE.isInteger === false)
  }

  "'DataType.isComplex'" must "always work correctly" in {
    assert(BOOLEAN.isComplex === false)
    assert(STRING.isComplex === false)
    // assert(FLOAT16.isComplex === false)
    assert(FLOAT32.isComplex === false)
    assert(FLOAT64.isComplex === false)
    // assert(BFLOAT16.isComplex === false)
    // assert(Complex64.isComplex === true)
    // assert(COMPLEX128.isComplex === true)
    assert(INT8.isComplex === false)
    assert(INT16.isComplex === false)
    assert(INT32.isComplex === false)
    assert(INT64.isComplex === false)
    assert(UINT8.isComplex === false)
    assert(UINT16.isComplex === false)
    assert(QINT8.isComplex === false)
    assert(QINT16.isComplex === false)
    assert(QINT32.isComplex === false)
    assert(QUINT8.isComplex === false)
    assert(QUINT16.isComplex === false)
    assert(RESOURCE.isComplex === false)
  }

  "'DataType.isQuantized'" must "always work correctly" in {
    assert(BOOLEAN.isQuantized === false)
    assert(STRING.isQuantized === false)
    // assert(FLOAT16.isQuantized === false)
    assert(FLOAT32.isQuantized === false)
    assert(FLOAT64.isQuantized === false)
    // assert(BFLOAT16.isQuantized === true)
    // assert(Complex64.isQuantized === false)
    // assert(COMPLEX128.isQuantized === false)
    assert(INT8.isQuantized === false)
    assert(INT16.isQuantized === false)
    assert(INT32.isQuantized === false)
    assert(INT64.isQuantized === false)
    assert(UINT8.isQuantized === false)
    assert(UINT16.isQuantized === false)
    assert(QINT8.isQuantized === true)
    assert(QINT16.isQuantized === true)
    assert(QINT32.isQuantized === true)
    assert(QUINT8.isQuantized === true)
    assert(QUINT16.isQuantized === true)
    assert(RESOURCE.isQuantized === false)
  }

  "'DataType.isUnsigned'" must "always work correctly" in {
    assert(BOOLEAN.isUnsigned === false)
    assert(STRING.isUnsigned === false)
    // assert(FLOAT16.isUnsigned === false)
    assert(FLOAT32.isUnsigned === false)
    assert(FLOAT64.isUnsigned === false)
    // assert(BFLOAT16.isUnsigned === false)
    // assert(Complex64.isUnsigned === false)
    // assert(COMPLEX128.isUnsigned === false)
    assert(INT8.isUnsigned === false)
    assert(INT16.isUnsigned === false)
    assert(INT32.isUnsigned === false)
    assert(INT64.isUnsigned === false)
    assert(UINT8.isUnsigned === true)
    assert(UINT16.isUnsigned === true)
    assert(QINT8.isUnsigned === false)
    assert(QINT16.isUnsigned === false)
    assert(QINT32.isUnsigned === false)
    assert(QUINT8.isUnsigned === false)
    assert(QUINT16.isUnsigned === false)
    assert(RESOURCE.isUnsigned === false)
  }

  // "'DataType.real'" must "always work correctly" in {
  //   assert(Bool.real === BOOLEAN)
  //   assert(Str.real === Strinh)
  //   assert(FLOAT16.real === FLOAT16)
  //   assert(FLOAT32.real === FLOAT32)
  //   assert(FLOAT64.real === FLOAT64)
  //   assert(BFLOAT16.real === BFLOAT16)
  //   assert(Complex64.real === FLOAT32)
  //   assert(COMPLEX128.real === FLOAT64)
  //   assert(INT8.real === INT8)
  //   assert(INT16.real === INT16)
  //   assert(INT32.real === INT32)
  //   assert(INT64.real === INT64)
  //   assert(UINT8.real === UINT8)
  //   assert(UINT16.real === UINT16)
  //   assert(QINT8.real === QINT8)
  //   assert(QINT16.real === QINT16)
  //   assert(QINT32.real === QINT32)
  //   assert(QUINT8.real === QUINT8)
  //   assert(QUINT16.real === QUINT16)
  //   assert(RESOURCE.real === RESOURCE)
  // }

  "'DataType.equals'" must "always work correctly" in {
    assert(FLOAT32 === FLOAT32)
    assert(FLOAT32 !== FLOAT64)
    assert(INT8 !== UINT8)
    assert(INT8 !== QINT8)
  }

  "'DataType.cast'" must "work correctly when provided values of supported types" in {
    assert(BOOLEAN.cast(false) === false)
    assert(BOOLEAN.cast(true) === true)
    assert(STRING.cast("foo") === "foo")
    // assert(STRING.cast(false) === "false")
    // assert(STRING.cast(1.0) === "1.0")
    // assert(STRING.cast(1f) === "1.0")
    // assert(STRING.cast(-2L) === "-2")
    // assert(FLOAT16.cast(-2.0) === -2f)
    // assert(FLOAT16.cast(-2L) === -2f)
    assert(FLOAT32.cast(-2.0) === -2f)
    assert(FLOAT32.cast(2) === 2f)
    assert(FLOAT64.cast(2) === 2.0)
    assert(FLOAT64.cast(-2f) === -2.0)
    assert(FLOAT64.cast(-2L) === -2.0)
    /// assert(BFLOAT16.cast(-2) === -2f)
    // TODO: Add complex data type checks.
    assert(INT8.cast(-2L) === -2.toByte)
    assert(INT8.cast(-2.0) === -2.toByte)
    assert(INT8.cast(UByte(2)) === 2.toByte)
    assert(INT16.cast(-2L) === -2.toShort)
    assert(INT32.cast(-2L) === -2)
    assert(INT64.cast(-2.0) === -2L)
    assert(INT64.cast(UByte(2)) === 2L)
    assert(UINT8.cast(UByte(2)) === UByte(2))
    assert(UINT8.cast(UShort(2)) === UByte(2))
    assert(UINT8.cast(2L) === UByte(2))
    assert(UINT8.cast(-2.0) === UByte(254)) // TODO: Should this throw an error?
    assert(UINT16.cast(-UByte(2)) === UShort(254)) // TODO: Should this throw an error?
    assert(UINT16.cast(UShort(2)) === UShort(2))
    assert(UINT16.cast(2L) === UShort(2))
    assert(UINT16.cast(2.0) === UShort(2))
    assert(QINT8.cast(-2L) === -2.toByte)
    assert(QINT8.cast(-2.0) === -2.toByte)
    assert(QINT8.cast(UByte(2)) === 2.toByte)
    assert(QINT16.cast(-2L) === -2.toShort)
    assert(QINT32.cast(-2L) === -2)
    assert(QUINT8.cast(UByte(2)) === UByte(2))
    assert(QUINT8.cast(UShort(2)) === UByte(2))
    assert(QUINT8.cast(2L) === UByte(2))
    assert(QUINT8.cast(-2.0) === UByte(254)) // TODO: Should this throw an error?
    assert(QUINT16.cast(-UByte(2)) === UShort(254)) // TODO: Should this throw an error?
    assert(QUINT16.cast(UShort(2)) === UShort(2))
    assert(QUINT16.cast(2L) === UShort(2))
    assert(QUINT16.cast(2.0) === UShort(2))
  }

  // TODO: Add 'InvalidCastException' checks.

  it must "not compile when invalid values are provided" in {
    assertDoesNotCompile("DataType.FLOAT32.cast(Array(1))")
    assertDoesNotCompile("DataType.FLOAT32.cast(1.asInstanceOf[Any])")
  }

  "'DataType.putElementInBuffer'" must "work correctly when provided values of supported types" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(12)
    INT32.putElementInBuffer(buffer, 0, 1)
    INT32.putElementInBuffer(buffer, 4, 16)
    INT32.putElementInBuffer(buffer, 8, 257)
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
    INT32.putElementInBuffer(buffer, 0, 1)
    INT32.putElementInBuffer(buffer, 4, 16)
    assertThrows[IndexOutOfBoundsException](INT32.putElementInBuffer(buffer, 8, 257))
  }

  "'DataType.getElementFromBuffer'" must "match the behavior of 'DataType.putElementInBuffer'" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(1024)
    // FLOAT16.putElementInBuffer(buffer, 0, 2f)
    FLOAT32.putElementInBuffer(buffer, 2, -4.23f)
    FLOAT64.putElementInBuffer(buffer, 6, 3.45)
    // BFLOAT16.putElementInBuffer(buffer, 14, -1.23f)
    INT8.putElementInBuffer(buffer, 16, 4.toByte)
    INT16.putElementInBuffer(buffer, 17, (-2).toShort)
    INT32.putElementInBuffer(buffer, 19, 54)
    INT64.putElementInBuffer(buffer, 23, -3416L)
    UINT8.putElementInBuffer(buffer, 31, UByte(34))
    UINT16.putElementInBuffer(buffer, 32, UShort(657))
    QINT8.putElementInBuffer(buffer, 34, (-4).toByte)
    QINT16.putElementInBuffer(buffer, 35, 32.toShort)
    QINT32.putElementInBuffer(buffer, 37, -548979)
    QUINT8.putElementInBuffer(buffer, 41, UByte(254))
    QUINT16.putElementInBuffer(buffer, 42, UShort(765))
    BOOLEAN.putElementInBuffer(buffer, 44, true)
    BOOLEAN.putElementInBuffer(buffer, 45, false)
    // TODO: Add checks for the string data type.
    // assert(FLOAT16.getElementFromBuffer(buffer, 0) === 2f)
    assert(FLOAT32.getElementFromBuffer(buffer, 2) === -4.23f)
    assert(FLOAT64.getElementFromBuffer(buffer, 6) === 3.45)
    // assert(BFLOAT16.getElementFromBuffer(buffer, 14) === -1.23f)
    assert(INT8.getElementFromBuffer(buffer, 16) === 4.toByte)
    assert(INT16.getElementFromBuffer(buffer, 17) === (-2).toShort)
    assert(INT32.getElementFromBuffer(buffer, 19) === 54)
    assert(INT64.getElementFromBuffer(buffer, 23) === -3416L)
    assert(UINT8.getElementFromBuffer(buffer, 31) === UByte(34))
    assert(UINT16.getElementFromBuffer(buffer, 32) === UShort(657))
    assert(QINT8.getElementFromBuffer(buffer, 34) === (-4).toByte)
    assert(QINT16.getElementFromBuffer(buffer, 35) === 32.toShort)
    assert(QINT32.getElementFromBuffer(buffer, 37) === -548979)
    assert(QUINT8.getElementFromBuffer(buffer, 41) === UByte(254))
    assert(QUINT16.getElementFromBuffer(buffer, 42) === UShort(765))
    assert(BOOLEAN.getElementFromBuffer(buffer, 44) === true)
    assert(BOOLEAN.getElementFromBuffer(buffer, 45) === false)
  }
}
