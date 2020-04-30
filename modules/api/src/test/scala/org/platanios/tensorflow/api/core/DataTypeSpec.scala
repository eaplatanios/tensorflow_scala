/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.DataType

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.ByteBuffer

/**
  * @author Emmanouil Antonios Platanios
  */
class DataTypeSpec extends AnyFlatSpec with Matchers {
  "'DataType.fromCValue'" must "work correctly when valid C values are provided" in {
    assert(DataType.fromCValue(1) == FLOAT32)
    assert(DataType.fromCValue(2) == FLOAT64)
    assert(DataType.fromCValue(3) == INT32)
    assert(DataType.fromCValue(4) == UINT8)
    assert(DataType.fromCValue(5) == INT16)
    assert(DataType.fromCValue(6) == INT8)
    assert(DataType.fromCValue(7) == STRING)
    assert(DataType.fromCValue(8) == COMPLEX64)
    assert(DataType.fromCValue(9) == INT64)
    assert(DataType.fromCValue(10) == BOOLEAN)
    assert(DataType.fromCValue(11) == QINT8)
    assert(DataType.fromCValue(12) == QUINT8)
    assert(DataType.fromCValue(13) == QINT32)
    assert(DataType.fromCValue(14) == BFLOAT16)
    assert(DataType.fromCValue(15) == QINT16)
    assert(DataType.fromCValue(16) == QUINT16)
    assert(DataType.fromCValue(17) == UINT16)
    assert(DataType.fromCValue(18) == COMPLEX128)
    assert(DataType.fromCValue(19) == FLOAT16)
    assert(DataType.fromCValue(20) == RESOURCE)
    assert(DataType.fromCValue(21) == VARIANT)
  }

  it must "throw an 'IllegalArgumentException' when invalid C values are provided" in {
    assertThrows[IllegalArgumentException](DataType.fromCValue(-10))
    assertThrows[IllegalArgumentException](DataType.fromCValue(-1))
    assertThrows[IllegalArgumentException](DataType.fromCValue(0))
    assertThrows[IllegalArgumentException](DataType.fromCValue(25))
    assertThrows[IllegalArgumentException](DataType.fromCValue(54))
    assertThrows[IllegalArgumentException](DataType.fromCValue(167))
  }

  "'DataType.byteSize'" must "give the correct result" in {
    assert(STRING.byteSize === None)
    assert(BOOLEAN.byteSize === Some(1))
    assert(FLOAT16.byteSize === Some(2))
    assert(FLOAT32.byteSize === Some(4))
    assert(FLOAT64.byteSize === Some(8))
    assert(BFLOAT16.byteSize === Some(2))
    assert(COMPLEX64.byteSize === Some(8))
    assert(COMPLEX128.byteSize === Some(16))
    assert(INT8.byteSize === Some(1))
    assert(INT16.byteSize === Some(2))
    assert(INT32.byteSize === Some(4))
    assert(INT64.byteSize === Some(8))
    assert(UINT8.byteSize === Some(1))
    assert(UINT16.byteSize === Some(2))
    assert(QINT8.byteSize === Some(1))
    assert(QINT16.byteSize === Some(2))
    assert(QINT32.byteSize === Some(4))
    assert(QUINT8.byteSize === Some(1))
    assert(QUINT16.byteSize === Some(2))
    assert(RESOURCE.byteSize === Some(1))
    assert(VARIANT.byteSize === Some(1))
  }

  // TODO: Add checks for data type priorities.

  "'DataType.isBoolean'" must "always work correctly" in {
    assert(BOOLEAN.isBoolean === true)
    assert(STRING.isBoolean === false)
    assert(FLOAT16.isBoolean === false)
    assert(FLOAT32.isBoolean === false)
    assert(FLOAT64.isBoolean === false)
    assert(BFLOAT16.isBoolean === false)
    assert(COMPLEX64.isBoolean === false)
    assert(COMPLEX128.isBoolean === false)
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
    assert(VARIANT.isBoolean === false)
  }

  "'DataType.isFloatingPoint'" must "always work correctly" in {
    assert(BOOLEAN.isFloatingPoint === false)
    assert(STRING.isFloatingPoint === false)
    assert(FLOAT16.isFloatingPoint === true)
    assert(FLOAT32.isFloatingPoint === true)
    assert(FLOAT64.isFloatingPoint === true)
    assert(BFLOAT16.isFloatingPoint === false)
    assert(COMPLEX64.isFloatingPoint === false)
    assert(COMPLEX128.isFloatingPoint === false)
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
    assert(VARIANT.isFloatingPoint === false)
  }

  "'DataType.isInteger'" must "always work correctly" in {
    assert(BOOLEAN.isInteger === false)
    assert(STRING.isInteger === false)
    assert(FLOAT16.isInteger === false)
    assert(FLOAT32.isInteger === false)
    assert(FLOAT64.isInteger === false)
    assert(BFLOAT16.isInteger === false)
    assert(COMPLEX64.isInteger === false)
    assert(COMPLEX128.isInteger === false)
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
    assert(FLOAT16.isComplex === false)
    assert(FLOAT32.isComplex === false)
    assert(FLOAT64.isComplex === false)
    assert(BFLOAT16.isComplex === false)
    assert(COMPLEX64.isComplex === true)
    assert(COMPLEX128.isComplex === true)
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
    assert(VARIANT.isComplex === false)
  }

  "'DataType.isQuantized'" must "always work correctly" in {
    assert(BOOLEAN.isQuantized === false)
    assert(STRING.isQuantized === false)
    assert(FLOAT16.isQuantized === false)
    assert(FLOAT32.isQuantized === false)
    assert(FLOAT64.isQuantized === false)
    assert(BFLOAT16.isQuantized === true)
    assert(COMPLEX64.isQuantized === false)
    assert(COMPLEX128.isQuantized === false)
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
    assert(VARIANT.isQuantized === false)
  }

  "'DataType.isUnsigned'" must "always work correctly" in {
    assert(BOOLEAN.isUnsigned === false)
    assert(STRING.isUnsigned === false)
    assert(FLOAT16.isUnsigned === false)
    assert(FLOAT32.isUnsigned === false)
    assert(FLOAT64.isUnsigned === false)
    assert(BFLOAT16.isUnsigned === false)
    assert(COMPLEX64.isUnsigned === false)
    assert(COMPLEX128.isUnsigned === false)
    assert(INT8.isUnsigned === false)
    assert(INT16.isUnsigned === false)
    assert(INT32.isUnsigned === false)
    assert(INT64.isUnsigned === false)
    assert(UINT8.isUnsigned === true)
    assert(UINT16.isUnsigned === true)
    assert(QINT8.isUnsigned === false)
    assert(QINT16.isUnsigned === false)
    assert(QINT32.isUnsigned === false)
    assert(QUINT8.isUnsigned === true)
    assert(QUINT16.isUnsigned === true)
    assert(RESOURCE.isUnsigned === false)
    assert(VARIANT.isUnsigned === false)
  }

  "'DataType.equals'" must "always work correctly" in {
    assert(FLOAT32 === FLOAT32)
    assert(FLOAT32 !== FLOAT64)
    assert(INT8 !== UINT8)
    assert(INT8 !== QINT8)
  }

  "'DataType.putElementInBuffer'" must "work correctly when provided values of supported types" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(12)
    DataType.putElementInBuffer[Int](buffer, 0, 1)
    DataType.putElementInBuffer[Int](buffer, 4, 16)
    DataType.putElementInBuffer[Int](buffer, 8, 257)
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
    DataType.putElementInBuffer[Int](buffer, 0, 1)
    DataType.putElementInBuffer[Int](buffer, 4, 16)
    assertThrows[IndexOutOfBoundsException](DataType.putElementInBuffer[Int](buffer, 8, 257))
  }

  "'DataType.getElementFromBuffer'" must "match the behavior of 'DataType.putElementInBuffer'" in {
    val buffer: ByteBuffer = ByteBuffer.allocateDirect(1024)
    // DataType.putElementInBuffer[Half](buffer, 0, 2f)
    DataType.putElementInBuffer[Float](buffer, 2, -4.23f)
    DataType.putElementInBuffer[Double](buffer, 6, 3.45)
    // DataType.putElementInBuffer[TruncatedHalf](buffer, 14, -1.23f)
    DataType.putElementInBuffer[Byte](buffer, 16, 4.toByte)
    DataType.putElementInBuffer[Short](buffer, 17, (-2).toShort)
    DataType.putElementInBuffer[Int](buffer, 19, 54)
    DataType.putElementInBuffer[Long](buffer, 23, -3416L)
    //    DataType.putElementInBuffer[UByte](buffer, 31, UByte(34))
    //    DataType.putElementInBuffer[UShort](buffer, 32, UShort(657))
    //    DataType.putElementInBuffer[QByte](buffer, 34, (-4).toByte)
    //    DataType.putElementInBuffer[QShort](buffer, 35, 32.toShort)
    //    DataType.putElementInBuffer[QInt](buffer, 37, -548979)
    //    DataType.putElementInBuffer[QUByte](buffer, 41, UByte(254))
    //    DataType.putElementInBuffer[QUShort](buffer, 42, UShort(765))
    DataType.putElementInBuffer[Boolean](buffer, 44, true)
    DataType.putElementInBuffer[Boolean](buffer, 45, false)
    // TODO: Add checks for the string data type.
    // assert(DataType.getElementFromBuffer[Half](buffer, 0) === 2f)
    assert(DataType.getElementFromBuffer[Float](buffer, 2) === -4.23f)
    assert(DataType.getElementFromBuffer[Double](buffer, 6) === 3.45)
    // assert(DataType.getElementFromBuffer[TruncatedHalf](buffer, 14) === -1.23f)
    assert(DataType.getElementFromBuffer[Byte](buffer, 16) === 4.toByte)
    assert(DataType.getElementFromBuffer[Short](buffer, 17) === (-2).toShort)
    assert(DataType.getElementFromBuffer[Int](buffer, 19) === 54)
    assert(DataType.getElementFromBuffer[Long](buffer, 23) === -3416L)
    //    assert(DataType.getElementFromBuffer[UByte](buffer, 31) === UByte(34))
    //    assert(DataType.getElementFromBuffer[UShort](buffer, 32) === UShort(657))
    //    assert(DataType.getElementFromBuffer[QByte](buffer, 34) === (-4).toByte)
    //    assert(DataType.getElementFromBuffer[QShort](buffer, 35) === 32.toShort)
    //    assert(DataType.getElementFromBuffer[QInt](buffer, 37) === -548979)
    //    assert(DataType.getElementFromBuffer[QUByte](buffer, 41) === UByte(254))
    //    assert(DataType.getElementFromBuffer[QUShort](buffer, 42) === UShort(765))
    assert(DataType.getElementFromBuffer[Boolean](buffer, 44) === true)
    assert(DataType.getElementFromBuffer[Boolean](buffer, 45) === false)
  }
}
