/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.types.SupportedType._
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import com.google.protobuf.ByteString
import org.tensorflow.framework.TensorProto

import spire.math.{UByte, UShort}

import java.nio.ByteBuffer
import java.nio.charset.Charset

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  private[api] trait API extends DataType.API

  val STRING: DataType.Aux[String] = new DataType.Aux[String] {
    override implicit val supportedType: SupportedType[String] = stringIsSupportedType

    override val name    : String = "STRING"
    override val cValue  : Int    = 7
    override val byteSize: Int    = -1
    override val priority: Int    = 1000

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_STRING

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: String): Int = {
      val stringBytes = element.getBytes(Charset.forName("ISO-8859-1"))
      NativeTensor.setStringBytes(stringBytes, buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): String = {
      val stringBytes = NativeTensor.getStringBytes(buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
      new String(stringBytes, Charset.forName("ISO-8859-1"))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: String): Unit = {
      tensorProtoBuilder.addStringVal(ByteString.copyFromUtf8(value))
    }
  }

  val BOOLEAN: DataType.Aux[Boolean] = new DataType.Aux[Boolean] {
    override implicit val supportedType: SupportedType[Boolean] = booleanIsSupportedType

    override val name    : String = "BOOLEAN"
    override val cValue  : Int    = 10
    override val byteSize: Int    = 1
    override val priority: Int    = 0

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_BOOL

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Boolean): Int = {
      buffer.put(index, if (element) 1 else 0)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Boolean = {
      buffer.get(index) == 1
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Boolean): Unit = {
      tensorProtoBuilder.addBoolVal(value)
    }
  }

  // TODO: Fix/complete the following implementations for FLOAT16, BFLOAT16, COMPLEX64, and COMPLEX128.

  val FLOAT16: DataType.Aux[Float] = new DataType.Aux[Float] {
    override implicit val supportedType: SupportedType[Float] = floatIsSupportedType

    override val name    : String = "FLOAT16"
    override val cValue  : Int    = 19
    override val byteSize: Int    = 2
    override val priority: Int    = -1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_HALF

    override def min: ScalaType = -65504f
    override def max: ScalaType = 65504f

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Float): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Float = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Float): Unit = {
      ???
    }
  }

  val FLOAT32: DataType.Aux[Float] = new DataType.Aux[Float] {
    override implicit val supportedType: SupportedType[Float] = floatIsSupportedType

    override val name    : String = "FLOAT32"
    override val cValue  : Int    = 1
    override val byteSize: Int    = 4
    override val priority: Int    = 220

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_FLOAT

    override def min: ScalaType = Float.MinValue
    override def max: ScalaType = Float.MaxValue

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Float): Int = {
      buffer.putFloat(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Float = {
      buffer.getFloat(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Float): Unit = {
      tensorProtoBuilder.addFloatVal(value)
    }
  }

  val FLOAT64: DataType.Aux[Double] = new DataType.Aux[Double] {
    override implicit val supportedType: SupportedType[Double] = doubleIsSupportedType

    override val name    : String = "FLOAT64"
    override val cValue  : Int    = 2
    override val byteSize: Int    = 8
    override val priority: Int    = 230

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_DOUBLE

    override def min: ScalaType = Double.MinValue
    override def max: ScalaType = Double.MaxValue

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Double): Int = {
      buffer.putDouble(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Double = {
      buffer.getDouble(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Double): Unit = {
      tensorProtoBuilder.addDoubleVal(value)
    }
  }

  val BFLOAT16: DataType.Aux[Float] = new DataType.Aux[Float] {
    override implicit val supportedType: SupportedType[Float] = floatIsSupportedType

    override val name    : String = "BFLOAT16"
    override val cValue  : Int    = 14
    override val byteSize: Int    = 2
    override val priority: Int    = -1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_BFLOAT16

    override def min: ScalaType = ???
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Float): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Float = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Float): Unit = {
      ???
    }
  }

  val COMPLEX64: DataType.Aux[Double] = new DataType.Aux[Double] {
    override implicit val supportedType: SupportedType[Double] = doubleIsSupportedType

    override val name    : String = "COMPLEX64"
    override val cValue  : Int    = 8
    override val byteSize: Int    = 8
    override val priority: Int    = -1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_COMPLEX64

    override def min: ScalaType = ???
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Double): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Double = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Double): Unit = {
      ???
    }
  }

  val COMPLEX128: DataType.Aux[Double] = new DataType.Aux[Double] {
    override implicit val supportedType: SupportedType[Double] = doubleIsSupportedType

    override val name    : String = "COMPLEX128"
    override val cValue  : Int    = 18
    override val byteSize: Int    = 16
    override val priority: Int    = -1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_COMPLEX128

    override def min: ScalaType = ???
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Double): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Double = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Double): Unit = {
      ???
    }
  }

  val INT8: DataType.Aux[Byte] = new DataType.Aux[Byte] {
    override implicit val supportedType: SupportedType[Byte] = byteIsSupportedType

    override val name    : String = "INT8"
    override val cValue  : Int    = 6
    override val byteSize: Int    = 1
    override val priority: Int    = 40

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT8

    override def min: ScalaType = (-128).toByte
    override def max: ScalaType = 127.toByte

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Byte): Int = {
      buffer.put(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Byte = {
      buffer.get(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Byte): Unit = {
      tensorProtoBuilder.addIntVal(value)
    }
  }

  val INT16: DataType.Aux[Short] = new DataType.Aux[Short] {
    override implicit val supportedType: SupportedType[Short] = shortIsSupportedType

    override val name    : String = "INT16"
    override val cValue  : Int    = 5
    override val byteSize: Int    = 2
    override val priority: Int    = 80

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT16

    override def min: ScalaType = (-32768).toShort
    override def max: ScalaType = 32767.toShort

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Short): Int = {
      buffer.putShort(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Short = {
      buffer.getShort(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Short): Unit = {
      tensorProtoBuilder.addIntVal(value)
    }
  }

  val INT32: DataType.Aux[Int] = new DataType.Aux[Int] {
    override implicit val supportedType: SupportedType[Int] = intIsSupportedType

    override val name    : String = "INT32"
    override val cValue  : Int    = 3
    override val byteSize: Int    = 4
    override val priority: Int    = 100

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT32

    override def min: ScalaType = -2147483648
    override def max: ScalaType = 2147483647

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Int): Int = {
      buffer.putInt(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Int = {
      buffer.getInt(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Int): Unit = {
      tensorProtoBuilder.addIntVal(value)
    }
  }

  val INT64: DataType.Aux[Long] = new DataType.Aux[Long] {
    override implicit val supportedType: SupportedType[Long] = longIsSupportedType

    override val name    : String = "INT64"
    override val cValue  : Int    = 9
    override val byteSize: Int    = 8
    override val priority: Int    = 110

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT64

    override def min: ScalaType = -9223372036854775808L
    override def max: ScalaType = 9223372036854775807L

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putLong(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getLong(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      tensorProtoBuilder.addInt64Val(value)
    }
  }

  val UINT8: DataType.Aux[UByte] = new DataType.Aux[UByte] {
    override implicit val supportedType: SupportedType[UByte] = uByteIsSupportedType

    override val name    : String = "UINT8"
    override val cValue  : Int    = 4
    override val byteSize: Int    = 1
    override val priority: Int    = 20

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_UINT8

    override def min: ScalaType = UByte(0)
    override def max: ScalaType = UByte(255)

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UByte): Int = {
      buffer.put(index, element.toByte)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UByte = {
      UByte(buffer.get(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UByte): Unit = {
      tensorProtoBuilder.addIntVal(value.toInt)
    }
  }

  val UINT16: DataType.Aux[UShort] = new DataType.Aux[UShort] {
    override implicit val supportedType: SupportedType[UShort] = uShortIsSupportedType

    override val name    : String = "UINT16"
    override val cValue  : Int    = 17
    override val byteSize: Int    = 2
    override val priority: Int    = 60

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_UINT16

    override def min: ScalaType = UShort(0)
    override def max: ScalaType = UShort(65535)

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UShort): Int = {
      buffer.putChar(index, element.toChar)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UShort = {
      UShort(buffer.getChar(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UShort): Unit = {
      tensorProtoBuilder.addIntVal(value.toInt)
    }
  }

  val UINT32: DataType.Aux[Long] = new DataType.Aux[Long] {
    override implicit val supportedType: SupportedType[Long] = longIsSupportedType

    override val name    : String = "UINT32"
    override val cValue  : Int    = 22
    override val byteSize: Int    = 4
    override val priority: Int    = 85

    override def protoType: org.tensorflow.framework.DataType = ???

    override def min: ScalaType = 0L
    override def max: ScalaType = 9223372036854775807L

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putInt(index, element.toInt)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getInt(index).toLong
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }

  val UINT64: DataType.Aux[Long] = new DataType.Aux[Long] {
    override implicit val supportedType: SupportedType[Long] = longIsSupportedType

    override val name    : String = "UINT64"
    override val cValue  : Int    = 23
    override val byteSize: Int    = 8
    override val priority: Int    = 105

    override def protoType: org.tensorflow.framework.DataType = ???

    override def min: ScalaType = 0L
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }

  val QINT8: DataType.Aux[Byte] = new DataType.Aux[Byte] {
    override implicit val supportedType: SupportedType[Byte] = byteIsSupportedType

    override val name    : String = "QINT8"
    override val cValue  : Int    = 11
    override val byteSize: Int    = 1
    override val priority: Int    = 30

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QINT8

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Byte): Int = {
      buffer.put(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Byte = {
      buffer.get(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Byte): Unit = {
      ???
    }
  }

  val QINT16: DataType.Aux[Short] = new DataType.Aux[Short] {
    override implicit val supportedType: SupportedType[Short] = shortIsSupportedType

    override val name    : String = "QINT16"
    override val cValue  : Int    = 15
    override val byteSize: Int    = 2
    override val priority: Int    = 70

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QINT16

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Short): Int = {
      buffer.putShort(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Short = {
      buffer.getShort(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Short): Unit = {
      ???
    }
  }

  val QINT32: DataType.Aux[Int] = new DataType.Aux[Int] {
    override implicit val supportedType: SupportedType[Int] = intIsSupportedType

    override val name    : String = "QINT32"
    override val cValue  : Int    = 13
    override val byteSize: Int    = 4
    override val priority: Int    = 90

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QINT32

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Int): Int = {
      buffer.putInt(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Int = {
      buffer.getInt(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Int): Unit = {
      ???
    }
  }

  val QUINT8: DataType.Aux[UByte] = new DataType.Aux[UByte] {
    override implicit val supportedType: SupportedType[UByte] = uByteIsSupportedType

    override val name    : String = "QUINT8"
    override val cValue  : Int    = 12
    override val byteSize: Int    = 1
    override val priority: Int    = 10

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QUINT8

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UByte): Int = {
      buffer.put(index, element.toByte)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UByte = {
      UByte(buffer.get(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UByte): Unit = {
      ???
    }
  }

  val QUINT16: DataType.Aux[UShort] = new DataType.Aux[UShort] {
    override implicit val supportedType: SupportedType[UShort] = uShortIsSupportedType

    override val name    : String = "QUINT16"
    override val cValue  : Int    = 16
    override val byteSize: Int    = 2
    override val priority: Int    = 50

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QUINT16

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UShort): Int = {
      buffer.putChar(index, element.toChar)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UShort = {
      UShort(buffer.getChar(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UShort): Unit = {
      ???
    }
  }

  val RESOURCE: DataType.Aux[Long] = new DataType.Aux[Long] {
    override implicit val supportedType: SupportedType[Long] = longIsSupportedType

    override val name    : String = "RESOURCE"
    override val cValue  : Int    = 20
    override val byteSize: Int    = -1
    override val priority: Int    = -1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_RESOURCE

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putLong(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getLong(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }

  val VARIANT: DataType.Aux[Long] = new DataType.Aux[Long] {
    override implicit val supportedType: SupportedType[Long] = longIsSupportedType

    override val name    : String = "VARIANT"
    override val cValue  : Int    = 21
    override val byteSize: Int    = -1
    override val priority: Int    = -1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_VARIANT

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putLong(index, element)
      byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getLong(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }
}
