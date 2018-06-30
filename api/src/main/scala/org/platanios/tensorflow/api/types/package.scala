/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import com.google.protobuf.ByteString
import org.tensorflow.framework.TensorProto
import spire.math.{UByte, UShort}

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

/**
  * @author Emmanouil Antonios Platanios
  */
package object types {
  private[api] trait API extends DataType.API

  type STRING = types.STRING.type
  type BOOLEAN = types.BOOLEAN.type
  type FLOAT16 = types.FLOAT16.type
  type FLOAT32 = types.FLOAT32.type
  type FLOAT64 = types.FLOAT64.type
  type BFLOAT16 = types.BFLOAT16.type
  type COMPLEX64 = types.COMPLEX64.type
  type COMPLEX128 = types.COMPLEX128.type
  type INT8 = types.INT8.type
  type INT16 = types.INT16.type
  type INT32 = types.INT32.type
  type INT64 = types.INT64.type
  type UINT8 = types.UINT8.type
  type UINT16 = types.UINT16.type
  type UINT32 = types.UINT32.type
  type UINT64 = types.UINT64.type
  type QINT8 = types.QINT8.type
  type QINT16 = types.QINT16.type
  type QINT32 = types.QINT32.type
  type QUINT8 = types.QUINT8.type
  type QUINT16 = types.QUINT16.type
  type RESOURCE = types.RESOURCE.type
  type VARIANT = types.VARIANT.type

  trait ReducibleDataType extends DataType
  trait NumericDataType extends ReducibleDataType
  trait NonQuantizedDataType extends NumericDataType
  trait MathDataType extends NonQuantizedDataType
  trait RealDataType extends MathDataType
  trait ComplexDataType extends MathDataType
  trait Int32OrInt64OrFloat16OrFloat32OrFloat64 extends RealDataType
  trait IntOrUInt extends RealDataType
  trait UInt8OrInt32OrInt64 extends IntOrUInt with Int32OrInt64OrFloat16OrFloat32OrFloat64
  trait Int32OrInt64 extends UInt8OrInt32OrInt64
  trait DecimalDataType extends RealDataType
  trait BFloat16OrFloat32OrFloat64 extends DecimalDataType
  trait Float16OrFloat32OrFloat64 extends DecimalDataType with Int32OrInt64OrFloat16OrFloat32OrFloat64 with BFloat16OrFloat32OrFloat64
  trait Float32OrFloat64 extends Float16OrFloat32OrFloat64
  trait Int32OrInt64OrFloat32OrFloat64 extends Float32OrFloat64 with Int32OrInt64
  trait QuantizedDataType extends NumericDataType

  object STRING extends ReducibleDataType {
    override type ScalaType = String

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.stringIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "STRING"
    override val cValue  : Int    = 7
    override val byteSize: Int    = -1
    override val priority: Int    = 1000

    override def zero: String = ""
    override def one: String = ???

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_STRING

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
      val stringBytes = element.getBytes(StandardCharsets.ISO_8859_1)
      NativeTensor.setStringBytes(stringBytes, buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): String = {
      val stringBytes = NativeTensor.getStringBytes(buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
      new String(stringBytes, StandardCharsets.ISO_8859_1)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: String): Unit = {
      tensorProtoBuilder.addStringVal(ByteString.copyFrom(value.getBytes))
    }
  }

  object BOOLEAN extends ReducibleDataType {
    override type ScalaType = Boolean

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.booleanIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "BOOLEAN"
    override val cValue  : Int    = 10
    override val byteSize: Int    = 1
    override val priority: Int    = 0

    override def zero: Boolean = false
    override def one: Boolean = true

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_BOOL

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object FLOAT16 extends Float16OrFloat32OrFloat64 {
    override type ScalaType = Float

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "FLOAT16"
    override val cValue  : Int    = 19
    override val byteSize: Int    = 2
    override val priority: Int    = -1

    override def zero: Float = 0.0f
    override def one: Float = 1.0f

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_HALF

    override def min: ScalaType = -65504f
    override def max: ScalaType = 65504f

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Float = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Float): Unit = {
      ???
    }
  }

  object FLOAT32 extends Float32OrFloat64 {
    override type ScalaType = Float

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.floatIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "FLOAT32"
    override val cValue  : Int    = 1
    override val byteSize: Int    = 4
    override val priority: Int    = 220

    override def zero: Float = 0.0f
    override def one: Float = 1.0f

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_FLOAT

    override def min: ScalaType = Float.MinValue
    override def max: ScalaType = Float.MaxValue

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object FLOAT64 extends Float32OrFloat64 {
    override type ScalaType = Double

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.doubleIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "FLOAT64"
    override val cValue  : Int    = 2
    override val byteSize: Int    = 8
    override val priority: Int    = 230

    override def zero: Double = 0.0
    override def one: Double = 1.0

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_DOUBLE

    override def min: ScalaType = Double.MinValue
    override def max: ScalaType = Double.MaxValue

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object BFLOAT16 extends BFloat16OrFloat32OrFloat64 {
    override type ScalaType = Float

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "BFLOAT16"
    override val cValue  : Int    = 14
    override val byteSize: Int    = 2
    override val priority: Int    = -1

    override def zero: Float = 0.0f
    override def one: Float = 1.0f

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_BFLOAT16

    override def min: ScalaType = ???
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Float = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Float): Unit = {
      ???
    }
  }

  object COMPLEX64 extends ComplexDataType {
    override type ScalaType = Double

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "COMPLEX64"
    override val cValue  : Int    = 8
    override val byteSize: Int    = 8
    override val priority: Int    = -1

    override def zero: Double = ???
    override def one: Double = ???

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_COMPLEX64

    override def min: ScalaType = ???
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Double = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Double): Unit = {
      ???
    }
  }

  object COMPLEX128 extends ComplexDataType {
    override type ScalaType = Double

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "COMPLEX128"
    override val cValue  : Int    = 18
    override val byteSize: Int    = 16
    override val priority: Int    = -1

    override def zero: Double = ???
    override def one: Double = ???

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_COMPLEX128

    override def min: ScalaType = ???
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Double = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Double): Unit = {
      ???
    }
  }

  object INT8 extends IntOrUInt {
    override type ScalaType = Byte

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.byteIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "INT8"
    override val cValue  : Int    = 6
    override val byteSize: Int    = 1
    override val priority: Int    = 40

    override def zero: Byte = 0
    override def one: Byte = 1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT8

    override def min: ScalaType = (-128).toByte
    override def max: ScalaType = 127.toByte

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object INT16 extends IntOrUInt {
    override type ScalaType = Short

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.shortIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "INT16"
    override val cValue  : Int    = 5
    override val byteSize: Int    = 2
    override val priority: Int    = 80

    override def zero: Short = 0
    override def one: Short = 1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT16

    override def min: ScalaType = (-32768).toShort
    override def max: ScalaType = 32767.toShort

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object INT32 extends Int32OrInt64 {
    override type ScalaType = Int

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.intIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "INT32"
    override val cValue  : Int    = 3
    override val byteSize: Int    = 4
    override val priority: Int    = 100

    override def zero: Int = 0
    override def one: Int = 1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT32

    override def min: ScalaType = -2147483648
    override def max: ScalaType = 2147483647

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object INT64 extends Int32OrInt64 {
    override type ScalaType = Long

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.longIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "INT64"
    override val cValue  : Int    = 9
    override val byteSize: Int    = 8
    override val priority: Int    = 110

    override def zero: Long = 0L
    override def one: Long = 1L

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_INT64

    override def min: ScalaType = -9223372036854775808L
    override def max: ScalaType = 9223372036854775807L

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object UINT8 extends UInt8OrInt32OrInt64 {
    override type ScalaType = UByte

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.uByteIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "UINT8"
    override val cValue  : Int    = 4
    override val byteSize: Int    = 1
    override val priority: Int    = 20

    override def zero: UByte = UByte(0)
    override def one: UByte = UByte(1)

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_UINT8

    override def min: ScalaType = UByte(0)
    override def max: ScalaType = UByte(255)

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object UINT16 extends IntOrUInt {
    override type ScalaType = UShort

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
      SupportedType.uShortIsSupported.asInstanceOf[SupportedType.Aux[ScalaType, this.type]]
    }

    override val name    : String = "UINT16"
    override val cValue  : Int    = 17
    override val byteSize: Int    = 2
    override val priority: Int    = 60

    override def zero: UShort = UShort(0)
    override def one: UShort = UShort(1)

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_UINT16

    override def min: ScalaType = UShort(0)
    override def max: ScalaType = UShort(65535)

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object UINT32 extends IntOrUInt {
    override type ScalaType = Long

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "UINT32"
    override val cValue  : Int    = 22
    override val byteSize: Int    = 4
    override val priority: Int    = 85

    override def zero: Long = 0L
    override def one: Long = 1L

    override def protoType: org.tensorflow.framework.DataType = ???

    override def min: ScalaType = 0L
    override def max: ScalaType = 9223372036854775807L

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object UINT64 extends IntOrUInt {
    override type ScalaType = Long

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "UINT64"
    override val cValue  : Int    = 23
    override val byteSize: Int    = 8
    override val priority: Int    = 105

    override def zero: Long = ???
    override def one: Long = ???

    override def protoType: org.tensorflow.framework.DataType = ???

    override def min: ScalaType = 0L
    override def max: ScalaType = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }

  object QINT8 extends QuantizedDataType {
    override type ScalaType = Byte

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "QINT8"
    override val cValue  : Int    = 11
    override val byteSize: Int    = 1
    override val priority: Int    = 30

    override def zero: Byte = 0
    override def one: Byte = 1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QINT8

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object QINT16 extends QuantizedDataType {
    override type ScalaType = Short

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "QINT16"
    override val cValue  : Int    = 15
    override val byteSize: Int    = 2
    override val priority: Int    = 70

    override def zero: Short = 0
    override def one: Short = 1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QINT16

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object QINT32 extends QuantizedDataType {
    override type ScalaType = Int

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "QINT32"
    override val cValue  : Int    = 13
    override val byteSize: Int    = 4
    override val priority: Int    = 90

    override def zero: Int = 0
    override def one: Int = 1

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QINT32

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object QUINT8 extends QuantizedDataType {
    override type ScalaType = UByte

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "QUINT8"
    override val cValue  : Int    = 12
    override val byteSize: Int    = 1
    override val priority: Int    = 10

    override def zero: UByte = UByte(0)
    override def one: UByte = UByte(1)

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QUINT8

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object QUINT16 extends QuantizedDataType {
    override type ScalaType = UShort

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "QUINT16"
    override val cValue  : Int    = 16
    override val byteSize: Int    = 2
    override val priority: Int    = 50

    override def zero: UShort = UShort(0)
    override def one: UShort = UShort(1)

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_QUINT16

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object RESOURCE extends DataType {
    override type ScalaType = Long

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "RESOURCE"
    override val cValue  : Int    = 20
    override val byteSize: Int    = -1
    override val priority: Int    = -1

    override def zero: Long = ???
    override def one: Long = ???

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_RESOURCE

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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

  object VARIANT extends DataType {
    override type ScalaType = Long

    override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = null

    override val name    : String = "VARIANT"
    override val cValue  : Int    = 21
    override val byteSize: Int    = -1
    override val priority: Int    = -1

    override def zero: Long = ???
    override def one: Long = ???

    override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_VARIANT

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
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
