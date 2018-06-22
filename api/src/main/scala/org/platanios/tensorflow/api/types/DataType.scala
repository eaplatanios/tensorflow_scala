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

package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.jni.{Tensor => NativeTensor, TensorFlow => NativeLibrary}

import com.google.protobuf.ByteString
import org.tensorflow.framework.TensorProto
import spire.math.{UByte, UShort}

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

// TODO: Add min/max-value and "isSigned" information.
// TODO: Casts are unsafe (i.e., downcasting is allowed).

/** Represents the data type of the elements in a tensor.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataType {
  type ScalaType

  private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type]

  //region Data Type Properties

  /** Name of the data type (mainly useful for logging purposes). */
  val name: String

  /** Integer representing this data type in the `TF_DataType` enum of the TensorFlow C API. */
  private[api] val cValue: Int

  /** Size in bytes of each value with this data type. Returns `-1` if the size is not available. */
  val byteSize: Int

  /** Size in bytes of each value with this data type, as returned by the native TensorFlow library. Returns `None` if
    * the size is not available.
    *
    * Note that this value is currently not used anywhere within the TensorFlow Scala API.
    */
  private[api] lazy val nativeByteSize: Option[Int] = {
    val nativeLibrarySize = NativeLibrary.dataTypeSize(cValue)
    if (nativeLibrarySize == 0)
      None
    else
      Some(nativeLibrarySize)
  }

  private[api] val priority: Int

  /** ProtoBuf data type used in serialized representations of tensors. */
  def protoType: org.tensorflow.framework.DataType

  //endregion Data Type Properties

  //region Data Type Set Helper Methods

  /** Zero value for this data type. */
  def zero: ScalaType

  /** One value for this data type. */
  def one: ScalaType

  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
  def isFloatingPoint: Boolean = !isQuantized && DataType.floatingPointDataTypes.contains(this)

  /** Returns `true` if this data type represents a complex data types. */
  def isComplex: Boolean = DataType.complexDataTypes.contains(this)

  /** Returns `true` if this data type represents a non-quantized integer data type. */
  def isInteger: Boolean = !isQuantized && DataType.integerDataTypes.contains(this)

  /** Returns `true` if this data type represents a quantized data type. */
  def isQuantized: Boolean = DataType.quantizedDataTypes.contains(this)

  /** Returns `true` if this data type represents a non-quantized unsigned data type. */
  def isUnsigned: Boolean = !isQuantized && DataType.unsignedDataTypes.contains(this)

  /** Returns `true` if this data type represents a numeric data type. */
  def isNumeric: Boolean = DataType.numericDataTypes.contains(this)

  /** Returns `true` if this data type represents a boolean data type. */
  def isBoolean: Boolean = this == BOOLEAN

  //endregion Data Type Set Helper Methods

  /** Returns a data type that corresponds to this data type's real part. */
  def real: DataType = this match {
    case COMPLEX64 => FLOAT32
    case COMPLEX128 => FLOAT64
    case _ => this
  }

  /** Returns the smallest value that can be represented by this data type. */
  def min: ScalaType = throw new UnsupportedOperationException(s"Cannot determine max value for '$this' data type.")

  /** Returns the largest value that can be represented by this data type. */
  def max: ScalaType = throw new UnsupportedOperationException(s"Cannot determine max value for '$this' data type.")

  // TODO: !!! [TYPES] Make this safer.

  /** Casts the provided value to this data type.
    *
    * Note that this method allows downcasting.
    *
    * @param  value Value to cast.
    * @return Casted value.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  @inline def cast[R](value: R)(implicit ev: SupportedType.Aux[R, _]): ScalaType = evSupportedType.cast(value)

  /** Puts an element of this data type into the provided byte buffer.
    *
    * @param  buffer  Byte buffer in which to put the element.
    * @param  index   Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @param  element Element to put into the provided byte buffer.
    * @return Number of bytes written. For all data types with a known byte size (i.e., not equal to `-1`), the return
    *         value is equal to the byte size.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int

  /** Gets an element of this data type from the provided byte buffer.
    *
    * @param  buffer Byte buffer from which to get an element.
    * @param  index  Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @return Obtained element.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType

  private[api] def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: ScalaType): Unit

  override def toString: String = name

  override def equals(that: Any): Boolean = that match {
    case that: DataType => this.cValue == that.cValue
    case _ => false
  }

  override def hashCode: Int = cValue
}

/** Contains all supported data types along with some helper functions for dealing with them. */
object DataType {
  //region Data Type Sets

  /** Set of all floating-point data types. */
  val floatingPointDataTypes: Set[DataType] = {
    Set(FLOAT16, FLOAT32, FLOAT64, BFLOAT16)
  }

  /** Set of all complex data types. */
  val complexDataTypes: Set[DataType] = {
    Set(COMPLEX64, COMPLEX128)
  }

  /** Set of all integer data types. */
  val integerDataTypes: Set[DataType] = {
    Set(INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all quantized data types. */
  val quantizedDataTypes: Set[DataType] = {
    Set(BFLOAT16, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all unsigned data types. */
  val unsignedDataTypes: Set[DataType] = {
    Set(UINT8, UINT16, UINT32, UINT64, QUINT8, QUINT16)
  }

  /** Set of all numeric data types. */
  val numericDataTypes: Set[DataType] = {
    floatingPointDataTypes ++ complexDataTypes ++ integerDataTypes ++ quantizedDataTypes
  }

  //endregion Data Type Sets

  //region Helper Methods

  /** Returns the [[DataType]] of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    */
  @inline def dataTypeOf[T, D <: DataType](value: T)(implicit evSupported: SupportedType.Aux[T, D]): D = {
    evSupported.dataType
  }

  /** Returns the data type corresponding to the provided C value.
    *
    * By C value here we refer to an integer representing a data type in the `TF_DataType` enum of the TensorFlow C
    * API.
    *
    * @param  cValue C value.
    * @return Data type corresponding to the provided C value.
    * @throws IllegalArgumentException If an invalid C value is provided.
    */
  @throws[IllegalArgumentException]
  private[api] def fromCValue[D <: DataType](cValue: Int): D = {
    val dataType = cValue match {
      case BOOLEAN.cValue => BOOLEAN
      case STRING.cValue => STRING
      case FLOAT16.cValue => FLOAT16
      case FLOAT32.cValue => FLOAT32
      case FLOAT64.cValue => FLOAT64
      case BFLOAT16.cValue => BFLOAT16
      case COMPLEX64.cValue => COMPLEX64
      case COMPLEX128.cValue => COMPLEX128
      case INT8.cValue => INT8
      case INT16.cValue => INT16
      case INT32.cValue => INT32
      case INT64.cValue => INT64
      case UINT8.cValue => UINT8
      case UINT16.cValue => UINT16
      case UINT32.cValue => UINT32
      case UINT64.cValue => UINT64
      case QINT8.cValue => QINT8
      case QINT16.cValue => QINT16
      case QINT32.cValue => QINT32
      case QUINT8.cValue => QUINT8
      case QUINT16.cValue => QUINT16
      case RESOURCE.cValue => RESOURCE
      case VARIANT.cValue => VARIANT
      case value => throw new IllegalArgumentException(
        s"Data type C value '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
    }
    dataType.asInstanceOf[D]
  }

  /** Returns the data type corresponding to the provided name.
    *
    * @param  name Data type name.
    * @return Data type corresponding to the provided C value.
    * @throws IllegalArgumentException If an invalid data type name is provided.
    */
  @throws[IllegalArgumentException]
  private[api] def fromName[D <: DataType](name: String): D = {
    val dataType = name match {
      case "BOOLEAN" => BOOLEAN
      case "STRING" => STRING
      case "FLOAT16" => FLOAT16
      case "FLOAT32" => FLOAT32
      case "FLOAT64" => FLOAT64
      case "BFLOAT16" => BFLOAT16
      case "COMPLEX64" => COMPLEX64
      case "COMPLEX128" => COMPLEX128
      case "INT8" => INT8
      case "INT16" => INT16
      case "INT32" => INT32
      case "INT64" => INT64
      case "UINT8" => UINT8
      case "UINT16" => UINT16
      case "UINT32" => UINT32
      case "UINT64" => UINT64
      case "QINT8" => QINT8
      case "QINT16" => QINT16
      case "QINT32" => QINT32
      case "QUINT8" => QUINT8
      case "QUINT16" => QUINT16
      case "RESOURCE" => RESOURCE
      case "VARIANT" => VARIANT
      case value => throw new IllegalArgumentException(
        s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
    }
    dataType.asInstanceOf[D]
  }

  /** Returns the most precise data type out of the provided data types, based on their `priority` field.
    *
    * @param  dataTypes Data types out of which to pick the most precise.
    * @return Most precise data type in `dataTypes`.
    */
  def mostPrecise(dataTypes: DataType*): DataType = dataTypes.maxBy(_.priority)

  /** Returns the most precise data type out of the provided data types, based on their `priority` field.
    *
    * @param  dataTypes Data types out of which to pick the most precise.
    * @return Most precise data type in `dataTypes`.
    */
  def leastPrecise(dataTypes: DataType*): DataType = dataTypes.minBy(_.priority)

  //endregion Helper Methods

  private[types] trait API {
    @inline def dataTypeOf[T, D <: DataType](value: T)(implicit evSupportedType: SupportedType.Aux[T, D]): D = {
      DataType.dataTypeOf(value)
    }

    @throws[IllegalArgumentException]
    def dataType(cValue: Int): DataType = DataType.fromCValue(cValue)

    @throws[IllegalArgumentException]
    def dataType(name: String): DataType = DataType.fromName(name)

    def mostPreciseDataType(dataTypes: DataType*): DataType = DataType.mostPrecise(dataTypes: _*)
    def leastPreciseDataType(dataTypes: DataType*): DataType = DataType.leastPrecise(dataTypes: _*)
  }
}

sealed trait ReducibleDataType extends DataType
sealed trait NumericDataType extends ReducibleDataType
sealed trait NonQuantizedDataType extends NumericDataType
sealed trait MathDataType extends NonQuantizedDataType
sealed trait RealDataType extends MathDataType
sealed trait ComplexDataType extends MathDataType
sealed trait Int32OrInt64OrFloat16OrFloat32OrFloat64 extends RealDataType
sealed trait IntOrUInt extends RealDataType
sealed trait UInt8OrInt32OrInt64 extends IntOrUInt with Int32OrInt64OrFloat16OrFloat32OrFloat64
sealed trait Int32OrInt64 extends UInt8OrInt32OrInt64
sealed trait DecimalDataType extends RealDataType
sealed trait BFloat16OrFloat32OrFloat64 extends DecimalDataType
sealed trait Float16OrFloat32OrFloat64 extends DecimalDataType with Int32OrInt64OrFloat16OrFloat32OrFloat64 with BFloat16OrFloat32OrFloat64
sealed trait Float32OrFloat64 extends Float16OrFloat32OrFloat64
sealed trait Int32OrInt64OrFloat32OrFloat64 extends Float32OrFloat64 with Int32OrInt64
sealed trait QuantizedDataType extends NumericDataType

object STRING extends ReducibleDataType {
  override type ScalaType = String

  override private[api] implicit val evSupportedType: SupportedType.Aux[ScalaType, this.type] = {
    SupportedType.stringIsSupported
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
    SupportedType.booleanIsSupported
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
    SupportedType.floatIsSupported
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
    SupportedType.doubleIsSupported
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
    SupportedType.byteIsSupported
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
    SupportedType.shortIsSupported
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
    SupportedType.intIsSupported
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
    SupportedType.longIsSupported
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
    SupportedType.uByteIsSupported
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
    SupportedType.uShortIsSupported
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
