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

package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.types.SupportedType._
import org.platanios.tensorflow.jni.{Tensor => NativeTensor, TensorFlow => NativeLibrary}

import java.nio.ByteBuffer
import java.nio.charset.Charset

import spire.math.{UByte, UShort}

// TODO: Add min/max-value and "isSigned" information.
// TODO: Casts are unsafe (i.e., downcasting is allowed).

/** Represents the data type of the elements in a tensor.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataType {
  type ScalaType
  implicit val supportedType: SupportedType[ScalaType]

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

  /** Casts the provided value to this data type.
    *
    * Note that this method allows downcasting.
    *
    * @param  value Value to cast.
    * @return Casted value.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  @inline def cast[T](value: T)(implicit evidence: SupportedType[T]): ScalaType = value.cast(this)

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

  override def toString: String = name

  override def equals(that: Any): Boolean = that match {
    case that: DataType => this.cValue == that.cValue
    case _ => false
  }

  override def hashCode: Int = cValue
}

/** Contains all supported data types along with some helper functions for dealing with them. */
object DataType {
  trait Aux[T] extends DataType {
    override type ScalaType = T
  }

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
    Set(INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all quantized data types. */
  val quantizedDataTypes: Set[DataType] = {
    Set(BFLOAT16, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all unsigned data types. */
  val unsignedDataTypes: Set[DataType] = {
    Set(UINT8, UINT16, UINT32, QUINT8, QUINT16)
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
  @inline def dataTypeOf[T: SupportedType](value: T): DataType = value.dataType

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
  def fromCValue(cValue: Int): DataType = cValue match {
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

  /** Returns the data type corresponding to the provided name.
    *
    * @param  name Data type name.
    * @return Data type corresponding to the provided C value.
    * @throws IllegalArgumentException If an invalid data type name is provided.
    */
  @throws[IllegalArgumentException]
  def fromName(name: String): DataType = name match {
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
    @inline def dataTypeOf[T: SupportedType](value: T): DataType = DataType.dataTypeOf(value)

    @throws[IllegalArgumentException]
    def dataType(cValue: Int): DataType = DataType.fromCValue(cValue)

    @throws[IllegalArgumentException]
    def dataType(name: String): DataType = DataType.fromName(name)

    def mostPreciseDataType(dataTypes: DataType*): DataType = DataType.mostPrecise(dataTypes: _*)
    def leastPreciseDataType(dataTypes: DataType*): DataType = DataType.leastPrecise(dataTypes: _*)
  }
}

private[api] object STRING extends DataType.Aux[String] {
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
}

private[api] object BOOLEAN extends DataType.Aux[Boolean] {
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
}

// TODO: Fix/complete the following implementations for FLOAT16, BFLOAT16, COMPLEX64, and COMPLEX128.

private[api] object FLOAT16 extends DataType.Aux[Float] {
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
}

private[api] object FLOAT32 extends DataType.Aux[Float] {
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
}

private[api] object FLOAT64 extends DataType.Aux[Double] {
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
}

private[api] object BFLOAT16 extends DataType.Aux[Float] {
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
}

private[api] object COMPLEX64 extends DataType.Aux[Double] {
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
}

private[api] object COMPLEX128 extends DataType.Aux[Double] {
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
}

private[api] object INT8 extends DataType.Aux[Byte] {
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
}

private[api] object INT16 extends DataType.Aux[Short] {
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
}

private[api] object INT32 extends DataType.Aux[Int] {
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
}

private[api] object INT64 extends DataType.Aux[Long] {
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
}

private[api] object UINT8 extends DataType.Aux[UByte] {
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
}

private[api] object UINT16 extends DataType.Aux[UShort] {
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
}

private[api] object UINT32 extends DataType.Aux[Long] {
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
}

// TODO: !!! [TYPES] Add UINT64 support.

private[api] object QINT8 extends DataType.Aux[Byte] {
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
}

private[api] object QINT16 extends DataType.Aux[Short] {
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
}

private[api] object QINT32 extends DataType.Aux[Int] {
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
}

private[api] object QUINT8 extends DataType.Aux[UByte] {
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
}

private[api] object QUINT16 extends DataType.Aux[UShort] {
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
}

private[api] object RESOURCE extends DataType.Aux[Long] {
  override implicit val supportedType: SupportedType[Long] = longIsSupportedType

  override val name    : String = "RESOURCE"
  override val cValue  : Int    = 20
  override val byteSize: Int    = -1
  override val priority: Int    = -1

  override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_RESOURCE

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
    throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
    throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
  }
}

private[api] object VARIANT extends DataType.Aux[Long] {
  override implicit val supportedType: SupportedType[Long] = longIsSupportedType

  override val name    : String = "VARIANT"
  override val cValue  : Int    = 21
  override val byteSize: Int    = -1
  override val priority: Int    = -1

  override def protoType: org.tensorflow.framework.DataType = org.tensorflow.framework.DataType.DT_VARIANT

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
    throw new UnsupportedOperationException("The variant data type is not supported on the Scala side.")
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
    throw new UnsupportedOperationException("The variant data type is not supported on the Scala side.")
  }
}
