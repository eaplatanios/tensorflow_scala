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

import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

import com.google.protobuf.ByteString
import org.tensorflow.framework.DataType._
import org.tensorflow.framework.TensorProto

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets

/** Represents the data type of the elements in a tensor.
  *
  * @param  name      Name of this data type (mainly useful for logging purposes).
  * @param  cValue    Represents this data type in the `TF_DataType` enum of the TensorFlow C API.
  * @param  byteSize  Size in bytes of each value with this data type. Set to `None` if the size is not fixed.
  * @param  protoType ProtoBuf data type used in serialized representations of tensors.
  * @tparam T         Corresponding Scala type for this TensorFlow data type.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class DataType[T](
    val name: String,
    private[api] val cValue: Int,
    val byteSize: Option[Int],
    val protoType: org.tensorflow.framework.DataType
)(implicit val evSupportedType: SupportedType[T]) {
  //region Data Type Properties

  /** Size in bytes of each value with this data type, as returned by the native TensorFlow library. Returns `None` if
    * the size is not available.
    *
    * Note that this value is currently not used anywhere within the TensorFlow Scala API.
    */
  private[types] lazy val nativeByteSize: Option[Int] = {
    val nativeLibrarySize = NativeLibrary.dataTypeSize(cValue)
    if (nativeLibrarySize == 0)
      None
    else
      Some(nativeLibrarySize)
  }

  // TODO: [TYPES] Remove once the symbolic API becomes generic.

  private[api] val priority: Int

  //endregion Data Type Properties

  //region Data Type Set Helper Methods

  /** Zero value for this data type. */
  def zero: T

  /** One value for this data type. */
  def one: T

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
  def isBoolean: Boolean = this == DataType.BOOLEAN

  // TODO: [TYPES] Removes this after we properly support data types.
  def real: DataType[_] = this match {
    case DataType.COMPLEX64 => DataType.FLOAT32
    case DataType.COMPLEX128 => DataType.FLOAT64
    case d => d
  }

  //endregion Data Type Set Helper Methods

  /** Returns the smallest value that can be represented by this data type. */
  def min: T = throw new UnsupportedOperationException(s"Cannot determine min value for '$this' data type.")

  /** Returns the largest value that can be represented by this data type. */
  def max: T = throw new UnsupportedOperationException(s"Cannot determine max value for '$this' data type.")

  // TODO: [TYPES] !!! Remove the next two methods.

  def minTensor: Tensor[T] = Tensor(min)

  def maxTensor: Tensor[T] = Tensor(max)

  /** Casts the provided value to this data type.
    *
    * Note that this method allows downcasting.
    *
    * @param  value Value to cast.
    * @return Casted value.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  @inline def cast[R](value: R)(implicit ev: SupportedType[R]): T = {
    evSupportedType.cast(value)
  }

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
  private[api] def putElementInBuffer(buffer: ByteBuffer, index: Int, element: T): Int

  /** Gets an element of this data type from the provided byte buffer.
    *
    * @param  buffer Byte buffer from which to get an element.
    * @param  index  Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @return Obtained element.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def getElementFromBuffer(buffer: ByteBuffer, index: Int): T

  private[api] def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: T): Unit

  override def toString: String = name

  override def equals(that: Any): Boolean = that match {
    case that: DataType[T] => this.cValue == that.cValue
    case _ => false
  }

  override def hashCode: Int = cValue
}

/** Contains all supported data types along with some helper functions for dealing with them. */
object DataType {
  //region Helper Methods

  /** Returns the [[DataType]] of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    */
  @inline def dataTypeOf[T](value: T)(implicit ev: SupportedType[T]): DataType[T] = {
    ev.dataType
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
  private[api] def fromCValue[T](cValue: Int): DataType[T] = {
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
    dataType.asInstanceOf[DataType[T]]
  }

  /** Returns the data type corresponding to the provided name.
    *
    * @param  name Data type name.
    * @return Data type corresponding to the provided C value.
    * @throws IllegalArgumentException If an invalid data type name is provided.
    */
  @throws[IllegalArgumentException]
  private[api] def fromName[T](name: String): DataType[T] = {
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
    dataType.asInstanceOf[DataType[T]]
  }

  /** Returns the most precise data type out of the provided data types, based on their `priority` field.
    *
    * @param  dataTypes Data types out of which to pick the most precise.
    * @return Most precise data type in `dataTypes`.
    */
  def mostPrecise(dataTypes: DataType[_]*): DataType[_] = dataTypes.maxBy(_.priority)

  /** Returns the most precise data type out of the provided data types, based on their `priority` field.
    *
    * @param  dataTypes Data types out of which to pick the most precise.
    * @return Most precise data type in `dataTypes`.
    */
  def leastPrecise(dataTypes: DataType[_]*): DataType[_] = dataTypes.minBy(_.priority)

  //endregion Helper Methods

  //region Data Type Instances

  val STRING: DataType[String] = new DataType[String](
    name = "STRING",
    cValue = 7,
    byteSize = None,
    protoType = DT_STRING
  ) {
    override val priority: Int    = 1000

    override def zero: String = ""
    override def one: String = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: String): Int = {
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

  val BOOLEAN: DataType[Boolean] = new DataType[Boolean](
    name = "BOOLEAN",
    cValue = 10,
    byteSize = Some(1),
    protoType = DT_BOOL
  ) {
    override val priority: Int    = 0

    override def zero: Boolean = false
    override def one: Boolean = true

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Boolean): Int = {
      buffer.put(index, if (element) 1 else 0)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Boolean = {
      buffer.get(index) == 1
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Boolean): Unit = {
      tensorProtoBuilder.addBoolVal(value)
    }
  }

  // TODO: Fix/complete the following implementations for FLOAT16, BFLOAT16, COMPLEX64, and COMPLEX128.

  val FLOAT16: DataType[Half] = new DataType[Half](
    name = "FLOAT16",
    cValue = 19,
    byteSize = Some(2),
    protoType = DT_HALF
  ) {
    override val priority: Int    = -1

    override def zero: Half = ??? // 0.0f
    override def one: Half = ??? // 1.0f
    override def min: Half = ??? // -65504f
    override def max: Half = ??? // 65504f

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Half): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Half = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Half): Unit = {
      ???
    }
  }

  val FLOAT32: DataType[Float] = new DataType[Float](
    name = "FLOAT32",
    cValue = 1,
    byteSize = Some(4),
    protoType = DT_FLOAT
  ) {
    override val priority: Int    = 220

    override def zero: Float = 0.0f
    override def one: Float = 1.0f
    override def min: Float = Float.MinValue
    override def max: Float = Float.MaxValue

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Float): Int = {
      buffer.putFloat(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Float = {
      buffer.getFloat(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Float): Unit = {
      tensorProtoBuilder.addFloatVal(value)
    }
  }

  val FLOAT64: DataType[Double] = new DataType[Double](
    name = "FLOAT64",
    cValue = 2,
    byteSize = Some(8),
    protoType = DT_DOUBLE
  ) {
    override val priority: Int    = 230

    override def zero: Double = 0.0
    override def one: Double = 1.0
    override def min: Double = Double.MinValue
    override def max: Double = Double.MaxValue

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Double): Int = {
      buffer.putDouble(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Double = {
      buffer.getDouble(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Double): Unit = {
      tensorProtoBuilder.addDoubleVal(value)
    }
  }

  val BFLOAT16: DataType[TruncatedHalf] = new DataType[TruncatedHalf](
    name = "BFLOAT16",
    cValue = 14,
    byteSize = Some(2),
    protoType = DT_BFLOAT16
  ) {
    override val priority: Int    = -1

    override def zero: TruncatedHalf = ??? // 0.0f
    override def one: TruncatedHalf = ??? // 1.0f
    override def min: TruncatedHalf = ???
    override def max: TruncatedHalf = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: TruncatedHalf): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): TruncatedHalf = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: TruncatedHalf): Unit = {
      ???
    }
  }

  val COMPLEX64: DataType[ComplexFloat] = new DataType[ComplexFloat](
    name = "COMPLEX64",
    cValue = 8,
    byteSize = Some(8),
    protoType = DT_COMPLEX64
  ) {
    override val priority: Int    = -1

    override def zero: ComplexFloat = ???
    override def one: ComplexFloat = ???
    override def min: ComplexFloat = ???
    override def max: ComplexFloat = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ComplexFloat): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ComplexFloat = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: ComplexFloat): Unit = {
      ???
    }
  }

  val COMPLEX128: DataType[ComplexDouble] = new DataType[ComplexDouble](
    name = "COMPLEX128",
    cValue = 18,
    byteSize = Some(16),
    protoType = DT_COMPLEX128
  ) {
    override val priority: Int    = -1

    override def zero: ComplexDouble = ???
    override def one: ComplexDouble = ???
    override def min: ComplexDouble = ???
    override def max: ComplexDouble = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ComplexDouble): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ComplexDouble = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: ComplexDouble): Unit = {
      ???
    }
  }

  val INT8: DataType[Byte] = new DataType[Byte](
    name = "INT8",
    cValue = 6,
    byteSize = Some(1),
    protoType = DT_INT8
  ) {
    override val priority: Int    = 40

    override def zero: Byte = 0
    override def one: Byte = 1
    override def min: Byte = (-128).toByte
    override def max: Byte = 127.toByte

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Byte): Int = {
      buffer.put(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Byte = {
      buffer.get(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Byte): Unit = {
      tensorProtoBuilder.addIntVal(value)
    }
  }

  val INT16: DataType[Short] = new DataType[Short](
    name = "INT16",
    cValue = 5,
    byteSize = Some(2),
    protoType = DT_INT16
  ) {
    override val priority: Int    = 80

    override def zero: Short = 0
    override def one: Short = 1
    override def min: Short = (-32768).toShort
    override def max: Short = 32767.toShort

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Short): Int = {
      buffer.putShort(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Short = {
      buffer.getShort(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Short): Unit = {
      tensorProtoBuilder.addIntVal(value)
    }
  }

  val INT32: DataType[Int] = new DataType[Int](
    name = "INT32",
    cValue = 3,
    byteSize = Some(4),
    protoType = DT_INT32
  ) {
    override val priority: Int    = 100

    override def zero: Int = 0
    override def one: Int = 1
    override def min: Int = -2147483648
    override def max: Int = 2147483647

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Int): Int = {
      buffer.putInt(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Int = {
      buffer.getInt(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Int): Unit = {
      tensorProtoBuilder.addIntVal(value)
    }
  }

  val INT64 = new DataType[Long](
    name = "INT64",
    cValue = 9,
    byteSize = Some(8),
    protoType = DT_INT64
  ) {
    override val priority: Int    = 110

    override def zero: Long = 0L
    override def one: Long = 1L
    override def min: Long = -9223372036854775808L
    override def max: Long = 9223372036854775807L

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putLong(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getLong(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      tensorProtoBuilder.addInt64Val(value)
    }
  }

  val UINT8: DataType[UByte] = new DataType[UByte](
    name = "UINT8",
    cValue = 4,
    byteSize = Some(1),
    protoType = DT_UINT8
  ) {
    override val priority: Int    = 20

    override def zero: UByte = ??? // UByte(0)
    override def one: UByte = ??? // UByte(1)
    override def min: UByte = ??? // UByte(0)
    override def max: UByte = ??? // UByte(255)

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UByte): Int = {
      buffer.put(index, element.data)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UByte = {
      UByte(buffer.get(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UByte): Unit = {
      tensorProtoBuilder.addIntVal(value.data.toInt)
    }
  }

  val UINT16: DataType[UShort] = new DataType[UShort](
    name = "UINT16",
    cValue = 17,
    byteSize = Some(2),
    protoType = DT_UINT16
  ) {
    override val priority: Int    = 60

    override def zero: UShort = ??? // UShort(0)
    override def one: UShort = ??? // UShort(1)
    override def min: UShort = ??? // UShort(0)
    override def max: UShort = ??? // UShort(65535)

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UShort): Int = {
      buffer.putChar(index, element.data.toChar)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UShort = {
      UShort(buffer.getChar(index).toShort)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UShort): Unit = {
      tensorProtoBuilder.addIntVal(value.data.toInt)
    }
  }

  val UINT32: DataType[UInt] = new DataType[UInt](
    name = "UINT32",
    cValue = 22,
    byteSize = Some(4),
    protoType = DT_UINT32
  ) {
    override val priority: Int    = 85

    override def zero: UInt = ??? // 0L
    override def one: UInt = ??? // 1L
    override def min: UInt = ??? // 0L
    override def max: UInt = ??? // 9223372036854775807L

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: UInt): Int = {
      buffer.putInt(index, element.data.toInt)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): UInt = {
      ??? // buffer.getInt(index).toLong
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: UInt): Unit = {
      ???
    }
  }

  val UINT64: DataType[ULong] = new DataType[ULong](
    name = "UINT64",
    cValue = 23,
    byteSize = Some(8),
    protoType = DT_UINT64
  ) {
    override val priority: Int    = 105

    override def zero: ULong = ???
    override def one: ULong = ???
    override def min: ULong = ???
    override def max: ULong = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ULong): Int = {
      ???
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ULong = {
      ???
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: ULong): Unit = {
      ???
    }
  }

  val QINT8: DataType[QByte] = new DataType[QByte](
    name = "QINT8",
    cValue = 11,
    byteSize = Some(1),
    protoType = DT_QINT8
  ) {
    override val priority: Int    = 30

    override def zero: QByte = ???
    override def one: QByte = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: QByte): Int = {
      ???
      // buffer.put(index, element)
      // byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): QByte = {
      ??? // buffer.get(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: QByte): Unit = {
      ???
    }
  }

  val QINT16: DataType[QShort] = new DataType[QShort](
    name = "QINT16",
    cValue = 15,
    byteSize = Some(2),
    protoType = DT_QINT16
  ) {
    override val priority: Int    = 70

    override def zero: QShort = ???
    override def one: QShort = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: QShort): Int = {
      ???
      // buffer.putShort(index, element)
      // byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): QShort = {
      ??? // buffer.getShort(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: QShort): Unit = {
      ???
    }
  }

  val QINT32: DataType[QInt] = new DataType[QInt](
    name = "QINT32",
    cValue = 13,
    byteSize = Some(4),
    protoType = DT_QINT32
  ) {
    override val priority: Int    = 90

    override def zero: QInt = ???
    override def one: QInt = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: QInt): Int = {
      ???
      // buffer.putInt(index, element)
      // byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): QInt = {
      ??? // buffer.getInt(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: QInt): Unit = {
      ???
    }
  }

  val QUINT8: DataType[QUByte] = new DataType[QUByte](
    name = "QUINT8",
    cValue = 12,
    byteSize = Some(1),
    protoType = DT_QUINT8
  ) {
    override val priority: Int    = 10

    override def zero: QUByte = ???
    override def one: QUByte = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: QUByte): Int = {
      ???
      // buffer.put(index, element.toByte)
      // byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): QUByte = {
      ??? // UByte(buffer.get(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: QUByte): Unit = {
      ???
    }
  }

  val QUINT16: DataType[QUShort] = new DataType[QUShort](
    name = "QUINT16",
    cValue = 16,
    byteSize = Some(2),
    protoType = DT_QUINT16
  ) {
    override val priority: Int    = 50

    override def zero: QUShort = ???
    override def one: QUShort = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: QUShort): Int = {
      ???
      // buffer.putChar(index, element.toChar)
      // byteSize
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): QUShort = {
      ??? // UShort(buffer.getChar(index))
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: QUShort): Unit = {
      ???
    }
  }

  val RESOURCE: DataType[Long] = new DataType[Long](
    name = "RESOURCE",
    cValue = 20,
    byteSize = Some(1),
    protoType = DT_RESOURCE
  ) {
    override val priority: Int    = -1

    override def zero: Long = ???
    override def one: Long = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putLong(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getLong(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }

  val VARIANT: DataType[Long] = new DataType[Long](
    name = "VARIANT",
    cValue = 21,
    byteSize = Some(1),
    protoType = DT_VARIANT
  ) {
    override val priority: Int    = -1

    override def zero: Long = ???
    override def one: Long = ???

    private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: Long): Int = {
      buffer.putLong(index, element)
      byteSize.get
    }

    private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): Long = {
      buffer.getLong(index)
    }

    private[api] override def addToTensorProtoBuilder(tensorProtoBuilder: TensorProto.Builder, value: Long): Unit = {
      ???
    }
  }

  //endregion Data Type Instances

  //region Data Type Sets

  /** Set of all floating-point data types. */
  val floatingPointDataTypes: Set[DataType[_]] = {
    Set(FLOAT16, FLOAT32, FLOAT64, BFLOAT16)
  }

  /** Set of all complex data types. */
  val complexDataTypes: Set[DataType[_]] = {
    Set(COMPLEX64, COMPLEX128)
  }

  /** Set of all integer data types. */
  val integerDataTypes: Set[DataType[_]] = {
    Set(INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all quantized data types. */
  val quantizedDataTypes: Set[DataType[_]] = {
    Set(BFLOAT16, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all unsigned data types. */
  val unsignedDataTypes: Set[DataType[_]] = {
    Set(UINT8, UINT16, UINT32, UINT64, QUINT8, QUINT16)
  }

  /** Set of all numeric data types. */
  val numericDataTypes: Set[DataType[_]] = {
    floatingPointDataTypes ++ complexDataTypes ++ integerDataTypes ++ quantizedDataTypes
  }

  //endregion Data Type Sets
}
