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
case class DataType[+T](
    name: String,
    private[api] val cValue: Int,
    byteSize: Option[Int],
    protoType: org.tensorflow.framework.DataType
)(implicit val evSupportedType: SupportedType[T]) {
  //region Data Type Properties

  /** Size in bytes of each value with this data type, as returned by the native TensorFlow library. Returns `None` if
    * the size is not available.
    *
    * Note that this value is currently not used anywhere within the TensorFlow Scala API.
    */
  private[types] lazy val nativeByteSize: Option[Int] = {
    NativeLibrary.dataTypeSize(cValue) match {
      case 0 => None
      case s => Some(s)
    }
  }

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
  def isBoolean: Boolean = this == DataType.BOOLEAN

  //endregion Data Type Set Helper Methods

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

  override def toString: String = {
    name
  }

  override def equals(that: Any): Boolean = that match {
    case that: DataType[T] => this.cValue == that.cValue
    case _ => false
  }

  override def hashCode: Int = {
    cValue
  }
}

/** Contains all supported data types along with some helper functions for dealing with them. */
object DataType {
  //region Data Type Instances

  val STRING    : DataType[String]        = DataType[String]("STRING", cValue = 7, byteSize = None, DT_STRING)
  val BOOLEAN   : DataType[Boolean]       = DataType[Boolean]("BOOLEAN", cValue = 10, byteSize = Some(1), DT_BOOL)
  val FLOAT16   : DataType[Half]          = DataType[Half]("FLOAT16", cValue = 19, byteSize = Some(2), DT_HALF)
  val FLOAT32   : DataType[Float]         = DataType[Float]("FLOAT32", cValue = 1, byteSize = Some(4), DT_FLOAT)
  val FLOAT64   : DataType[Double]        = DataType[Double]("FLOAT64", cValue = 2, byteSize = Some(8), DT_DOUBLE)
  val BFLOAT16  : DataType[TruncatedHalf] = DataType[TruncatedHalf]("BFLOAT16", cValue = 14, byteSize = Some(2), DT_BFLOAT16)
  val COMPLEX64 : DataType[ComplexFloat]  = DataType[ComplexFloat]("COMPLEX64", cValue = 8, byteSize = Some(8), DT_COMPLEX64)
  val COMPLEX128: DataType[ComplexDouble] = DataType[ComplexDouble]("COMPLEX128", cValue = 18, byteSize = Some(16), DT_COMPLEX128)
  val INT8      : DataType[Byte]          = DataType[Byte]("INT8", cValue = 6, byteSize = Some(1), DT_INT8)
  val INT16     : DataType[Short]         = DataType[Short]("INT16", cValue = 5, byteSize = Some(2), DT_INT16)
  val INT32     : DataType[Int]           = DataType[Int]("INT32", cValue = 3, byteSize = Some(4), DT_INT32)
  val INT64     : DataType[Long]          = DataType[Long]("INT64", cValue = 9, byteSize = Some(8), DT_INT64)
  val UINT8     : DataType[UByte]         = DataType[UByte]("UINT8", cValue = 4, byteSize = Some(1), DT_UINT8)
  val UINT16    : DataType[UShort]        = DataType[UShort]("UINT16", cValue = 17, byteSize = Some(2), DT_UINT16)
  val UINT32    : DataType[UInt]          = DataType[UInt]("UINT32", cValue = 22, byteSize = Some(4), DT_UINT32)
  val UINT64    : DataType[ULong]         = DataType[ULong]("UINT64", cValue = 23, byteSize = Some(8), DT_UINT64)
  val QINT8     : DataType[QByte]         = DataType[QByte]("QINT8", cValue = 11, byteSize = Some(1), DT_QINT8)
  val QINT16    : DataType[QShort]        = DataType[QShort]("QINT16", cValue = 15, byteSize = Some(2), DT_QINT16)
  val QINT32    : DataType[QInt]          = DataType[QInt]("QINT32", cValue = 13, byteSize = Some(4), DT_QINT32)
  val QUINT8    : DataType[QUByte]        = DataType[QUByte]("QUINT8", cValue = 12, byteSize = Some(1), DT_QUINT8)
  val QUINT16   : DataType[QUShort]       = DataType[QUShort]("QUINT16", cValue = 16, byteSize = Some(2), DT_QUINT16)
  val RESOURCE  : DataType[Long]          = DataType[Long]("RESOURCE", cValue = 20, byteSize = Some(1), DT_RESOURCE)
  val VARIANT   : DataType[Long]          = DataType[Long]("VARIANT", cValue = 21, byteSize = Some(1), DT_VARIANT)

  //endregion Data Type Instances

  //region Helper Methods

  /** Returns the data type of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    */
  @inline def dataTypeOf[T](value: T)(implicit ev: SupportedType[T]): DataType[T] = {
    ev.dataType
  }

  /** Returns the data type that corresponds to the provided C value.
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
        s"Data type C value '$value' is not recognized in Scala " +
            s"(TensorFlow version ${NativeLibrary.version}).")
    }
    dataType.asInstanceOf[DataType[T]]
  }

  /** Returns the data type that corresponds to the provided name.
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
        s"Data type name '$value' is not recognized in Scala " +
            s"(TensorFlow version ${NativeLibrary.version}).")
    }
    dataType.asInstanceOf[DataType[T]]
  }

  //endregion Helper Methods

  /** "Zero" value for the provided data type.
    *
    * @param  dataType Data type.
    * @return "Zero" value for the provided data type.
    * @throws IllegalArgumentException If the provided data type is not supported (which should never happen).
    */
  @throws[IllegalArgumentException]
  def zero[T](dataType: DataType[T]): T = {
    val value = dataType match {
      case STRING => ""
      case BOOLEAN => false
      case FLOAT16 => ???
      case FLOAT32 => 0.0f
      case FLOAT64 => 0.0
      case BFLOAT16 => ???
      case COMPLEX64 => ???
      case COMPLEX128 => ???
      case INT8 => 0
      case INT16 => 0
      case INT32 => 0
      case INT64 => 0L
      case UINT8 => ???
      case UINT16 => ???
      case UINT32 => ???
      case UINT64 => ???
      case QINT8 => ???
      case QINT16 => ???
      case QINT32 => ???
      case QUINT8 => ???
      case QUINT16 => ???
      case RESOURCE => ???
      case VARIANT => ???
      case _ => throw new IllegalArgumentException(
        "Invalid data type encountered. This should never happen.")
    }
    value.asInstanceOf[T]
  }

  /** "One" value for the provided data type.
    *
    * @param  dataType Data type.
    * @return "One" value for the provided data type.
    * @throws IllegalArgumentException If the provided data type is not supported (which should never happen).
    */
  @throws[IllegalArgumentException]
  def one[T](dataType: DataType[T]): T = {
    val value = dataType match {
      case STRING => ???
      case BOOLEAN => true
      case FLOAT16 => ???
      case FLOAT32 => 1.0f
      case FLOAT64 => 1.0
      case BFLOAT16 => ???
      case COMPLEX64 => ???
      case COMPLEX128 => ???
      case INT8 => 1
      case INT16 => 1
      case INT32 => 1
      case INT64 => 1L
      case UINT8 => ???
      case UINT16 => ???
      case UINT32 => ???
      case UINT64 => ???
      case QINT8 => ???
      case QINT16 => ???
      case QINT32 => ???
      case QUINT8 => ???
      case QUINT16 => ???
      case RESOURCE => ???
      case VARIANT => ???
      case _ => throw new IllegalArgumentException(
        "Invalid data type encountered. This should never happen.")
    }
    value.asInstanceOf[T]
  }

  /** Puts an element of the specified data type into the provided byte buffer.
    *
    * @param  buffer   Byte buffer in which to put the element.
    * @param  index    Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @param  dataType Data type of the elements stored in the buffer.
    * @param  value    Element to put into the provided byte buffer.
    * @return Number of bytes written. For all data types with a known byte size (i.e., not equal to `-1`), the return
    *         value is equal to the byte size.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def putElementInBuffer[T](
      buffer: ByteBuffer,
      index: Int,
      dataType: DataType[T],
      value: T
  ): Int = {
    (value, dataType) match {
      case (v: String, STRING) =>
        val stringBytes = v.getBytes(StandardCharsets.ISO_8859_1)
        NativeTensor.setStringBytes(
          stringBytes,
          buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
      case (v: Boolean, BOOLEAN) => buffer.put(index, if (v) 1 else 0)
      case (v: Half, FLOAT16) => ???
      case (v: Float, FLOAT32) => buffer.putFloat(index, v)
      case (v: Double, FLOAT64) => buffer.putDouble(index, v)
      case (v: TruncatedHalf, BFLOAT16) => ???
      case (v: ComplexFloat, COMPLEX64) => ???
      case (v: ComplexDouble, COMPLEX128) => ???
      case (v: Byte, INT8) => buffer.put(index, v)
      case (v: Short, INT16) => buffer.putShort(index, v)
      case (v: Int, INT32) => buffer.putInt(index, v)
      case (v: Long, INT64) => buffer.putLong(index, v)
      case (v: UByte, UINT8) => buffer.put(index, v.data)
      case (v: UShort, UINT16) => buffer.putChar(index, v.data.toChar)
      case (v: UInt, UINT32) => buffer.putInt(index, v.data.toInt)
      case (v: ULong, UINT64) => ???
      case (v: QByte, QINT8) => buffer.put(index, v.data)
      case (v: QShort, QINT16) => buffer.putChar(index, v.data.toChar)
      case (v: QInt, QINT32) => buffer.putInt(index, v.data.toInt)
      case (v: QUByte, QUINT8) => buffer.put(index, v.data)
      case (v: QUShort, QUINT16) => buffer.putChar(index, v.data.toChar)
      case (v: Long, RESOURCE) => buffer.putLong(index, v)
      case (v: Long, VARIANT) => buffer.putLong(index, v)
      case _ => ???
    }
    dataType.byteSize.get
  }

  /** Gets an element of the specified data type, from the provided byte buffer.
    *
    * @param  buffer   Byte buffer from which to get an element.
    * @param  index    Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @param  dataType Data type of the elements stored in the buffer.
    * @return Obtained element.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def getElementFromBuffer[T](
      buffer: ByteBuffer,
      index: Int,
      dataType: DataType[T]
  ): T = {
    val value = dataType match {
      case STRING =>
        val bufferWithOffset = buffer.duplicate().position(index).asInstanceOf[ByteBuffer]
        val stringBytes = NativeTensor.getStringBytes(bufferWithOffset.slice())
        new String(stringBytes, StandardCharsets.ISO_8859_1)
      case BOOLEAN => buffer.get(index) == 1
      case FLOAT16 => ???
      case FLOAT32 => buffer.getFloat(index)
      case FLOAT64 => buffer.getDouble(index)
      case BFLOAT16 => ???
      case COMPLEX64 => ???
      case COMPLEX128 => ???
      case INT8 => buffer.get(index)
      case INT16 => buffer.getShort(index)
      case INT32 => buffer.getInt(index)
      case INT64 => buffer.getLong(index)
      case UINT8 => UByte(buffer.get(index))
      case UINT16 => UShort(buffer.getChar(index).toShort)
      case UINT32 => UInt(buffer.getInt(index))
      case UINT64 => ???
      case QINT8 => QByte(buffer.get(index))
      case QINT16 => QShort(buffer.getChar(index).toShort)
      case QINT32 => QInt(buffer.getInt(index))
      case QUINT8 => QUByte(buffer.get(index))
      case QUINT16 => QUShort(buffer.getChar(index).toShort)
      case RESOURCE => buffer.getLong(index)
      case VARIANT => buffer.getLong(index)
      case _ => ???
    }
    value.asInstanceOf[T]
  }

  private[api] def addToTensorProtoBuilder[T](
      builder: TensorProto.Builder,
      dataType: DataType[T],
      value: T
  ): Unit = {
    (value, dataType) match {
      case (v: String, STRING) => builder.addStringVal(ByteString.copyFrom(v.getBytes))
      case (v: Boolean, BOOLEAN) => builder.addBoolVal(v)
      case (v: Half, FLOAT16) => ???
      case (v: Float, FLOAT32) => builder.addFloatVal(v)
      case (v: Double, FLOAT64) => builder.addDoubleVal(v)
      case (v: TruncatedHalf, BFLOAT16) => ???
      case (v: ComplexFloat, COMPLEX64) => ???
      case (v: ComplexDouble, COMPLEX128) => ???
      case (v: Byte, INT8) => builder.addIntVal(v)
      case (v: Short, INT16) => builder.addIntVal(v)
      case (v: Int, INT32) => builder.addIntVal(v)
      case (v: Long, INT64) => builder.addInt64Val(v)
      case (v: UByte, UINT8) => builder.addIntVal(v.data.toInt)
      case (v: UShort, UINT16) => builder.addIntVal(v.data.toInt)
      case (v: UInt, UINT32) => ???
      case (v: ULong, UINT64) => ???
      case (v: QByte, QINT8) => ???
      case (v: QShort, QINT16) => ???
      case (v: QInt, QINT32) => ???
      case (v: QUByte, QUINT8) => ???
      case (v: QUShort, QUINT16) => ???
      case (v: Long, RESOURCE) => ???
      case (v: Long, VARIANT) => ???
      case _ => ???
    }
  }

  //region Data Type Sets

  /** Set of all floating-point data types. */
  val floatingPointDataTypes: Set[DataType[Any]] = {
    Set(FLOAT16, FLOAT32, FLOAT64, BFLOAT16)
  }

  /** Set of all complex data types. */
  val complexDataTypes: Set[DataType[Any]] = {
    Set(COMPLEX64, COMPLEX128)
  }

  /** Set of all integer data types. */
  val integerDataTypes: Set[DataType[Any]] = {
    Set(INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all quantized data types. */
  val quantizedDataTypes: Set[DataType[Any]] = {
    Set(BFLOAT16, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all unsigned data types. */
  val unsignedDataTypes: Set[DataType[Any]] = {
    Set(UINT8, UINT16, UINT32, UINT64, QUINT8, QUINT16)
  }

  /** Set of all numeric data types. */
  val numericDataTypes: Set[DataType[Any]] = {
    floatingPointDataTypes ++ complexDataTypes ++ integerDataTypes ++ quantizedDataTypes
  }

  //endregion Data Type Sets
}
