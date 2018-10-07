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
) {
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

  val STRING    : DataType[String]        = DataType[String]("String", cValue = 7, byteSize = None, DT_STRING)
  val BOOLEAN   : DataType[Boolean]       = DataType[Boolean]("Boolean", cValue = 10, byteSize = Some(1), DT_BOOL)
  val FLOAT16   : DataType[Half]          = DataType[Half]("Half", cValue = 19, byteSize = Some(2), DT_HALF)
  val FLOAT32   : DataType[Float]         = DataType[Float]("Float", cValue = 1, byteSize = Some(4), DT_FLOAT)
  val FLOAT64   : DataType[Double]        = DataType[Double]("Double", cValue = 2, byteSize = Some(8), DT_DOUBLE)
  val BFLOAT16  : DataType[TruncatedHalf] = DataType[TruncatedHalf]("TruncatedFloat", cValue = 14, byteSize = Some(2), DT_BFLOAT16)
  val COMPLEX64 : DataType[ComplexFloat]  = DataType[ComplexFloat]("ComplexFloat", cValue = 8, byteSize = Some(8), DT_COMPLEX64)
  val COMPLEX128: DataType[ComplexDouble] = DataType[ComplexDouble]("ComplexDouble", cValue = 18, byteSize = Some(16), DT_COMPLEX128)
  val INT8      : DataType[Byte]          = DataType[Byte]("Byte", cValue = 6, byteSize = Some(1), DT_INT8)
  val INT16     : DataType[Short]         = DataType[Short]("Short", cValue = 5, byteSize = Some(2), DT_INT16)
  val INT32     : DataType[Int]           = DataType[Int]("Int", cValue = 3, byteSize = Some(4), DT_INT32)
  val INT64     : DataType[Long]          = DataType[Long]("Long", cValue = 9, byteSize = Some(8), DT_INT64)
  val UINT8     : DataType[UByte]         = DataType[UByte]("UByte", cValue = 4, byteSize = Some(1), DT_UINT8)
  val UINT16    : DataType[UShort]        = DataType[UShort]("UShort", cValue = 17, byteSize = Some(2), DT_UINT16)
  val UINT32    : DataType[UInt]          = DataType[UInt]("UInt", cValue = 22, byteSize = Some(4), DT_UINT32)
  val UINT64    : DataType[ULong]         = DataType[ULong]("ULong", cValue = 23, byteSize = Some(8), DT_UINT64)
  val QINT8     : DataType[QByte]         = DataType[QByte]("QByte", cValue = 11, byteSize = Some(1), DT_QINT8)
  val QINT16    : DataType[QShort]        = DataType[QShort]("QShort", cValue = 15, byteSize = Some(2), DT_QINT16)
  val QINT32    : DataType[QInt]          = DataType[QInt]("QInt", cValue = 13, byteSize = Some(4), DT_QINT32)
  val QUINT8    : DataType[QUByte]        = DataType[QUByte]("QUByte", cValue = 12, byteSize = Some(1), DT_QUINT8)
  val QUINT16   : DataType[QUShort]       = DataType[QUShort]("QUShort", cValue = 16, byteSize = Some(2), DT_QUINT16)
  val RESOURCE  : DataType[Resource]      = DataType[Resource]("Resource", cValue = 20, byteSize = Some(1), DT_RESOURCE)
  val VARIANT   : DataType[Variant]       = DataType[Variant]("Variant", cValue = 21, byteSize = Some(1), DT_VARIANT)

  // TODO: [TYPES] !!! Remove the following.

  @inline private def erasedValue[T]: T = {
    null.asInstanceOf[T]
  }

  @inline implicit def apply[T]: DataType[T] = {
    val dataType = erasedValue[T] match {
      case _: String => STRING
      case _: Boolean => BOOLEAN
      case _: Half => FLOAT16
      case _: Float => FLOAT32
      case _: Double => FLOAT64
      case _: TruncatedHalf => BFLOAT16
      case _: ComplexFloat => COMPLEX64
      case _: ComplexDouble => COMPLEX128
      case _: Byte => INT8
      case _: Short => INT16
      case _: Int => INT32
      case _: Long => INT64
      case _: UByte => UINT8
      case _: UShort => UINT16
      case _: UInt => UINT32
      case _: ULong => UINT64
      case _: QByte => QINT8
      case _: QShort => QINT16
      case _: QInt => QINT32
      case _: QUByte => QUINT8
      case _: QUShort => QUINT16
      case _: Resource => RESOURCE
      case _: Variant => VARIANT
      case _ => ???
    }
    dataType.asInstanceOf[DataType[T]]
  }

  //endregion Data Type Instances

  //region Helper Methods

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

  //endregion Helper Methods

  /** "Zero" value for the provided data type.
    *
    * @tparam T Data type.
    * @return "Zero" value for the provided data type.
    * @throws IllegalArgumentException If the provided data type is not supported (which should never happen).
    */
  @throws[IllegalArgumentException]
  @inline def zero[T: TF]: T = {
    val dataType = TF[T].dataType
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
    * @tparam T Data type.
    * @return "One" value for the provided data type.
    * @throws IllegalArgumentException If the provided data type is not supported (which should never happen).
    */
  @throws[IllegalArgumentException]
  @inline def one[T: TF]: T = {
    val dataType = TF[T].dataType
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
    * @param  buffer Byte buffer in which to put the element.
    * @param  index  Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @param  value  Element to put into the provided byte buffer.
    * @tparam T      Data type of the elements stored in the buffer.
    * @return Number of bytes written. For all data types with a known byte size (i.e., not equal to `-1`), the return
    *         value is equal to the byte size.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def putElementInBuffer[T: TF](
      buffer: ByteBuffer,
      index: Int,
      value: T
  ): Int = {
    val dataType = TF[T].dataType
    (value, dataType) match {
      case (v: String, STRING) =>
        val stringBytes = v.getBytes(StandardCharsets.ISO_8859_1)
        NativeTensor.setStringBytes(
          stringBytes,
          buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
      case (v: Boolean, BOOLEAN) =>
        buffer.put(index, if (v) 1 else 0)
        dataType.byteSize.get
      case (v: Half, FLOAT16) => ???
      case (v: Float, FLOAT32) =>
        buffer.putFloat(index, v)
        dataType.byteSize.get
      case (v: Double, FLOAT64) =>
        buffer.putDouble(index, v)
        dataType.byteSize.get
      case (v: TruncatedHalf, BFLOAT16) => ???
      case (v: ComplexFloat, COMPLEX64) => ???
      case (v: ComplexDouble, COMPLEX128) => ???
      case (v: Byte, INT8) =>
        buffer.put(index, v)
        dataType.byteSize.get
      case (v: Short, INT16) =>
        buffer.putShort(index, v)
        dataType.byteSize.get
      case (v: Int, INT32) =>
        buffer.putInt(index, v)
        dataType.byteSize.get
      case (v: Long, INT64) =>
        buffer.putLong(index, v)
        dataType.byteSize.get
      case (v: UByte, UINT8) =>
        buffer.put(index, v.data)
        dataType.byteSize.get
      case (v: UShort, UINT16) =>
        buffer.putChar(index, v.data.toChar)
        dataType.byteSize.get
      case (v: UInt, UINT32) =>
        buffer.putInt(index, v.data.toInt)
        dataType.byteSize.get
      case (v: ULong, UINT64) => ???
      case (v: QByte, QINT8) =>
        buffer.put(index, v.data)
        dataType.byteSize.get
      case (v: QShort, QINT16) =>
        buffer.putChar(index, v.data.toChar)
        dataType.byteSize.get
      case (v: QInt, QINT32) =>
        buffer.putInt(index, v.data.toInt)
        dataType.byteSize.get
      case (v: QUByte, QUINT8) =>
        buffer.put(index, v.data)
        dataType.byteSize.get
      case (v: QUShort, QUINT16) =>
        buffer.putChar(index, v.data.toChar)
        dataType.byteSize.get
      case (v: Long, RESOURCE) =>
        buffer.putLong(index, v)
        dataType.byteSize.get
      case (v: Long, VARIANT) =>
        buffer.putLong(index, v)
        dataType.byteSize.get
      case _ => ???
    }
  }

  /** Gets an element of the specified data type, from the provided byte buffer.
    *
    * @param  buffer Byte buffer from which to get an element.
    * @param  index  Index of the element in the byte buffer (i.e., byte index where the element's bytes start).
    * @tparam T      Data type of the elements stored in the buffer.
    * @return Obtained element.
    * @throws UnsupportedOperationException For unsupported data types on the Scala side.
    */
  @throws[UnsupportedOperationException]
  private[api] def getElementFromBuffer[T: TF](
      buffer: ByteBuffer,
      index: Int
  ): T = {
    val dataType = TF[T].dataType
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

  private[api] def addToTensorProtoBuilder[T: TF](
      builder: TensorProto.Builder,
      value: T
  ): Unit = {
    val dataType = TF[T].dataType
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
