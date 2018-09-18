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

import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

import org.tensorflow.framework.TensorProto

import java.nio.ByteBuffer

// TODO: Add "isSigned" information.

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
  def isBoolean: Boolean = this == BOOLEAN

  //endregion Data Type Set Helper Methods

  /** Returns the smallest value that can be represented by this data type. */
  def min: T = throw new UnsupportedOperationException(s"Cannot determine min value for '$this' data type.")

  /** Returns the largest value that can be represented by this data type. */
  def max: T = throw new UnsupportedOperationException(s"Cannot determine max value for '$this' data type.")

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
}
