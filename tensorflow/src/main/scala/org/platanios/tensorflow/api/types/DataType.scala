package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.types.SupportedType.Implicits._
import org.platanios.tensorflow.jni.{Tensor => NativeTensor, TensorFlow => NativeLibrary}

import java.nio.ByteBuffer
import java.nio.charset.Charset

import spire.math.{UByte, UShort}

// TODO: Add min/max-value and "isSigned" information.
// TODO: Add support for half-precision floating-point numbers and for complex numbers.
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
    // case DataType.Complex64 => DataType.Float32 TODO: [COMPLEX]
    // case DataType.Complex128 => DataType.Float64
    case _ => this
  }

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

sealed trait FixedSizeDataType extends DataType

sealed trait NumericDataType extends FixedSizeDataType {
  override implicit val supportedType: NumericSupportedType[ScalaType]
}

sealed trait SignedNumericDataType extends NumericDataType {
  override implicit val supportedType: SignedNumericSupportedType[ScalaType]
}

sealed trait RealNumericDataType extends SignedNumericDataType {
  override implicit val supportedType: RealNumericSupportedType[ScalaType]
}

sealed trait ComplexNumericDataType extends SignedNumericDataType {
  override implicit val supportedType: ComplexNumericSupportedType[ScalaType]
}

private[api] object STRING extends DataType {
  override type ScalaType = String
  override implicit val supportedType = StringIsSupportedType

  override val name    : String = "STRING"
  override val cValue  : Int    = 7
  override val byteSize: Int    = -1
  override val priority: Int    = 1000

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    val stringBytes = element.getBytes(Charset.forName("UTF-8"))
    NativeTensor.setStringBytes(stringBytes, buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    val stringBytes = NativeTensor.getStringBytes(buffer.duplicate().position(index).asInstanceOf[ByteBuffer].slice())
    new String(stringBytes, Charset.forName("UTF-8"))
  }
}

private[api] object BOOLEAN extends FixedSizeDataType {
  override type ScalaType = Boolean
  override implicit val supportedType = BooleanIsSupportedType

  override val name    : String = "BOOLEAN"
  override val cValue  : Int    = 10
  override val byteSize: Int    = 1
  override val priority: Int    = 0

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.put(index, if (element) 1 else 0)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.get(index) == 1
  }
}

// TODO: Add Float16(cValue = 19, byteSize = 2).

private[api] object FLOAT32 extends RealNumericDataType {
  override type ScalaType = Float
  override implicit val supportedType = FloatIsSupportedType

  override val name    : String = "FLOAT32"
  override val cValue  : Int    = 1
  override val byteSize: Int    = 4
  override val priority: Int    = 220

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putFloat(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getFloat(index)
  }
}

private[api] object FLOAT64 extends RealNumericDataType {
  override type ScalaType = Double
  override implicit val supportedType = DoubleIsSupportedType

  override val name    : String = "FLOAT64"
  override val cValue  : Int    = 2
  override val byteSize: Int    = 8
  override val priority: Int    = 230

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putDouble(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getDouble(index)
  }
}

// TODO: Add BFLOAT16(cValue = 14, byteSize = 2).
// TODO: Add COMPLEX64(cValue = 8, byteSize = 8).
// TODO: Add COMPLEX128(cValue = 18, byteSize = 16).

private[api] object INT8 extends RealNumericDataType {
  override type ScalaType = Byte
  override implicit val supportedType = ByteIsSupportedType

  override val name    : String = "INT8"
  override val cValue  : Int    = 6
  override val byteSize: Int    = 1
  override val priority: Int    = 40

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.put(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.get(index)
  }
}

private[api] object INT16 extends RealNumericDataType {
  override type ScalaType = Short
  override implicit val supportedType = ShortIsSupportedType

  override val name    : String = "INT16"
  override val cValue  : Int    = 5
  override val byteSize: Int    = 2
  override val priority: Int    = 80

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putShort(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getShort(index)
  }
}

private[api] object INT32 extends RealNumericDataType {
  override type ScalaType = Int
  override implicit val supportedType = IntIsSupportedType

  override val name    : String = "INT32"
  override val cValue  : Int    = 3
  override val byteSize: Int    = 4
  override val priority: Int    = 100

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putInt(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getInt(index)
  }
}

private[api] object INT64 extends RealNumericDataType {
  override type ScalaType = Long
  override implicit val supportedType = LongIsSupportedType

  override val name    : String = "INT64"
  override val cValue  : Int    = 9
  override val byteSize: Int    = 8
  override val priority: Int    = 110

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putLong(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getLong(index)
  }
}

private[api] object UINT8 extends NumericDataType {
  override type ScalaType = UByte
  override implicit val supportedType = UByteIsSupportedType

  override val name    : String = "UINT8"
  override val cValue  : Int    = 4
  override val byteSize: Int    = 1
  override val priority: Int    = 20

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.put(index, element.toByte)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    UByte(buffer.get(index))
  }
}

private[api] object UINT16 extends NumericDataType {
  override type ScalaType = UShort
  override implicit val supportedType = UShortIsSupportedType

  override val name    : String = "UINT16"
  override val cValue  : Int    = 17
  override val byteSize: Int    = 2
  override val priority: Int    = 60

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putChar(index, element.toChar)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    UShort(buffer.getChar(index))
  }
}

private[api] object QINT8 extends RealNumericDataType {
  override type ScalaType = Byte
  override implicit val supportedType = ByteIsSupportedType

  override val name    : String = "QINT8"
  override val cValue  : Int    = 11
  override val byteSize: Int    = 1
  override val priority: Int    = 30

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.put(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.get(index)
  }
}

private[api] object QINT16 extends RealNumericDataType {
  override type ScalaType = Short
  override implicit val supportedType = ShortIsSupportedType

  override val name    : String = "QINT16"
  override val cValue  : Int    = 15
  override val byteSize: Int    = 2
  override val priority: Int    = 70

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putShort(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getShort(index)
  }
}

private[api] object QINT32 extends RealNumericDataType {
  override type ScalaType = Int
  override implicit val supportedType = IntIsSupportedType

  override val name    : String = "QINT32"
  override val cValue  : Int    = 13
  override val byteSize: Int    = 4
  override val priority: Int    = 90

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putInt(index, element)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    buffer.getInt(index)
  }
}

private[api] object QUINT8 extends NumericDataType {
  override type ScalaType = UByte
  override implicit val supportedType = UByteIsSupportedType

  override val name    : String = "QUINT8"
  override val cValue  : Int    = 12
  override val byteSize: Int    = 1
  override val priority: Int    = 10

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.put(index, element.toByte)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    UByte(buffer.get(index))
  }
}

private[api] object QUINT16 extends NumericDataType {
  override type ScalaType = UShort
  override implicit val supportedType = UShortIsSupportedType

  override val name    : String = "QUINT16"
  override val cValue  : Int    = 16
  override val byteSize: Int    = 2
  override val priority: Int    = 50

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    buffer.putChar(index, element.toChar)
    byteSize
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    UShort(buffer.getChar(index))
  }
}

private[api] object RESOURCE extends DataType {
  override type ScalaType = Long
  override implicit val supportedType = LongIsSupportedType

  override val name    : String = "RESOURCE"
  override val cValue  : Int    = 20
  override val byteSize: Int    = -1
  override val priority: Int    = -1

  private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
    throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
  }

  private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
    throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
  }
}

/** Contains all supported data types along with some helper functions for dealing with them. */
private[api] object DataType {
  //region Data Type Sets

  /** Set of all floating-point data types. */
  private[api] val floatingPointDataTypes: Set[DataType] = {
    Set(FLOAT32, FLOAT64) // TODO: TFFloat16, TFBFloat16.
  }

  /** Set of all complex data types. */
  private[api] val complexDataTypes: Set[DataType] = {
    Set() // TODO: [COMPLEX] TFComplex64, TFComplex128.
  }

  /** Set of all integer data types. */
  private[api] val integerDataTypes: Set[DataType] = {
    Set(INT8, INT16, INT32, INT64, UINT8, UINT16, QINT8, QINT16, QINT32, QUINT8, QUINT16)
  }

  /** Set of all quantized data types. */
  private[api] val quantizedDataTypes: Set[DataType] = {
    Set(QINT8, QINT16, QINT32, QUINT8, QUINT16) // TODO: TFBFloat16.
  }

  /** Set of all unsigned data types. */
  private[api] val unsignedDataTypes: Set[DataType] = {
    Set(UINT8, UINT16, QUINT8, QUINT16)
  }

  /** Set of all numeric data types. */
  private[api] val numericDataTypes: Set[DataType] = {
    floatingPointDataTypes ++ complexDataTypes ++ integerDataTypes ++ quantizedDataTypes
  }

  //endregion Data Type Sets

  //region Helper Methods

  /** Returns the [[DataType]] of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    */
  @inline private[api] def dataTypeOf[T: SupportedType](value: T): DataType = value.dataType

  /** Returns the data type corresponding to the provided C value.
    *
    * By C value here we refer to an integer representing a data type in the `TF_DataType` enum of the TensorFlow C API.
    *
    * @param  cValue C value.
    * @return Data type corresponding to the provided C value.
    * @throws IllegalArgumentException If an invalid C value is provided.
    */
  @throws[IllegalArgumentException]
  private[api] def fromCValue(cValue: Int): DataType = cValue match {
    case BOOLEAN.cValue => BOOLEAN
    case STRING.cValue => STRING
    // case TFFloat16.cValue => TFFloat16
    case FLOAT32.cValue => FLOAT32
    case FLOAT64.cValue => FLOAT64
    // case TFBFloat16.cValue => TFBFloat16
    // case TFComplex64.cValue => TFComplex64
    // case TFComplex128.cValue => TFComplex128
    case INT8.cValue => INT8
    case INT16.cValue => INT16
    case INT32.cValue => INT32
    case INT64.cValue => INT64
    case UINT8.cValue => UINT8
    case UINT16.cValue => UINT16
    case QINT8.cValue => QINT8
    case QINT16.cValue => QINT16
    case QINT32.cValue => QINT32
    case QUINT8.cValue => QUINT8
    case QUINT16.cValue => QUINT16
    case RESOURCE.cValue => RESOURCE
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
  private[api] def fromName(name: String): DataType = name match {
    case "BOOLEAN" => BOOLEAN
    case "STRING" => STRING
    // case "FLOAT16" => TFFloat16
    case "FLOAT32" => FLOAT32
    case "FLOAT64" => FLOAT64
    // case "BFLOAT16" => TFBFloat16
    // case "COMPLEX64" => TFComplex64
    // case "COMPLEX128" => TFComplex128
    case "INT8" => INT8
    case "INT16" => INT16
    case "INT32" => INT32
    case "INT64" => INT64
    case "UINT8" => UINT8
    case "UINT16" => UINT16
    case "QINT8" => QINT8
    case "QINT16" => QINT16
    case "QINT32" => QINT32
    case "QUINT8" => QUINT8
    case "QUINT16" => QUINT16
    case "RESOURCE" => RESOURCE
    case value => throw new IllegalArgumentException(
      s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
  }

  //endregion Helper Methods
}
