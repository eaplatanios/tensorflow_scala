package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.types.TypeImplicits._
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
  val cValue: Int

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

  val priority: Int

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
  def isBoolean: Boolean = this == DataType.Bool

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

/** Contains all supported data types along with some helper functions for dealing with them. */
object DataType {
  //region Supported TensorFlow Data Types Definitions

  object Bool extends DataType {
    override type ScalaType = Boolean
    override implicit val supportedType = BooleanIsSupportedType
    
    override val name    : String = "Bool"
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

  object Str extends DataType {
    override type ScalaType = String
    override implicit val supportedType = StringIsSupportedType

    override val name    : String = "String"
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

  // object Float16 extends DataType {
  //   override val name    : String = "Float16"
  //   override val cValue  : Int    = 19
  //   override val byteSize: Int    = 2
  //   override val priority: Int    = 210
  //
  //   override type ScalaType = Float32 // TODO: What data type should we actually use for this?
  //
  //   private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
  //     buffer.putFloat(index, element)
  //   }
  //
  //   private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
  //     org.platanios.tensorflow.api.types.Float32(buffer.getFloat(index))
  //   }
  // }

  object Float32 extends DataType {
    override type ScalaType = Float
    override implicit val supportedType = FloatIsSupportedType

    override val name    : String = "Float32"
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

  object Float64 extends DataType {
    override type ScalaType = Double
    override implicit val supportedType = DoubleIsSupportedType

    override val name    : String = "Float64"
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

  // object BFloat16 extends DataType {
  //   override val name    : String = "BFloat16"
  //   override val cValue  : Int    = 14
  //   override val byteSize: Int    = 2
  //   override val priority: Int    = 200
  //
  //   override type ScalaType = Float32 // TODO: What data type should we actually use for this?
  //
  //   private[api] override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Int = {
  //     buffer.putFloat(index, element)
  //     byteSize
  //   }
  //
  //   private[api] override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
  //     org.platanios.tensorflow.api.types.Float32(buffer.getFloat(index))
  //   }
  // }

  // TODO: Add Complex64(cValue = 8, byteSize = 8).
  // TODO: Add Complex128(cValue = 18, byteSize = 16).

  object Int8 extends DataType {
    override type ScalaType = Byte
    override implicit val supportedType = ByteIsSupportedType

    override val name    : String = "Int8"
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

  object Int16 extends DataType {
    override type ScalaType = Short
    override implicit val supportedType = ShortIsSupportedType

    override val name    : String = "Int16"
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

  object Int32 extends DataType {
    override type ScalaType = Int
    override implicit val supportedType = IntIsSupportedType

    override val name    : String = "Int32"
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

  object Int64 extends DataType {
    override type ScalaType = Long
    override implicit val supportedType = LongIsSupportedType

    override val name    : String = "Int64"
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

  object UInt8 extends DataType {
    override type ScalaType = UByte
    override implicit val supportedType = UByteIsSupportedType

    override val name    : String = "UInt8"
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

  object UInt16 extends DataType {
    override type ScalaType = UShort
    override implicit val supportedType = UShortIsSupportedType

    override val name    : String = "UInt16"
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

  object QInt8 extends DataType {
    override type ScalaType = Byte
    override implicit val supportedType = ByteIsSupportedType

    override val name    : String = "QInt8"
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

  object QInt16 extends DataType {
    override type ScalaType = Short
    override implicit val supportedType = ShortIsSupportedType

    override val name    : String = "QInt16"
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

  object QInt32 extends DataType {
    override type ScalaType = Int
    override implicit val supportedType = IntIsSupportedType

    override val name    : String = "QInt32"
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

  object QUInt8 extends DataType {
    override type ScalaType = UByte
    override implicit val supportedType = UByteIsSupportedType

    override val name    : String = "QUInt8"
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

  object QUInt16 extends DataType {
    override type ScalaType = UShort
    override implicit val supportedType = UShortIsSupportedType

    override val name    : String = "QUInt16"
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

  object Resource extends DataType {
    override type ScalaType = Long
    override implicit val supportedType = LongIsSupportedType

    override val name    : String = "Resource"
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

  //endregion Supported TensorFlow Data Types Definitions

  //region TensorFlow Data Type Sets

  /** Set of all floating-point data types. */
  val floatingPointDataTypes: Set[DataType] = {
    Set(Float32, Float64) // TODO: Float16, BFloat16.
  }

  /** Set of all complex data types. */
  val complexDataTypes: Set[DataType] = {
    Set() // TODO: [COMPLEX] Complex64, Complex128.
  }

  /** Set of all integer data types. */
  val integerDataTypes: Set[DataType] = {
    Set(Int8, Int16, Int32, Int64, UInt8, UInt16, QInt8, QInt16, QInt32, QUInt8, QUInt16)
  }

  /** Set of all quantized data types. */
  val quantizedDataTypes: Set[DataType] = {
    Set(QInt8, QInt16, QInt32, QUInt8, QUInt16) // TODO: BFloat16.
  }

  /** Set of all unsigned data types. */
  val unsignedDataTypes: Set[DataType] = {
    Set(UInt8, UInt16, QUInt8, QUInt16)
  }

  /** Set of all numeric data types. */
  val numericDataTypes: Set[DataType] = {
    floatingPointDataTypes ++ complexDataTypes ++ integerDataTypes ++ quantizedDataTypes
  }

  //endregion TensorFlow Data Type Sets

  //region Helper Methods

  /** Returns the [[DataType]] of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    */
  @inline private[api] def dataTypeOf[T: SupportedType](value: T): DataType = value.dataType

  // /** Returns the [[DataType]] of the provided [[Tensor]].
  //   *
  //   * @param  tensor Tensor whose data type to return.
  //   * @return Data type of the provided tensor.
  //   */
  // @inline private[api] def dataTypeOf(tensor: Tensor): DataType = tensor.dataType

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
    case Bool.cValue => Bool
    case Str.cValue => Str
    // case Float16.cValue => Float16
    case Float32.cValue => Float32
    case Float64.cValue => Float64
    // case BFloat16.cValue => BFloat16
    // case Complex64.cValue => Complex64
    // case Complex128.cValue => Complex128
    case Int8.cValue => Int8
    case Int16.cValue => Int16
    case Int32.cValue => Int32
    case Int64.cValue => Int64
    case UInt8.cValue => UInt8
    case UInt16.cValue => UInt16
    case QInt8.cValue => QInt8
    case QInt16.cValue => QInt16
    case QInt32.cValue => QInt32
    case QUInt8.cValue => QUInt8
    case QUInt16.cValue => QUInt16
    case Resource.cValue => Resource
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
    case "Boolean" => Bool
    case "String" => Str
    // case "Float16" => Float16
    case "Float32" => Float32
    case "Float64" => Float64
    // case "BFloat16" => BFloat16
    // case "Complex64" => Complex64
    // case "Complex128" => Complex128
    case "Int8" => Int8
    case "Int16" => Int16
    case "Int32" => Int32
    case "Int64" => Int64
    case "UInt8" => UInt8
    case "UInt16" => UInt16
    case "QInt8" => QInt8
    case "QInt16" => QInt16
    case "QInt32" => QInt32
    case "QUInt8" => QUInt8
    case "QUInt16" => QUInt16
    case "Resource" => Resource
    case value => throw new IllegalArgumentException(
      s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
  }

  //endregion Helper Methods
}
