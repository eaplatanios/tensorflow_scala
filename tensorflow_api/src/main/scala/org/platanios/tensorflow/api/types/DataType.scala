package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.Exception.InvalidCastException
import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

import java.nio.ByteBuffer

// TODO: Figure out how to build a type hierarchy to use when constructing tensors.
// TODO: Improve handling of the String data type (e.g., in the dataTypeOf function).
// TODO: Add support for unsigned numbers and for complex numbers.
// TODO: How to issue a warning/error when negative values are fed into unsigned types.
// TODO: Is spire necessary?
// TODO: Casts are unsafe (i.e., downcasting is allowed).
// TODO: Figure out how to deal with reference data types.

// TODO: Unstable types: Float16, BFloat16, Complex, and UInts.

/** Represents the data type of the elements in a tensor.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataType {
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

  //endregion Data Type Properties

  //region Data Type Set Helper Methods

//  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
//  def isFloatingPoint: Boolean = !isQuantized && DataType.floatingPointDataTypes.contains(this)
//
//  /** Returns `true` if this data type represents a complex data types. */
//  def isComplex: Boolean = DataType.complexDataTypes.contains(this)
//
//  /** Returns `true` if this data type represents a non-quantized integer data type. */
//  def isInteger: Boolean = !isQuantized && DataType.integerDataTypes.contains(this)
//
//  /** Returns `true` if this data type represents a quantized data type. */
//  def isQuantized: Boolean = DataType.quantizedDataTypes.contains(this)
//
//  /** Returns `true` if this data type represents a non-quantized unsigned data type. */
//  def isUnsigned: Boolean = !isQuantized && DataType.unsignedDataTypes.contains(this)
//
//  /** Returns `true` if this data type represents a numeric data type. */
//  def isNumeric: Boolean = DataType.numericDataTypes.contains(this)
  def isNumeric: Boolean = true

  /** Returns `true` if this data type represents a boolean data type. */
  def isBoolean: Boolean = this == DataType.Bool

  //endregion Data Type Set Helper Methods

//  /** Returns a data type that corresponds to this data type's real part. */
//  def real: DataType = this match {
//    case DataType.Complex64 => DataType.Float32
//    case DataType.Complex128 => DataType.Float64
//    case _ => this
//  }

  /** Scala type corresponding to this TensorFlow data type. */
  type ScalaType <: SupportedScalaType

  /** Casts the provided value to this data type.
    *
    * Note that this method allows downcasting.
    *
    * @param  value Value to cast.
    * @return Casted value.
    */
  @inline def cast(value: SupportedScalaType): ScalaType = value.cast(this)

  def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit
  def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType

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

//  object Float16 extends DataType {
//    override val name    : String = "Float16"
//    override val cValue  : Int    = 19
//    override val byteSize: Int    = 2
//
//    override type ScalaType = Float32 // TODO: What data type should we actually use for this?
//
//    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
//      buffer.putFloat(index, element)
//    }
//
//    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
//      org.platanios.tensorflow.api.types.Float32(buffer.getFloat(index))
//    }
//  }

  object Float32 extends DataType {
    override val name    : String = "Float32"
    override val cValue  : Int    = 1
    override val byteSize: Int    = 4

    override type ScalaType = Float32

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putFloat(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Float32(buffer.getFloat(index))
    }
  }

  object Float64 extends DataType {
    override val name    : String = "Float64"
    override val cValue  : Int    = 2
    override val byteSize: Int    = 8

    override type ScalaType = Float64

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putDouble(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Float64(buffer.getDouble(index))
    }
  }

  // TODO: Add Complex64(cValue = 8, byteSize = 8).
  // TODO: Add Complex128(cValue = 18, byteSize = 16).

//  object BFloat16 extends DataType {
//    override val name    : String = "BFloat16"
//    override val cValue  : Int    = 14
//    override val byteSize: Int    = 2
//
//    override type ScalaType = Float32 // TODO: What data type should we actually use for this?
//
//    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
//      buffer.putFloat(index, element)
//    }
//
//    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
//      org.platanios.tensorflow.api.types.Float32(buffer.getFloat(index))
//    }
//  }

  object Int8 extends DataType {
    override val name    : String = "Int8"
    override val cValue  : Int    = 6
    override val byteSize: Int    = 1

    override type ScalaType = Int8

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int8(buffer.get(index))
    }
  }

  object Int16 extends DataType {
    override val name    : String = "Int16"
    override val cValue  : Int    = 5
    override val byteSize: Int    = 2

    override type ScalaType = Int16

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int16(buffer.getShort(index))
    }
  }

  object Int32 extends DataType {
    override val name    : String = "Int32"
    override val cValue  : Int    = 3
    override val byteSize: Int    = 4

    override type ScalaType = Int32

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putInt(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int32(buffer.getInt(index))
    }
  }

  object Int64 extends DataType {
    override val name    : String = "Int64"
    override val cValue  : Int    = 9
    override val byteSize: Int    = 8

    override type ScalaType = Int64

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putLong(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int64(buffer.getLong(index))
    }
  }

//  object UInt8 extends DataType {
//    override val name    : String = "UInt8"
//    override val cValue  : Int    = 4
//    override val byteSize: Int    = 1
//
//    override type ScalaType = UInt16
//
//    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
//      buffer.putChar(index, element)
//    }
//
//    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
//      org.platanios.tensorflow.api.types.UInt16(buffer.getChar(index))
//    }
//  }

  object UInt16 extends DataType {
    override val name    : String = "UInt16"
    override val cValue  : Int    = 17
    override val byteSize: Int    = 2

    override type ScalaType = UInt16

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putChar(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.UInt16(buffer.getChar(index))
    }
  }

  object QInt8 extends DataType {
    override val name    : String = "QInt8"
    override val cValue  : Int    = 11
    override val byteSize: Int    = 1

    override type ScalaType = Int8

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int8(buffer.get(index))
    }
  }

  object QInt16 extends DataType {
    override val name    : String = "QInt16"
    override val cValue  : Int    = 15
    override val byteSize: Int    = 2

    override type ScalaType = Int16

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int16(buffer.getShort(index))
    }
  }

  object QInt32 extends DataType {
    override val name    : String = "QInt32"
    override val cValue  : Int    = 13
    override val byteSize: Int    = 4

    override type ScalaType = Int32

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putInt(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Int32(buffer.getInt(index))
    }
  }

//  object QUInt8 extends DataType {
//    override val name    : String = "QUInt8"
//    override val cValue  : Int    = 12
//    override val byteSize: Int    = 1
//
//    override type ScalaType = UInt16
//
//    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
//      buffer.putChar(index, element)
//    }
//
//    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
//      org.platanios.tensorflow.api.types.UInt16(buffer.getChar(index))
//    }
//  }

  object QUInt16 extends DataType {
    override val name    : String = "QUInt16"
    override val cValue  : Int    = 16
    override val byteSize: Int    = 2

    override type ScalaType = UInt16

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putChar(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.UInt16(buffer.getChar(index))
    }
  }

  object Bool extends DataType {
    override val name    : String = "Bool"
    override val cValue  : Int    = 10
    override val byteSize: Int    = 1

    override type ScalaType = Bool

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, if (element) 1 else 0)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      org.platanios.tensorflow.api.types.Bool(buffer.get(index) == 1)
    }
  }

  // TODO: Add String(cValue = 7, byteSize = -1).
  // TODO: Add Resource(cValue = 20, byteSize = -1).

  //endregion Supported TensorFlow Data Types Definitions

  //region TensorFlow Data Type Sets

//  /** Set of all floating-point data types. */
//  val floatingPointDataTypes: Set[DataType] = {
//    Set(Float16, Float32, Float64, BFloat16)
//  }
//
//  /** Set of all complex data types. */
//  val complexDataTypes: Set[DataType] = {
//    Set(Complex64, Complex128)
//  }
//
//  /** Set of all integer data types. */
//  val integerDataTypes: Set[DataType] = {
//    Set(Int8, Int16, Int32, Int64, UInt8, UInt16, QInt8, QInt16, QInt32, QUInt8, QUInt16)
//  }
//
//  /** Set of all quantized data types. */
//  val quantizedDataTypes: Set[DataType] = {
//    Set(BFloat16, QInt8, QInt16, QInt32, QUInt8, QUInt16)
//  }
//
//  /** Set of all unsigned data types. */
//  val unsignedDataTypes: Set[DataType] = {
//    Set(UInt8, UInt16, QUInt8, QUInt16)
//  }
//
//  /** Set of all numeric data types. */
//  val numericDataTypes: Set[DataType] = {
//    floatingPointDataTypes ++ complexDataTypes ++ integerDataTypes ++ quantizedDataTypes
//  }

  //endregion TensorFlow Data Type Sets

  //region Helper Methods

  /** Returns the [[DataType]] of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    */
  @inline private[api] def dataTypeOf(value: SupportedScalaType): DataType = value.dataType

  //  /** Returns the [[DataType]] of the provided [[Tensor]].
  //    *
  //    * @param  tensor Tensor whose data type to return.
  //    * @return Data type of the provided tensor.
  //    */
  //  @inline private[api] def dataTypeOf(tensor: Tensor): DataType = tensor.dataType

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
//    case Float16.cValue => Float16
    case Float32.cValue => Float32
//    case Float64.cValue => Float64
//    case BFloat16.cValue => BFloat16
//    case Complex64.cValue => Complex64
//    case Complex128.cValue => Complex128
//    case Int8.cValue => Int8
//    case Int16.cValue => Int16
//    case Int32.cValue => Int32
//    case Int64.cValue => Int64
//    case UInt8.cValue => UInt8
//    case UInt16.cValue => UInt16
//    case QInt8.cValue => QInt8
//    case QInt16.cValue => QInt16
//    case QInt32.cValue => QInt32
//    case QUInt8.cValue => QUInt8
//    case QUInt16.cValue => QUInt16
//    case Boolean.cValue => Boolean
//    case String.cValue => String
//    case Resource.cValue => Resource
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
//    case "Float16" => Float16
    case "Float32" => Float32
//    case "Float64" => Float64
//    case "BFloat16" => BFloat16
//    case "Complex64" => Complex64
//    case "Complex128" => Complex128
//    case "Int8" => Int8
//    case "Int16" => Int16
//    case "Int32" => Int32
//    case "Int64" => Int64
//    case "UInt8" => UInt8
//    case "UInt16" => UInt16
//    case "QInt8" => QInt8
//    case "QInt16" => QInt16
//    case "QInt32" => QInt32
//    case "QUInt8" => QUInt8
//    case "QUInt16" => QUInt16
//    case "Boolean" => Boolean
//    case "String" => String
//    case "Resource" => Resource
    case value => throw new IllegalArgumentException(
      s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
  }

  //endregion Helper Methods
}
