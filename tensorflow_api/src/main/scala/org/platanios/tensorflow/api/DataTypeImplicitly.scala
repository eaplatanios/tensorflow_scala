package org.platanios.tensorflow.api

import java.nio.ByteBuffer

import org.platanios.tensorflow.api.Exception.InvalidCastException
import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

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
  def isBoolean: Boolean = this == DataType.Boolean

  //endregion Data Type Set Helper Methods

  /** Returns a data type that corresponds to this data type's real part. */
  def real: DataType = this match {
    case DataType.Complex64 => DataType.Float32
    case DataType.Complex128 => DataType.Float64
    case _ => this
  }

  /** Scala type corresponding to this TensorFlow data type. */
  type ScalaType

  private[api] implicit val scalaTypeImplicit: DataType.SupportedScalaType[ScalaType] = {
    implicitly[DataType.SupportedScalaType[ScalaType]]
  }

  /** Casts the provided value to this data type.
    *
    * Note that this method allows downcasting.
    *
    * @param  value Value to cast.
    * @return Casted value.
    */
  @inline def cast[T: DataType.SupportedScalaType](value: T): ScalaType = {
    implicitly[DataType.SupportedScalaType[T]].cast(value, this)
  }

  //  @inline def cast[T <: DataType.SupportedScalaTypes](value: T): ScalaType = cast(value)

  //region Specialized Scala Primitives Casting Methods

  @inline protected def cast(value: Float): ScalaType =
    throw InvalidCastException(s"'Float' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Double): ScalaType =
    throw InvalidCastException(s"'Double' cannot be cast to 'DataType.$name'.")
  // @inline protected def cast(value: Complex64): ScalaType =
  //   throw InvalidCastException(s"'Complex64' cannot be cast to 'DataType.$name'.")
  // @inline protected def cast(value: Complex128): ScalaType =
  //   throw InvalidCastException(s"'Complex128' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Byte): ScalaType =
  throw InvalidCastException(s"'Byte' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Short): ScalaType =
    throw InvalidCastException(s"'Short' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Int): ScalaType =
    throw InvalidCastException(s"'Int' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Long): ScalaType =
    throw InvalidCastException(s"'Long' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Char): ScalaType =
    throw InvalidCastException(s"'Char' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: Boolean): ScalaType =
    throw InvalidCastException(s"'Boolean' cannot be cast to 'DataType.$name'.")
  @inline protected def cast(value: String): ScalaType =
    throw InvalidCastException(s"'String' cannot be cast to 'DataType.$name'.")

  //endregion Specialized Scala Primitives Casting Methods

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
  @annotation.implicitNotFound(msg = "Scala type '${T}' cannot be converted to a TensorFlow data type.")
  trait SupportedScalaType[T] {
    @inline private[DataType] val dataType: DataType
    @inline private[DataType] def cast(value: T, dataType: DataType): dataType.ScalaType
  }

  //region Supported Scala Type Implicits

  private[api] implicit val floatScalaType: SupportedScalaType[Float] = new SupportedScalaType[Float] {
    @inline private[DataType] override val dataType: DataType = DataType.Float32
    @inline private[DataType] override def cast(value: Float, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  private[api] implicit val doubleScalaType: SupportedScalaType[Double] = new SupportedScalaType[Double] {
    @inline private[DataType] override val dataType: DataType = DataType.Float64
    @inline private[DataType] override def cast(value: Double, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

//  private[api] implicit val complexScalaType: SupportedScalaType[Complex[_]] = new SupportedScalaType[Complex[_]] {
//    @inline private[DataType] override def cast(value: Complex[_], dataType: DataType): dataType.ScalaType = {
//      dataType.cast(value)
//    }
//  }

  private[api] implicit val byteScalaType: SupportedScalaType[Byte] = new SupportedScalaType[Byte] {
    @inline private[DataType] override val dataType: DataType = DataType.Int8
    @inline private[DataType] override def cast(value: Byte, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  private[api] implicit val shortScalaType: SupportedScalaType[Short] = new SupportedScalaType[Short] {
    @inline private[DataType] override val dataType: DataType = DataType.Int16
    @inline private[DataType] override def cast(value: Short, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  private[api] implicit val intScalaType: SupportedScalaType[Int] = new SupportedScalaType[Int] {
    @inline private[DataType] override val dataType: DataType = DataType.Int32
    @inline private[DataType] override def cast(value: Int, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  private[api] implicit val longScalaType: SupportedScalaType[Long] = new SupportedScalaType[Long] {
    @inline private[DataType] override val dataType: DataType = DataType.Int64
    @inline private[DataType] override def cast(value: Long, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

//  private[api] implicit val ubyteScalaType: SupportedScalaType[UByte] = new SupportedScalaType[UByte] {
//    @inline private[DataType] override val dataType: DataType = DataType.UInt8
//    @inline private[DataType] override def cast(value: UByte, dataType: DataType): dataType.ScalaType = {
//      dataType.cast(value.toByte)
//    }
//  }
//
//  private[api] implicit val ushortScalaType: SupportedScalaType[UShort] = new SupportedScalaType[UShort] {
//    @inline private[DataType] override val dataType: DataType = DataType.UInt16
//    @inline private[DataType] override def cast(value: UShort, dataType: DataType): dataType.ScalaType = {
//      dataType.cast(value.toShort)
//    }
//  }

  private[api] implicit val charScalaType: SupportedScalaType[Char] = new SupportedScalaType[Char] {
    @inline private[DataType] override val dataType: DataType = DataType.UInt16
    @inline private[DataType] override def cast(value: Char, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  private[api] implicit val booleanScalaType: SupportedScalaType[Boolean] = new SupportedScalaType[Boolean] {
    @inline private[DataType] override val dataType: DataType = DataType.Boolean
    @inline private[DataType] override def cast(value: Boolean, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  private[api] implicit val stringScalaType: SupportedScalaType[String] = new SupportedScalaType[String] {
    @inline private[DataType] override val dataType: DataType = DataType.String
    @inline private[DataType] override def cast(value: String, dataType: DataType): dataType.ScalaType = {
      dataType.cast(value)
    }
  }

  //endregion Supported Scala Type Implicits

  //region Supported TensorFlow Data Types Definitions

  object Float16 extends DataType {
    override val name: String = "Float16"
    override val cValue: Int = 19
    override val byteSize: Int = 2

    override type ScalaType = Float // TODO: What data type should we actually use for this?

    @inline protected override def cast(value: Float): ScalaType = value
    @inline protected override def cast(value: Double): ScalaType = value.toFloat
    @inline protected override def cast(value: Byte): ScalaType = value.toFloat
    @inline protected override def cast(value: Short): ScalaType = value.toFloat
    @inline protected override def cast(value: Int): ScalaType = value.toFloat
    @inline protected override def cast(value: Long): ScalaType = value.toFloat
    @inline protected override def cast(value: Char): ScalaType = value.toFloat
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.0f else 0.0f

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element.asInstanceOf[Short]) // TODO: Something is off here.
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getShort(index).asInstanceOf[Float] // TODO: Something is off here.
    }
  }

  object Float32 extends DataType {
    override val name: String = "Float32"
    override val cValue: Int = 1
    override val byteSize: Int = 4

    override type ScalaType = Float

    @inline protected override def cast(value: Float): ScalaType = value
    @inline protected override def cast(value: Double): ScalaType = value.toFloat
    @inline protected override def cast(value: Byte): ScalaType = value.toFloat
    @inline protected override def cast(value: Short): ScalaType = value.toFloat
    @inline protected override def cast(value: Int): ScalaType = value.toFloat
    @inline protected override def cast(value: Long): ScalaType = value.toFloat
    @inline protected override def cast(value: Char): ScalaType = value.toFloat
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.0f else 0.0f

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putFloat(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getFloat(index)
    }
  }

  object Float64 extends DataType {
    override val name: String = "Float64"
    override val cValue: Int = 2
    override val byteSize: Int = 8

    override type ScalaType = Double

    @inline protected override def cast(value: Float): ScalaType = value.toDouble
    @inline protected override def cast(value: Double): ScalaType = value
    @inline protected override def cast(value: Byte): ScalaType = value.toDouble
    @inline protected override def cast(value: Short): ScalaType = value.toDouble
    @inline protected override def cast(value: Int): ScalaType = value.toDouble
    @inline protected override def cast(value: Long): ScalaType = value.toDouble
    @inline protected override def cast(value: Char): ScalaType = value.toDouble
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.0 else 0.0

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putDouble(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getDouble(index)
    }
  }

  object BFloat16 extends DataType {
    override val name: String = "BFloat16"
    override val cValue: Int = 14
    override val byteSize: Int = 2

    override type ScalaType = Float // TODO: What data type should we actually use for this?

    @inline protected override def cast(value: Float): ScalaType = value
    @inline protected override def cast(value: Double): ScalaType = value.toFloat
    @inline protected override def cast(value: Byte): ScalaType = value.toFloat
    @inline protected override def cast(value: Short): ScalaType = value.toFloat
    @inline protected override def cast(value: Int): ScalaType = value.toFloat
    @inline protected override def cast(value: Long): ScalaType = value.toFloat
    @inline protected override def cast(value: Char): ScalaType = value.toFloat
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.0f else 0.0f

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element.asInstanceOf[Short]) // TODO: Something is off here.
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getShort(index).asInstanceOf[Float] // TODO: Something is off here.
    }
  }

  object Complex64 extends DataType {
    override val name: String = "Complex64"
    override val cValue: Int = 8
    override val byteSize: Int = 8

    override type ScalaType = Float

    // @inline protected override def cast(value: Float): ScalaType = Complex[Float](value, 0.0f)
    // @inline protected override def cast(value: Double): ScalaType = Complex[Float](value.toFloat, 0.0f)
    // @inline protected override def cast(value: Complex64): ScalaType = value
    // @inline protected override def cast(value: Complex128): ScalaType = Complex[Float](value.real.toFloat, value.imag.toFloat)
    // @inline protected override def cast(value: Byte): ScalaType = Complex[Float](value.toFloat, 0.0f)
    // @inline protected override def cast(value: Short): ScalaType = Complex[Float](value.toFloat, 0.0f)
    // @inline protected override def cast(value: Int): ScalaType = Complex[Float](value.toFloat, 0.0f)
    // @inline protected override def cast(value: Long): ScalaType = Complex[Float](value.toFloat, 0.0f)
    // @inline protected override def cast(value: Char): ScalaType = Complex[Float](value.toFloat, 0.0f)
    // @inline protected override def cast(value: Boolean): ScalaType = Complex[Float](if (value) 1.0f else 0.0f, 0.0f)

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = ???
    // {
    //   buffer.putFloat(index, element.real)
    //   buffer.putFloat(index + 4, element.imag)
    // }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = ???
    // {
    //   Complex[Float](real = buffer.getFloat(index), imag = buffer.getFloat(index + 4))
    // }
  }

  object Complex128 extends DataType {
    override val name: String = "Complex128"
    override val cValue: Int = 18
    override val byteSize: Int = 16

    override type ScalaType = Double

    // @inline protected override def cast(value: Float): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Double): ScalaType = Complex[Double](value, 0.0)
    // @inline protected override def cast(value: Complex64): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Complex128): ScalaType = value
    // @inline protected override def cast(value: Byte): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Short): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Int): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Long): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Char): ScalaType = Complex[Double](value.toDouble, 0.0)
    // @inline protected override def cast(value: Boolean): ScalaType = Complex[Double](if (value) 1.0 else 0.0, 0.0)

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = ???
    // {
    //   buffer.putDouble(index, element.real)
    //   buffer.putDouble(index + 8, element.imag)
    // }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = ???
    // {
    //   Complex[Double](real = buffer.getDouble(index), imag = buffer.getDouble(index + 8))
    // }
  }

  object Int8 extends DataType {
    override val name: String = "Int8"
    override val cValue: Int = 6
    override val byteSize: Int = 1

    override type ScalaType = Byte

    @inline protected override def cast(value: Float): ScalaType = value.toByte
    @inline protected override def cast(value: Double): ScalaType = value.toByte
    @inline protected override def cast(value: Byte): ScalaType = value
    @inline protected override def cast(value: Short): ScalaType = value.toByte
    @inline protected override def cast(value: Int): ScalaType = value.toByte
    @inline protected override def cast(value: Long): ScalaType = value.toByte
    @inline protected override def cast(value: Char): ScalaType = value.toByte
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toByte else 0.toByte

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.get(index)
    }
  }

  object Int16 extends DataType {
    override val name: String = "Int16"
    override val cValue: Int = 5
    override val byteSize: Int = 2

    override type ScalaType = Short

    @inline protected override def cast(value: Float): ScalaType = value.toShort
    @inline protected override def cast(value: Double): ScalaType = value.toShort
    @inline protected override def cast(value: Byte): ScalaType = value.toShort
    @inline protected override def cast(value: Short): ScalaType = value
    @inline protected override def cast(value: Int): ScalaType = value.toShort
    @inline protected override def cast(value: Long): ScalaType = value.toShort
    @inline protected override def cast(value: Char): ScalaType = value.toShort
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toShort else 0.toShort

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getShort(index)
    }
  }

  object Int32 extends DataType {
    override val name: String = "Int32"
    override val cValue: Int = 3
    override val byteSize: Int = 4

    override type ScalaType = Int

    @inline protected override def cast(value: Float): ScalaType = value.toInt
    @inline protected override def cast(value: Double): ScalaType = value.toInt
    @inline protected override def cast(value: Byte): ScalaType = value.toInt
    @inline protected override def cast(value: Short): ScalaType = value.toInt
    @inline protected override def cast(value: Int): ScalaType = value
    @inline protected override def cast(value: Long): ScalaType = value.toInt
    @inline protected override def cast(value: Char): ScalaType = value.toInt
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1 else 0

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putInt(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getInt(index)
    }
  }

  object Int64 extends DataType {
    override val name: String = "Int64"
    override val cValue: Int = 9
    override val byteSize: Int = 8

    override type ScalaType = Long

    @inline protected override def cast(value: Float): ScalaType = value.toLong
    @inline protected override def cast(value: Double): ScalaType = value.toLong
    @inline protected override def cast(value: Byte): ScalaType = value.toLong
    @inline protected override def cast(value: Short): ScalaType = value.toLong
    @inline protected override def cast(value: Int): ScalaType = value.toLong
    @inline protected override def cast(value: Long): ScalaType = value
    @inline protected override def cast(value: Char): ScalaType = value.toLong
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1L else 0L

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putLong(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getLong(index)
    }
  }

  object UInt8 extends DataType {
    override val name: String = "UInt8"
    override val cValue: Int = 4
    override val byteSize: Int = 1

    override type ScalaType = Byte

    @inline protected override def cast(value: Float): ScalaType = value.toByte
    @inline protected override def cast(value: Double): ScalaType = value.toByte
    @inline protected override def cast(value: Byte): ScalaType = value
    @inline protected override def cast(value: Short): ScalaType = value.toByte
    @inline protected override def cast(value: Int): ScalaType = value.toByte
    @inline protected override def cast(value: Long): ScalaType = value.toByte
    @inline protected override def cast(value: Char): ScalaType = value.toByte
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toByte else 0.toByte

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.get(index)
    }
  }

  object UInt16 extends DataType {
    override val name: String = "UInt16"
    override val cValue: Int = 17
    override val byteSize: Int = 2

    override type ScalaType = Short

    @inline protected override def cast(value: Float): ScalaType = value.toShort
    @inline protected override def cast(value: Double): ScalaType = value.toShort
    @inline protected override def cast(value: Byte): ScalaType = value.toShort
    @inline protected override def cast(value: Short): ScalaType = value
    @inline protected override def cast(value: Int): ScalaType = value.toShort
    @inline protected override def cast(value: Long): ScalaType = value.toShort
    @inline protected override def cast(value: Char): ScalaType = value.toShort
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toShort else 0.toShort

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getShort(index)
    }
  }

  object QInt8 extends DataType {
    override val name: String = "QInt8"
    override val cValue: Int = 11
    override val byteSize: Int = 1

    override type ScalaType = Byte

    @inline protected override def cast(value: Float): ScalaType = value.toByte
    @inline protected override def cast(value: Double): ScalaType = value.toByte
    @inline protected override def cast(value: Byte): ScalaType = value
    @inline protected override def cast(value: Short): ScalaType = value.toByte
    @inline protected override def cast(value: Int): ScalaType = value.toByte
    @inline protected override def cast(value: Long): ScalaType = value.toByte
    @inline protected override def cast(value: Char): ScalaType = value.toByte
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toByte else 0.toByte

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.get(index)
    }
  }

  object QInt16 extends DataType {
    override val name: String = "QInt16"
    override val cValue: Int = 15
    override val byteSize: Int = 2

    override type ScalaType = Short

    @inline protected override def cast(value: Float): ScalaType = value.toShort
    @inline protected override def cast(value: Double): ScalaType = value.toShort
    @inline protected override def cast(value: Byte): ScalaType = value.toShort
    @inline protected override def cast(value: Short): ScalaType = value
    @inline protected override def cast(value: Int): ScalaType = value.toShort
    @inline protected override def cast(value: Long): ScalaType = value.toShort
    @inline protected override def cast(value: Char): ScalaType = value.toShort
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toShort else 0.toShort

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getShort(index)
    }
  }

  object QInt32 extends DataType {
    override val name: String = "QInt32"
    override val cValue: Int = 13
    override val byteSize: Int = 4

    override type ScalaType = Int

    @inline protected override def cast(value: Float): ScalaType = value.toInt
    @inline protected override def cast(value: Double): ScalaType = value.toInt
    @inline protected override def cast(value: Byte): ScalaType = value.toInt
    @inline protected override def cast(value: Short): ScalaType = value.toInt
    @inline protected override def cast(value: Int): ScalaType = value
    @inline protected override def cast(value: Long): ScalaType = value.toInt
    @inline protected override def cast(value: Char): ScalaType = value.toInt
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1 else 0

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putInt(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getInt(index)
    }
  }

  object QUInt8 extends DataType {
    override val name: String = "QUInt8"
    override val cValue: Int = 12
    override val byteSize: Int = 1

    override type ScalaType = Byte

    @inline protected override def cast(value: Float): ScalaType = value.toByte
    @inline protected override def cast(value: Double): ScalaType = value.toByte
    @inline protected override def cast(value: Byte): ScalaType = value
    @inline protected override def cast(value: Short): ScalaType = value.toByte
    @inline protected override def cast(value: Int): ScalaType = value.toByte
    @inline protected override def cast(value: Long): ScalaType = value.toByte
    @inline protected override def cast(value: Char): ScalaType = value.toByte
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toByte else 0.toByte

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.get(index)
    }
  }

  object QUInt16 extends DataType {
    override val name: String = "QUInt16"
    override val cValue: Int = 16
    override val byteSize: Int = 2

    override type ScalaType = Short

    @inline protected override def cast(value: Float): ScalaType = value.toShort
    @inline protected override def cast(value: Double): ScalaType = value.toShort
    @inline protected override def cast(value: Byte): ScalaType = value.toShort
    @inline protected override def cast(value: Short): ScalaType = value
    @inline protected override def cast(value: Int): ScalaType = value.toShort
    @inline protected override def cast(value: Long): ScalaType = value.toShort
    @inline protected override def cast(value: Char): ScalaType = value.toShort
    @inline protected override def cast(value: Boolean): ScalaType = if (value) 1.toShort else 0.toShort

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.putShort(index, element)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.getShort(index)
    }
  }

  object Boolean extends DataType {
    override val name: String = "Boolean"
    override val cValue: Int = 10
    override val byteSize: Int = 1

    override type ScalaType = Boolean

    @inline protected override def cast(value: Boolean): ScalaType = value

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = {
      buffer.put(index, if (element) 1 else 0)
    }

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = {
      buffer.get(index) == 1
    }
  }

  object String extends DataType {
    override val name: String = "String"
    override val cValue: Int = 7
    override val byteSize: Int = -1

    override type ScalaType = String

    @inline protected override def cast(value: Float): ScalaType = value.toString
    @inline protected override def cast(value: Double): ScalaType = value.toString
    @inline protected override def cast(value: Byte): ScalaType = value.toString
    @inline protected override def cast(value: Short): ScalaType = value.toString
    @inline protected override def cast(value: Int): ScalaType = value.toString
    @inline protected override def cast(value: Long): ScalaType = value.toString
    @inline protected override def cast(value: Char): ScalaType = value.toString
    @inline protected override def cast(value: Boolean): ScalaType = value.toString
    @inline protected override def cast(value: String): ScalaType = value

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit = ???

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType = ???
  }

  object Resource extends DataType {
    override val name: String = "Resource"
    override val cValue: Int = 20
    override val byteSize: Int = -1

    override type ScalaType = Long // TODO: Not supported on the Scala side.

    override def putElementInBuffer(buffer: ByteBuffer, index: Int, element: ScalaType): Unit =
      throw InvalidCastException(s"Tensors with data type '$name' cannot be represented in the Scala API. You should " +
                                     s"never need to feed or fetch '$name' tensors.")

    override def getElementFromBuffer(buffer: ByteBuffer, index: Int): ScalaType =
      throw InvalidCastException(s"Tensors with data type '$name' cannot be represented in the Scala API. You should " +
                                     s"never need to feed or fetch '$name' tensors.")
  }

  //endregion Supported TensorFlow Data Types Definitions

  //region TensorFlow Data Type Sets

  /** Set of all floating-point data types. */
  val floatingPointDataTypes: Set[DataType] = {
    Set(Float16, Float32, Float64, BFloat16)
  }

  /** Set of all complex data types. */
  val complexDataTypes: Set[DataType] = {
    Set(Complex64, Complex128)
  }

  /** Set of all integer data types. */
  val integerDataTypes: Set[DataType] = {
    Set(Int8, Int16, Int32, Int64, UInt8, UInt16, QInt8, QInt16, QInt32, QUInt8, QUInt16)
  }

  /** Set of all quantized data types. */
  val quantizedDataTypes: Set[DataType] = {
    Set(BFloat16, QInt8, QInt16, QInt32, QUInt8, QUInt16)
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
  @inline private[api] def dataTypeOf[T: DataType.SupportedScalaType](value: T): DataType = {
    implicitly[DataType.SupportedScalaType[T]].dataType
  }

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
    case Float16.cValue => Float16
    case Float32.cValue => Float32
    case Float64.cValue => Float64
    case BFloat16.cValue => BFloat16
    case Complex64.cValue => Complex64
    case Complex128.cValue => Complex128
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
    case Boolean.cValue => Boolean
    case String.cValue => String
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
    case "Float16" => Float16
    case "Float32" => Float32
    case "Float64" => Float64
    case "BFloat16" => BFloat16
    case "Complex64" => Complex64
    case "Complex128" => Complex128
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
    case "Boolean" => Boolean
    case "String" => String
    case "Resource" => Resource
    case value => throw new IllegalArgumentException(
      s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
  }

  //endregion Helper Methods
}
