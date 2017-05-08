package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.Exception.InvalidDataTypeException

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedType[@specialized T] extends Any {
  @inline def dataType: DataType = {
    throw InvalidDataTypeException("The Scala type of this data type is not supported.")
  }

  @throws[UnsupportedOperationException]
  @inline def cast[R: SupportedType](value: R, dataType: DataType): T = {
    throw InvalidDataTypeException("The Scala type of this data type is not supported.")
  }

  private[this] def invalidConversionException(value: T, t: String): Exception = {
    new IllegalArgumentException(s"Value '$value' cannot be converted to $t.")
  }

  def toBoolean(value: T): Boolean = throw invalidConversionException(value, "a boolean")
  def toFloat(value: T): Float = throw invalidConversionException(value, "a float")
  def toDouble(value: T): Double = throw invalidConversionException(value, "a double")
  def toByte(value: T): Byte = throw invalidConversionException(value, "a byte")
  def toShort(value: T): Short = throw invalidConversionException(value, "a short")
  def toInt(value: T): Int = throw invalidConversionException(value, "an int")
  def toLong(value: T): Long = throw invalidConversionException(value, "a long")
  def toUByte(value: T): UByte = throw invalidConversionException(value, "an unsigned byte")
  def toUShort(value: T): UShort = throw invalidConversionException(value, "an unsigned short")

  def toString(value: T): String = value.toString
}

object SupportedType {
  class SupportedTypeOps[T](val value: T) extends AnyVal {
    def dataType(implicit evidence: SupportedType[T]): DataType = evidence.dataType

    def cast(dataType: DataType)
        (implicit evidence: SupportedType[T],
            dataTypeEvidence: SupportedType[dataType.ScalaType]): dataType.ScalaType = {
      dataTypeEvidence.cast(value, dataType)
    }

    def toBoolean(implicit evidence: SupportedType[T]): Boolean = evidence.toBoolean(value)
    def toFloat(implicit evidence: SupportedType[T]): Float = evidence.toFloat(value)
    def toDouble(implicit evidence: SupportedType[T]): Double = evidence.toDouble(value)
    def toByte(implicit evidence: SupportedType[T]): Byte = evidence.toByte(value)
    def toShort(implicit evidence: SupportedType[T]): Short = evidence.toShort(value)
    def toInt(implicit evidence: SupportedType[T]): Int = evidence.toInt(value)
    def toLong(implicit evidence: SupportedType[T]): Long = evidence.toLong(value)
    def toUByte(implicit evidence: SupportedType[T]): UByte = evidence.toUByte(value)
    def toUShort(implicit evidence: SupportedType[T]): UShort = evidence.toUShort(value)
  }

  @inline final def apply[T](implicit evidence: SupportedType[T]): SupportedType[T] = evidence

  private[types] class BooleanIsSupportedType extends SupportedType[Boolean] {
    override def dataType: DataType = DataType.Bool
    override def cast[R: SupportedType](value: R, dataType: DataType): Boolean = value match {
      case value: Boolean => value
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a boolean.")
    }

    override def toBoolean(value: Boolean): Boolean = value
    override def toFloat(value: Boolean): Float = if (value) 1.0f else 0.0f
    override def toDouble(value: Boolean): Double = if (value) 1.0 else 0.0
    override def toByte(value: Boolean): Byte = if (value) 1 else 0
    override def toShort(value: Boolean): Short = if (value) 1 else 0
    override def toInt(value: Boolean): Int = if (value) 1 else 0
    override def toLong(value: Boolean): Long = if (value) 1L else 0L
    override def toUByte(value: Boolean): UByte = if (value) UByte(1) else UByte(0)
    override def toUShort(value: Boolean): UShort = if (value) UShort(1) else UShort(0)
  }

  private[types] class StringIsSupportedType extends SupportedType[String] {
    override def dataType: DataType = DataType.Str
    override def cast[R: SupportedType](value: R, dataType: DataType): String = value.toString

    override def toString(value: String): String = value
  }

  private[types] class FloatIsSupportedType extends SupportedType[Float] {
    override def dataType: DataType = DataType.Float32
    override def cast[R: SupportedType](value: R, dataType: DataType): Float = value match {
      case value: Boolean => if (value) 1.0f else 0.0f
      case value: Float => value.toFloat
      case value: Double => value.toFloat
      case value: Byte => value.toFloat
      case value: Short => value.toFloat
      case value: Int => value.toFloat
      case value: Long => value.toFloat
      case value: UByte => value.toFloat
      case value: UShort => value.toFloat
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a float.")
    }

    override def toFloat(value: Float): Float = value
    override def toDouble(value: Float): Double = value.toDouble
    override def toByte(value: Float): Byte = value.toByte
    override def toShort(value: Float): Short = value.toShort
    override def toInt(value: Float): Int = value.toInt
    override def toLong(value: Float): Long = value.toLong
    override def toUByte(value: Float): UByte = UByte(value.toByte)
    override def toUShort(value: Float): UShort = UShort(value.toChar)
  }

  private[types] class DoubleIsSupportedType extends SupportedType[Double] {
    override def dataType: DataType = DataType.Float64
    override def cast[R: SupportedType](value: R, dataType: DataType): Double = value match {
      case value: Boolean => if (value) 1.0 else 0.0
      case value: Float => value.toDouble
      case value: Double => value.toDouble
      case value: Byte => value.toDouble
      case value: Short => value.toDouble
      case value: Int => value.toDouble
      case value: Long => value.toDouble
      case value: UByte => value.toDouble
      case value: UShort => value.toDouble
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a double.")
    }

    override def toFloat(value: Double): Float = value.toFloat
    override def toDouble(value: Double): Double = value
    override def toByte(value: Double): Byte = value.toByte
    override def toShort(value: Double): Short = value.toShort
    override def toInt(value: Double): Int = value.toInt
    override def toLong(value: Double): Long = value.toLong
    override def toUByte(value: Double): UByte = UByte(value.toByte)
    override def toUShort(value: Double): UShort = UShort(value.toChar)
  }

  private[types] class ByteIsSupportedType extends SupportedType[Byte] {
    override def dataType: DataType = DataType.Int8
    override def cast[R: SupportedType](value: R, dataType: DataType): Byte = value match {
      case value: Boolean => if (value) 1 else 0
      case value: Float => value.toByte
      case value: Double => value.toByte
      case value: Byte => value.toByte
      case value: Short => value.toByte
      case value: Int => value.toByte
      case value: Long => value.toByte
      case value: UByte => value.toByte
      case value: UShort => value.toByte
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a byte.")
    }

    override def toFloat(value: Byte): Float = value.toFloat
    override def toDouble(value: Byte): Double = value.toDouble
    override def toByte(value: Byte): Byte = value
    override def toShort(value: Byte): Short = value.toShort
    override def toInt(value: Byte): Int = value.toInt
    override def toLong(value: Byte): Long = value.toLong
    override def toUByte(value: Byte): UByte = UByte(value)
    override def toUShort(value: Byte): UShort = UShort(value.toChar)
  }

  private[types] class ShortIsSupportedType extends SupportedType[Short] {
    override def dataType: DataType = DataType.Int16
    override def cast[R: SupportedType](value: R, dataType: DataType): Short = value match {
      case value: Boolean => if (value) 1 else 0
      case value: Float => value.toShort
      case value: Double => value.toShort
      case value: Byte => value.toShort
      case value: Short => value.toShort
      case value: Int => value.toShort
      case value: Long => value.toShort
      case value: UByte => value.toShort
      case value: UShort => value.toShort
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a short.")
    }

    override def toFloat(value: Short): Float = value.toFloat
    override def toDouble(value: Short): Double = value.toDouble
    override def toByte(value: Short): Byte = value.toByte
    override def toShort(value: Short): Short = value
    override def toInt(value: Short): Int = value.toInt
    override def toLong(value: Short): Long = value.toLong
    override def toUByte(value: Short): UByte = UByte(value.toByte)
    override def toUShort(value: Short): UShort = UShort(value.toChar)
  }

  private[types] class IntIsSupportedType extends SupportedType[Int] {
    override def dataType: DataType = DataType.Int32
    override def cast[R: SupportedType](value: R, dataType: DataType): Int = value match {
      case value: Boolean => if (value) 1 else 0
      case value: Float => value.toInt
      case value: Double => value.toInt
      case value: Byte => value.toInt
      case value: Short => value.toInt
      case value: Int => value.toInt
      case value: Long => value.toInt
      case value: UByte => value.toInt
      case value: UShort => value.toInt
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an integer.")
    }

    override def toFloat(value: Int): Float = value.toFloat
    override def toDouble(value: Int): Double = value.toDouble
    override def toByte(value: Int): Byte = value.toByte
    override def toShort(value: Int): Short = value.toShort
    override def toInt(value: Int): Int = value
    override def toLong(value: Int): Long = value.toLong
    override def toUByte(value: Int): UByte = UByte(value.toByte)
    override def toUShort(value: Int): UShort = UShort(value.toChar)
  }

  private[types] class LongIsSupportedType extends SupportedType[Long] {
    override def dataType: DataType = DataType.Int64
    override def cast[R: SupportedType](value: R, dataType: DataType): Long = value match {
      case value: Boolean => if (value) 1L else 0L
      case value: Float => value.toLong
      case value: Double => value.toLong
      case value: Byte => value.toLong
      case value: Short => value.toLong
      case value: Int => value.toLong
      case value: Long => value.toLong
      case value: UByte => value.toLong
      case value: UShort => value.toLong
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a long.")
    }

    override def toFloat(value: Long): Float = value.toFloat
    override def toDouble(value: Long): Double = value.toDouble
    override def toByte(value: Long): Byte = value.toByte
    override def toShort(value: Long): Short = value.toShort
    override def toInt(value: Long): Int = value.toInt
    override def toLong(value: Long): Long = value
    override def toUByte(value: Long): UByte = UByte(value.toByte)
    override def toUShort(value: Long): UShort = UShort(value.toChar)
  }

  private[types] class UByteIsSupportedType extends SupportedType[UByte] {
    override def dataType: DataType = DataType.UInt8
    override def cast[R: SupportedType](value: R, dataType: DataType): UByte = value match {
      case value: Boolean => if (value) UByte(1) else UByte(0)
      case value: Float => UByte(value.toInt)
      case value: Double => UByte(value.toInt)
      case value: Byte => UByte(value)
      case value: Short => UByte(value)
      case value: Int => UByte(value)
      case value: Long => UByte(value.toInt)
      case value: UByte => value
      case value: UShort => UByte(value.toInt)
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned byte.")
    }

    override def toFloat(value: UByte): Float = value.toFloat
    override def toDouble(value: UByte): Double = value.toDouble
    override def toByte(value: UByte): Byte = value.toByte
    override def toShort(value: UByte): Short = value.toShort
    override def toInt(value: UByte): Int = value.toInt
    override def toLong(value: UByte): Long = value.toLong
    override def toUByte(value: UByte): UByte = value
    override def toUShort(value: UByte): UShort = UShort(value.toChar)
  }

  private[types] class UShortIsSupportedType extends SupportedType[UShort] {
    override def dataType: DataType = DataType.UInt16
    override def cast[R: SupportedType](value: R, dataType: DataType): UShort = value match {
      case value: Boolean => if (value) UShort(1) else UShort(0)
      case value: Float => UShort(value.toInt)
      case value: Double => UShort(value.toInt)
      case value: Byte => UShort(value)
      case value: Short => UShort(value)
      case value: Int => UShort(value)
      case value: Long => UShort(value.toInt)
      case value: UByte => UShort(value.toShort)
      case value: UShort => value
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to an unsigned short.")
    }

    override def toFloat(value: UShort): Float = value.toFloat
    override def toDouble(value: UShort): Double = value.toDouble
    override def toByte(value: UShort): Byte = value.toByte
    override def toShort(value: UShort): Short = value.toShort
    override def toInt(value: UShort): Int = value.toInt
    override def toLong(value: UShort): Long = value.toLong
    override def toUByte(value: UShort): UByte = UByte(value.toByte)
    override def toUShort(value: UShort): UShort = value
  }

  trait Implicits {
    implicit def toSupportedTypeOps[@specialized T: SupportedType](value: T): SupportedTypeOps[T] = {
      new SupportedTypeOps(value)
    }

    implicit final val BooleanIsSupportedType: SupportedType[Boolean] = new BooleanIsSupportedType
    implicit final val StringIsSupportedType : SupportedType[String]  = new StringIsSupportedType
    implicit final val FloatIsSupportedType  : SupportedType[Float]   = new FloatIsSupportedType
    implicit final val DoubleIsSupportedType : SupportedType[Double]  = new DoubleIsSupportedType
    implicit final val ByteIsSupportedType   : SupportedType[Byte]    = new ByteIsSupportedType
    implicit final val ShortIsSupportedType  : SupportedType[Short]   = new ShortIsSupportedType
    implicit final val IntIsSupportedType    : SupportedType[Int]     = new IntIsSupportedType
    implicit final val LongIsSupportedType   : SupportedType[Long]    = new LongIsSupportedType
    implicit final val UByteIsSupportedType  : SupportedType[UByte]   = new UByteIsSupportedType
    implicit final val UShortIsSupportedType : SupportedType[UShort]  = new UShortIsSupportedType
  }

  object Implicits extends Implicits
}
