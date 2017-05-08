package org.platanios.tensorflow.api.types

import org.platanios.tensorflow.api.Exception.InvalidDataTypeException

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedType[T] {
  @inline def dataType: DataType[T]
  @throws[UnsupportedOperationException]
  @inline def cast[R: SupportedType](value: R, dataType: DataType[T]): T

  private[this] def invalidConversionException(value: T, t: String): Exception = {
    new IllegalArgumentException(s"Value '$value' cannot be converted to $t.")
  }

  def toBoolean(value: T): Boolean = throw invalidConversionException(value, "a boolean")
  def toString(value: T): String = throw invalidConversionException(value, "a string")
  def toFloat(value: T): Float = throw invalidConversionException(value, "a float")
  def toDouble(value: T): Double = throw invalidConversionException(value, "a double")
  def toByte(value: T): Byte = throw invalidConversionException(value, "a byte")
  def toShort(value: T): Short = throw invalidConversionException(value, "a short")
  def toInt(value: T): Int = throw invalidConversionException(value, "an int")
  def toLong(value: T): Long = throw invalidConversionException(value, "a long")
  def toUByte(value: T): UByte = throw invalidConversionException(value, "an unsigned byte")
  def toUShort(value: T): UShort = throw invalidConversionException(value, "an unsigned short")
}

object SupportedType {
  implicit class SupportedTypeOps[T](val value: T) extends AnyVal {
    def dataType: DataType[T] = implicitly[SupportedType[T]].dataType
    def cast[R: SupportedType](dataType: DataType[R]): R = implicitly[SupportedType[R]].cast(value, dataType)

    def toBoolean(value: T): Boolean = implicitly[SupportedType[T]].toBoolean(value)
    def toString(value: T): String = implicitly[SupportedType[T]].toString(value)
    def toFloat(value: T): Float = implicitly[SupportedType[T]].toFloat(value)
    def toDouble(value: T): Double = implicitly[SupportedType[T]].toDouble(value)
    def toByte(value: T): Byte = implicitly[SupportedType[T]].toByte(value)
    def toShort(value: T): Short = implicitly[SupportedType[T]].toShort(value)
    def toInt(value: T): Int = implicitly[SupportedType[T]].toInt(value)
    def toLong(value: T): Long = implicitly[SupportedType[T]].toLong(value)
    def toUByte(value: T): UByte = implicitly[SupportedType[T]].toUByte(value)
    def toUShort(value: T): UShort = implicitly[SupportedType[T]].toUShort(value)
  }

  implicit val SupportedTypeBoolean = new SupportedType[Boolean] {
    override def dataType: DataType[Boolean] = DataType.Bool
    override def cast[R: SupportedType](value: R, dataType: DataType[Boolean]): Boolean = value match {
      case value: Boolean => value
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a boolean.")
    }

    override def toBoolean(value: Boolean): Boolean = value
    override def toString(value: Boolean): String = value.toString
    override def toFloat(value: Boolean): Float = if (value) 1.0f else 0.0f
    override def toDouble(value: Boolean): Double = if (value) 1.0 else 0.0
    override def toByte(value: Boolean): Byte = if (value) 1 else 0
    override def toShort(value: Boolean): Short = if (value) 1 else 0
    override def toInt(value: Boolean): Int = if (value) 1 else 0
    override def toLong(value: Boolean): Long = if (value) 1L else 0L
    override def toUByte(value: Boolean): UByte = if (value) UByte(1) else UByte(0)
    override def toUShort(value: Boolean): UShort = if (value) UShort(1) else UShort(0)
  }

  implicit val SupportedTypeString = new SupportedType[String] {
    override def dataType: DataType[String] = DataType.Str
    override def cast[R: SupportedType](value: R, dataType: DataType[String]): String = value.toString

    override def toString(value: String): String = value
  }

  implicit val SupportedTypeFloat = new SupportedType[Float] {
    override def dataType: DataType[Float] = DataType.Float32
    override def cast[R: SupportedType](value: R, dataType: DataType[Float]): Float = value match {
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

    override def toString(value: Float): String = value.toString
    override def toFloat(value: Float): Float = value
    override def toDouble(value: Float): Double = value.toDouble
    override def toByte(value: Float): Byte = value.toByte
    override def toShort(value: Float): Short = value.toShort
    override def toInt(value: Float): Int = value.toInt
    override def toLong(value: Float): Long = value.toLong
    override def toUByte(value: Float): UByte = UByte(value.toByte)
    override def toUShort(value: Float): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeDouble = new SupportedType[Double] {
    override def dataType: DataType[Double] = DataType.Float64
    override def cast[R: SupportedType](value: R, dataType: DataType[Double]): Double = value match {
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

    override def toString(value: Double): String = value.toString
    override def toFloat(value: Double): Float = value.toFloat
    override def toDouble(value: Double): Double = value
    override def toByte(value: Double): Byte = value.toByte
    override def toShort(value: Double): Short = value.toShort
    override def toInt(value: Double): Int = value.toInt
    override def toLong(value: Double): Long = value.toLong
    override def toUByte(value: Double): UByte = UByte(value.toByte)
    override def toUShort(value: Double): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeByte = new SupportedType[Byte] {
    override def dataType: DataType[Byte] = DataType.Int8
    override def cast[R: SupportedType](value: R, dataType: DataType[Byte]): Byte = value match {
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

    override def toString(value: Byte): String = value.toString
    override def toFloat(value: Byte): Float = value.toFloat
    override def toDouble(value: Byte): Double = value.toDouble
    override def toByte(value: Byte): Byte = value
    override def toShort(value: Byte): Short = value.toShort
    override def toInt(value: Byte): Int = value.toInt
    override def toLong(value: Byte): Long = value.toLong
    override def toUByte(value: Byte): UByte = UByte(value)
    override def toUShort(value: Byte): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeShort = new SupportedType[Short] {
    override def dataType: DataType[Short] = DataType.Int16
    override def cast[R: SupportedType](value: R, dataType: DataType[Short]): Short = value match {
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

    override def toString(value: Short): String = value.toString
    override def toFloat(value: Short): Float = value.toFloat
    override def toDouble(value: Short): Double = value.toDouble
    override def toByte(value: Short): Byte = value.toByte
    override def toShort(value: Short): Short = value
    override def toInt(value: Short): Int = value.toInt
    override def toLong(value: Short): Long = value.toLong
    override def toUByte(value: Short): UByte = UByte(value.toByte)
    override def toUShort(value: Short): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeInt = new SupportedType[Int] {
    override def dataType: DataType[Int] = DataType.Int32
    override def cast[R: SupportedType](value: R, dataType: DataType[Int]): Int = value match {
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

    override def toString(value: Int): String = value.toString
    override def toFloat(value: Int): Float = value.toFloat
    override def toDouble(value: Int): Double = value.toDouble
    override def toByte(value: Int): Byte = value.toByte
    override def toShort(value: Int): Short = value.toShort
    override def toInt(value: Int): Int = value
    override def toLong(value: Int): Long = value.toLong
    override def toUByte(value: Int): UByte = UByte(value.toByte)
    override def toUShort(value: Int): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeLong = new SupportedType[Long] {
    override def dataType: DataType[Long] = DataType.Int64
    override def cast[R: SupportedType](value: R, dataType: DataType[Long]): Long = value match {
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

    override def toString(value: Long): String = value.toString
    override def toFloat(value: Long): Float = value.toFloat
    override def toDouble(value: Long): Double = value.toDouble
    override def toByte(value: Long): Byte = value.toByte
    override def toShort(value: Long): Short = value.toShort
    override def toInt(value: Long): Int = value.toInt
    override def toLong(value: Long): Long = value
    override def toUByte(value: Long): UByte = UByte(value.toByte)
    override def toUShort(value: Long): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeUByte = new SupportedType[UByte] {
    override def dataType: DataType[UByte] = DataType.UInt8
    override def cast[R: SupportedType](value: R, dataType: DataType[UByte]): UByte = value match {
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

    override def toString(value: UByte): String = value.toString
    override def toFloat(value: UByte): Float = value.toFloat
    override def toDouble(value: UByte): Double = value.toDouble
    override def toByte(value: UByte): Byte = value.toByte
    override def toShort(value: UByte): Short = value.toShort
    override def toInt(value: UByte): Int = value.toInt
    override def toLong(value: UByte): Long = value.toLong
    override def toUByte(value: UByte): UByte = value
    override def toUShort(value: UByte): UShort = UShort(value.toChar)
  }

  implicit val SupportedTypeUShort = new SupportedType[UShort] {
    override def dataType: DataType[UShort] = DataType.UInt16
    override def cast[R: SupportedType](value: R, dataType: DataType[UShort]): UShort = value match {
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

    override def toString(value: UShort): String = value.toString
    override def toFloat(value: UShort): Float = value.toFloat
    override def toDouble(value: UShort): Double = value.toDouble
    override def toByte(value: UShort): Byte = value.toByte
    override def toShort(value: UShort): Short = value.toShort
    override def toInt(value: UShort): Int = value.toInt
    override def toLong(value: UShort): Long = value.toLong
    override def toUByte(value: UShort): UByte = UByte(value.toByte)
    override def toUShort(value: UShort): UShort = value
  }
}

//sealed trait SupportedScalaType extends Any {
//  @inline def dataType: DataType
//
//  @throws[UnsupportedOperationException]
//  @inline def cast(dataType: DataType): dataType.ScalaType
//
//  def asNumeric: SupportedScalaNumberType[_] = throw InvalidDataTypeException(s"'$this' is not a numeric Scala type.")
//
//  def toStr: Str = Str(toString)
//
//  override def toString: String
//
//  override def equals(that: Any): Boolean
//
//  override def hashCode: Int
//}
//
//sealed trait ComparableSupportedScalaType[C <: ComparableSupportedScalaType[C]] extends Any with SupportedScalaType {
//  def ==(that: ComparableSupportedScalaType[_]): Boolean
//  def <(that: ComparableSupportedScalaType[_]): Boolean
//  def !=(that: ComparableSupportedScalaType[_]): Boolean = !(this == that)
//
//  def ===(that: ComparableSupportedScalaType[_]): Boolean = this == that
//  def =!=(that: ComparableSupportedScalaType[_]): Boolean = !(this === that)
//
//  def <=(that: ComparableSupportedScalaType[_]): Boolean = this < that || this == that
//  def >(that: ComparableSupportedScalaType[_]): Boolean = !(this <= that)
//  def >=(that: ComparableSupportedScalaType[_]): Boolean = !(this < that)
//}
//
//final case class Bool(value: Boolean) extends AnyVal with ComparableSupportedScalaType[Bool] {
//  @inline override def dataType: DataType = DataType.Bool
//
//  @inline override def cast(dataType: DataType): dataType.ScalaType = {
//    (dataType match {
//      // case DataType.Float16 => if (value) Float16(1.0f) else Float16(0.0f)
//      case DataType.Float32 => if (value) Float32(1.0f) else Float32(0.0f)
//      case DataType.Float64 => if (value) Float64(1.0) else Float64(0.0)
//      // case DataType.BFloat16 => if (value) BFloat16(1.0f) else BFloat16(0.0f)
//      case DataType.Int8 => if (value) Int8(1) else Int8(0)
//      case DataType.Int16 => if (value) Int16(1) else Int16(0)
//      case DataType.Int32 => if (value) Int32(1) else Int32(0)
//      case DataType.Int64 => if (value) Int64(1) else Int64(0)
//      case DataType.UInt8 => if (value) UInt8(1) else UInt8(0)
//      case DataType.UInt16 => if (value) UInt16(1) else UInt16(0)
//      case DataType.QInt8 => if (value) Int8(1) else Int8(0)
//      case DataType.QInt16 => if (value) Int16(1) else Int16(0)
//      case DataType.QInt32 => if (value) Int32(1) else Int32(0)
//      case DataType.QUInt8 => if (value) UInt8(1) else UInt8(0)
//      case DataType.QUInt16 => if (value) UInt16(1) else UInt16(0)
//      case DataType.Bool => this
//      case DataType.Str => toString
//      case DataType.Resource =>
//        throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
//    }).asInstanceOf[dataType.ScalaType]
//  }
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Bool => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Bool => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def toString: String = value.toString
//}
//
//final case class Str(value: String) extends AnyVal with ComparableSupportedScalaType[Str] {
//  override def dataType: DataType = DataType.Str
//  override def cast(dataType: DataType): dataType.ScalaType = {
//    (dataType match {
//      // case DataType.Float16 => ???
//      case DataType.Float32 => Float32(value.toFloat)
//      case DataType.Float64 => Float64(value.toDouble)
//      // case DataType.BFloat16 => ???
//      case DataType.Int8 => Int8(value.toByte)
//      case DataType.Int16 => Int16(value.toShort)
//      case DataType.Int32 => Int32(value.toInt)
//      case DataType.Int64 => Int64(value.toLong)
//      case DataType.UInt8 => UInt8(value.toByte)
//      case DataType.UInt16 => UInt16(value.toShort)
//      case DataType.QInt8 => Int8(value.toByte)
//      case DataType.QInt16 => Int16(value.toShort)
//      case DataType.QInt32 => Int32(value.toInt)
//      case DataType.QUInt8 => UInt8(value.toByte)
//      case DataType.QUInt16 => UInt16(value.toShort)
//      case DataType.Bool => Bool(value.toBoolean)
//      case DataType.Str => this
//      case DataType.Resource =>
//        throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
//    }).asInstanceOf[dataType.ScalaType]
//  }
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Str => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Str => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def toString: String = value
//}
//
//sealed trait SupportedScalaNumberType[N <: SupportedScalaNumberType[N]]
//    extends Any with ComparableSupportedScalaType[N] {
//  @inline override def cast(dataType: DataType): dataType.ScalaType = dataType match {
//    // case DataType.Float16 => toFloat16.asInstanceOf[dataType.ScalaType]
//    case DataType.Float32 => toFloat32.asInstanceOf[dataType.ScalaType]
//    case DataType.Float64 => toFloat64.asInstanceOf[dataType.ScalaType]
//    // case DataType.BFloat16 => toBFloat16.asInstanceOf[dataType.ScalaType]
//    case DataType.Int8 => toInt8.asInstanceOf[dataType.ScalaType]
//    case DataType.Int16 => toInt16.asInstanceOf[dataType.ScalaType]
//    case DataType.Int32 => toInt32.asInstanceOf[dataType.ScalaType]
//    case DataType.Int64 => toInt64.asInstanceOf[dataType.ScalaType]
//    case DataType.UInt8 => toUInt8.asInstanceOf[dataType.ScalaType]
//    case DataType.UInt16 => toUInt16.asInstanceOf[dataType.ScalaType]
//    case DataType.QInt8 => toQInt8.asInstanceOf[dataType.ScalaType]
//    case DataType.QInt16 => toQInt16.asInstanceOf[dataType.ScalaType]
//    case DataType.QInt32 => toQInt32.asInstanceOf[dataType.ScalaType]
//    case DataType.QUInt8 => toQUInt8.asInstanceOf[dataType.ScalaType]
//    case DataType.QUInt16 => toQUInt16.asInstanceOf[dataType.ScalaType]
//    case DataType.Bool => toBool.asInstanceOf[dataType.ScalaType]
//    case DataType.Str => toStr.asInstanceOf[dataType.ScalaType]
//    case DataType.Resource =>
//      throw new UnsupportedOperationException("The resource data type is not supported on the Scala side.")
//  }
//
//  def toByte: Byte = toInt8.value
//  def toShort: Short = toInt16.value
//  def toInt: Int = toInt32.value
//  def toLong: Long = toInt64.value
//  def toChar: Char = toUInt16.value
//  def toFloat: Float = toFloat32.value
//  def toDouble: Double = toFloat64.value
//  def toBoolean: Boolean = toBool.value
//
//  // def toFloat16: Float16
//  def toFloat32: Float32
//  def toFloat64: Float64
//  // def toBFloat16: BFloat16
//  def toInt8: Int8
//  def toInt16: Int16
//  def toInt32: Int32
//  def toInt64: Int64
//  def toUInt8: UInt8
//  def toUInt16: UInt16
//  def toQInt8: Int8 = toInt8
//  def toQInt16: Int16 = toInt16
//  def toQInt32: Int32 = toInt32
//  def toQUInt8: UInt8 = toUInt8
//  def toQUInt16: UInt16 = toUInt16
//  def toBool: Bool
//
//  def unary_- : N
//
//  def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_]
//  def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_]
//  def *(that: N): N
//  def /(that: N): N
//  def %(that: N): N
//  def **(that: N): N
//
//  override def asNumeric: SupportedScalaNumberType[_] = this
//}
//
//final case class Float32(value: Float) extends AnyVal with SupportedScalaNumberType[Float32] {
//  @inline override def dataType: DataType = DataType.Float32
//
//  override def toFloat32: Float32 = this
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = UInt8(value.toByte)
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Float32 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Float32 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : Float32 = Float32(-this.value)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Float32(this.value + that.value)
//    case that: Int16 => Float32(this.value + that.value)
//    case that: Int32 => Float32(this.value + that.value)
//    case that: Int64 => Float64(this.value + that.value)
//    case that: UInt8 => Float32(this.value + that.value)
//    case that: UInt16 => Float32(this.value + that.value)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Float32(this.value - that.value)
//    case that: Int16 => Float32(this.value - that.value)
//    case that: Int32 => Float32(this.value - that.value)
//    case that: Int64 => Float64(this.value - that.value)
//    case that: UInt8 => Float32(this.value - that.value)
//    case that: UInt16 => Float32(this.value - that.value)
//  }
//
//  override def *(that: Float32): Float32 = Float32(this.value * that.value)
//  override def /(that: Float32): Float32 = Float32(this.value / that.value)
//  override def %(that: Float32): Float32 = Float32(this.value % that.value)
//  override def **(that: Float32): Float32 = Float32(math.pow(this.value, that.value).toFloat)
//
//  override def toString: String = value.toString
//}
//
//final case class Float64(value: Double) extends AnyVal with SupportedScalaNumberType[Float64] {
//  @inline override def dataType: DataType = DataType.Float64
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = this
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = UInt8(value.toByte)
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Float64 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Float64 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : Float64 = Float64(-this.value)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float64(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Float64(this.value + that.value)
//    case that: Int16 => Float64(this.value + that.value)
//    case that: Int32 => Float64(this.value + that.value)
//    case that: Int64 => Float64(this.value + that.value)
//    case that: UInt8 => Float64(this.value + that.value)
//    case that: UInt16 => Float64(this.value + that.value)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float64(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Float64(this.value - that.value)
//    case that: Int16 => Float64(this.value - that.value)
//    case that: Int32 => Float64(this.value - that.value)
//    case that: Int64 => Float64(this.value - that.value)
//    case that: UInt8 => Float64(this.value - that.value)
//    case that: UInt16 => Float64(this.value - that.value)
//  }
//
//  override def *(that: Float64): Float64 = Float64(this.value * that.value)
//  override def /(that: Float64): Float64 = Float64(this.value / that.value)
//  override def %(that: Float64): Float64 = Float64(this.value % that.value)
//  override def **(that: Float64): Float64 = Float64(math.pow(this.value, that.value))
//
//  override def toString: String = value.toString
//}
//
//final case class Int8(value: Byte) extends AnyVal with SupportedScalaNumberType[Int8] {
//  @inline override def dataType: DataType = DataType.Int8
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = this
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = UInt8(value)
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int8 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int8 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : Int8 = Int8((-this.value).toByte)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Int8((this.value + that.value).toByte)
//    case that: Int16 => Int16((this.value + that.value).toShort)
//    case that: Int32 => Int32(this.value + that.value)
//    case that: Int64 => Int64(this.value + that.value)
//    case that: UInt8 => Int16((this.value + that.value).toShort)
//    case that: UInt16 => Int16((this.value + that.value).toShort)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Int8((this.value - that.value).toByte)
//    case that: Int16 => Int16((this.value - that.value).toShort)
//    case that: Int32 => Int32(this.value - that.value)
//    case that: Int64 => Int64(this.value - that.value)
//    case that: UInt8 => Int16((this.value - that.value).toShort)
//    case that: UInt16 => Int16((this.value - that.value).toShort)
//  }
//
//  override def *(that: Int8): Int8 = Int8((this.value * that.value).toByte)
//  override def /(that: Int8): Int8 = Int8((this.value / that.value).toByte)
//  override def %(that: Int8): Int8 = Int8((this.value % that.value).toByte)
//  override def **(that: Int8): Int8 = Int8(math.pow(this.value, that.value).toByte)
//
//  override def toString: String = value.toString
//}
//
//final case class Int16(value: Short) extends AnyVal with SupportedScalaNumberType[Int16] {
//  @inline override def dataType: DataType = DataType.Int16
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = this
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = UInt8(value.toByte)
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int16 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int16 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : Int16 = Int16((-this.value).toShort)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Int16((this.value + that.value).toShort)
//    case that: Int16 => Int16((this.value + that.value).toShort)
//    case that: Int32 => Int32(this.value + that.value)
//    case that: Int64 => Int64(this.value + that.value)
//    case that: UInt8 => Int16((this.value + that.value).toShort)
//    case that: UInt16 => Int32(this.value + that.value)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Int16((this.value - that.value).toShort)
//    case that: Int16 => Int16((this.value - that.value).toShort)
//    case that: Int32 => Int32(this.value - that.value)
//    case that: Int64 => Int64(this.value - that.value)
//    case that: UInt8 => Int16((this.value - that.value).toShort)
//    case that: UInt16 => Int32(this.value - that.value)
//  }
//
//  override def *(that: Int16): Int16 = Int16((this.value * that.value).toShort)
//  override def /(that: Int16): Int16 = Int16((this.value / that.value).toShort)
//  override def %(that: Int16): Int16 = Int16((this.value % that.value).toShort)
//  override def **(that: Int16): Int16 = Int16(math.pow(this.value, that.value).toShort)
//
//  override def toString: String = value.toString
//}
//
//final case class Int32(value: Int) extends AnyVal with SupportedScalaNumberType[Int32] {
//  @inline override def dataType: DataType = DataType.Int32
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = this
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = UInt8(value.toByte)
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int32 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int32 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : Int32 = Int32(-this.value)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Int32(this.value + that.value)
//    case that: Int16 => Int32(this.value + that.value)
//    case that: Int32 => Int32(this.value + that.value)
//    case that: Int64 => Int64(this.value + that.value)
//    case that: UInt8 => Int32(this.value + that.value)
//    case that: UInt16 => Int32(this.value + that.value)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Int32(this.value - that.value)
//    case that: Int16 => Int32(this.value - that.value)
//    case that: Int32 => Int32(this.value - that.value)
//    case that: Int64 => Int64(this.value - that.value)
//    case that: UInt8 => Int32(this.value - that.value)
//    case that: UInt16 => Int32(this.value - that.value)
//  }
//
//  override def *(that: Int32): Int32 = Int32(this.value * that.value)
//  override def /(that: Int32): Int32 = Int32(this.value / that.value)
//  override def %(that: Int32): Int32 = Int32(this.value % that.value)
//  override def **(that: Int32): Int32 = Int32(math.pow(this.value, that.value).toInt)
//
//  override def toString: String = value.toString
//}
//
//final case class Int64(value: Long) extends AnyVal with SupportedScalaNumberType[Int64] {
//  @inline override def dataType: DataType = DataType.Int64
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = this
//  override def toUInt8: UInt8 = UInt8(value.toByte)
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int64 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: Int64 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : Int64 = Int64(-this.value)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float64(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Int64(this.value + that.value)
//    case that: Int16 => Int64(this.value + that.value)
//    case that: Int32 => Int64(this.value + that.value)
//    case that: Int64 => Int64(this.value + that.value)
//    case that: UInt8 => Int64(this.value + that.value)
//    case that: UInt16 => Int64(this.value + that.value)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float64(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Int64(this.value - that.value)
//    case that: Int16 => Int64(this.value - that.value)
//    case that: Int32 => Int64(this.value - that.value)
//    case that: Int64 => Int64(this.value - that.value)
//    case that: UInt8 => Int64(this.value - that.value)
//    case that: UInt16 => Int64(this.value - that.value)
//  }
//
//  override def *(that: Int64): Int64 = Int64(this.value * that.value)
//  override def /(that: Int64): Int64 = Int64(this.value / that.value)
//  override def %(that: Int64): Int64 = Int64(this.value % that.value)
//  override def **(that: Int64): Int64 = Int64(math.pow(this.value, that.value).toLong)
//
//  override def toString: String = value.toString
//}
//
//// TODO: [DATA_TYPE] There might be some issues with the unsigned data types.
//
//final case class UInt8(value: Byte) extends AnyVal with SupportedScalaNumberType[UInt8] {
//  @inline override def dataType: DataType = DataType.UInt8
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = this
//  override def toUInt16: UInt16 = UInt16(value.toChar)
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: UInt8 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: UInt8 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : UInt8 = UInt8((-this.value).toByte)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => Int16((this.value + that.value).toShort)
//    case that: Int16 => Int16((this.value + that.value).toShort)
//    case that: Int32 => Int32(this.value + that.value)
//    case that: Int64 => Int64(this.value + that.value)
//    case that: UInt8 => UInt8((this.value + that.value).toByte)
//    case that: UInt16 => UInt16((this.value + that.value).toChar)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => Int16((this.value - that.value).toShort)
//    case that: Int16 => Int16((this.value - that.value).toShort)
//    case that: Int32 => Int32(this.value - that.value)
//    case that: Int64 => Int64(this.value - that.value)
//    case that: UInt8 => UInt8((this.value - that.value).toByte)
//    case that: UInt16 => UInt16((this.value - that.value).toChar)
//  }
//
//  override def *(that: UInt8): UInt8 = UInt8((this.value * that.value).toByte)
//  override def /(that: UInt8): UInt8 = UInt8((this.toInt / that.toInt).toByte)
//  override def %(that: UInt8): UInt8 = UInt8((this.toInt % that.toInt).toByte)
//  override def **(that: UInt8): UInt8 = UInt8(math.pow(this.toLong, that.toLong).toByte)
//
//  override def toString: String = value.toString
//}
//
//object UInt8 {
//  def apply(value: Short): UInt8 = UInt8(value.toByte)
//  def apply(value: Int): UInt8 = UInt8(value.toByte)
//  def apply(value: Long): UInt8 = UInt8(value.toByte)
//  def apply(value: Char): UInt8 = UInt8(value.toByte)
//}
//
//final case class UInt16(value: Char) extends AnyVal with SupportedScalaNumberType[UInt16] {
//  @inline override def dataType: DataType = DataType.UInt16
//
//  override def toFloat32: Float32 = Float32(value.toFloat)
//  override def toFloat64: Float64 = Float64(value.toDouble)
//  override def toInt8: Int8 = Int8(value.toByte)
//  override def toInt16: Int16 = Int16(value.toShort)
//  override def toInt32: Int32 = Int32(value.toInt)
//  override def toInt64: Int64 = Int64(value.toLong)
//  override def toUInt8: UInt8 = UInt8(value.toByte)
//  override def toUInt16: UInt16 = this
//  override def toBool: Bool = Bool(value != 0)
//
//  def ==(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: UInt16 => this.value == that.value
//    case _ => false
//  }
//
//  def <(that: ComparableSupportedScalaType[_]): Boolean = that match {
//    case that: UInt16 => this.value < that.value
//    case _ => throw new UnsupportedOperationException("Unsupported comparison operation.")
//  }
//
//  override def unary_- : UInt16 = UInt16((-this.value).toChar)
//
//  override def +(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value + that.value)
//    case that: Float64 => Float64(this.value + that.value)
//    case that: Int8 => UInt16((this.value + that.value).toChar)
//    case that: Int16 => Int32(this.value + that.value)
//    case that: Int32 => Int32(this.value + that.value)
//    case that: Int64 => Int64(this.value + that.value)
//    case that: UInt8 => UInt16((this.value + that.value).toChar)
//    case that: UInt16 => UInt16((this.value + that.value).toChar)
//  }
//
//  override def -(that: SupportedScalaNumberType[_]): SupportedScalaNumberType[_] = that match {
//    case that: Float32 => Float32(this.value - that.value)
//    case that: Float64 => Float64(this.value - that.value)
//    case that: Int8 => UInt16((this.value - that.value).toChar)
//    case that: Int16 => Int32(this.value - that.value)
//    case that: Int32 => Int32(this.value - that.value)
//    case that: Int64 => Int64(this.value - that.value)
//    case that: UInt8 => UInt16((this.value - that.value).toChar)
//    case that: UInt16 => UInt16((this.value - that.value).toChar)
//  }
//
//  override def *(that: UInt16): UInt16 = UInt16((this.value * that.value).toChar)
//  override def /(that: UInt16): UInt16 = UInt16((this.toInt / that.toInt).toChar)
//  override def %(that: UInt16): UInt16 = UInt16((this.toInt % that.toInt).toChar)
//  override def **(that: UInt16): UInt16 = UInt16(math.pow(this.toLong, that.toLong).toChar)
//
//  override def toString: String = value.toString
//}
//
//object UInt16 {
//  def apply(value: Byte): UInt16 = UInt16(value.toChar)
//  def apply(value: Short): UInt16 = UInt16(value.toChar)
//  def apply(value: Int): UInt16 = UInt16(value.toChar)
//  def apply(value: Long): UInt16 = UInt16(value.toChar)
//}
