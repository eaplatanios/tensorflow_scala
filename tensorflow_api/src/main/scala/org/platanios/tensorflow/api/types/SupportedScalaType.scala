package org.platanios.tensorflow.api.types

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedScalaType extends Any {
  @inline def dataType: DataType
  @inline def cast(dataType: DataType): dataType.ScalaType
}

object SupportedScalaType {
  @inline implicit def supportedScalaTypeToValue(value: Bool): Boolean = value.value
  @inline implicit def supportedScalaTypeToValue(value: Float32): Float = value.value
  @inline implicit def supportedScalaTypeToValue(value: Float64): Double = value.value
  @inline implicit def supportedScalaTypeToValue(value: Int8): Byte = value.value
  @inline implicit def supportedScalaTypeToValue(value: Int16): Short = value.value
  @inline implicit def supportedScalaTypeToValue(value: Int32): Int = value.value
  @inline implicit def supportedScalaTypeToValue(value: Int64): Long = value.value
  @inline implicit def supportedScalaTypeToValue(value: UInt16): Char = value.value
}

sealed trait ComparableSupportedScalaType[C <: ComparableSupportedScalaType[C]] extends Any with SupportedScalaType {
  def ==(that: C): Boolean
  def <(that: C): Boolean
  def !=(that: C): Boolean = !(this == that)

  def ===(that: C): Boolean = this == that
  def =!=(that: C): Boolean = !(this === that)

  def <=(that: C): Boolean = this < that || this == that
  def >(that: C): Boolean = !(this <= that)
  def >=(that: C): Boolean = !(this < that)
}

final case class Bool(value: Boolean) extends AnyVal with ComparableSupportedScalaType[Bool] {
  @inline override def dataType: DataType = DataType.Bool

  @inline override def cast(dataType: DataType): dataType.ScalaType = {
    (dataType match {
      // case DataType.Float16 => if (value) Float16(1.0f) else Float16(0.0f)
      case DataType.Float32 => if (value) Float32(1.0f) else Float32(0.0f)
      case DataType.Float64 => if (value) Float64(1.0) else Float64(0.0)
      // case DataType.BFloat16 => if (value) BFloat16(1.0f) else BFloat16(0.0f)
      case DataType.Int8    => if (value) Int8(1) else Int8(0)
      case DataType.Int16   => if (value) Int16(1) else Int16(0)
      case DataType.Int32   => if (value) Int32(1) else Int32(0)
      case DataType.Int64   => if (value) Int64(1) else Int64(0)
      case DataType.UInt8   => if (value) UInt8(1) else UInt8(0)
      case DataType.UInt16  => if (value) UInt16(1) else UInt16(0)
      case DataType.QInt8   => if (value) Int8(1) else Int8(0)
      case DataType.QInt16  => if (value) Int16(1) else Int16(0)
      case DataType.QInt32  => if (value) Int32(1) else Int32(0)
      case DataType.QUInt8  => if (value) UInt8(1) else UInt8(0)
      case DataType.QUInt16 => if (value) UInt16(1) else UInt16(0)
      case DataType.Bool    => this
    }).asInstanceOf[dataType.ScalaType]
  }

  override def ==(that: Bool): Boolean = this.value == that.value
  override def <(that: Bool): Boolean = this.value < that.value
}

private[types] trait SupportedScalaNumberType[N <: SupportedScalaNumberType[N]]
    extends Any with ComparableSupportedScalaType[N] {
  @inline override def cast(dataType: DataType): dataType.ScalaType = dataType match {
    // case DataType.Float16 => toFloat16.asInstanceOf[dataType.ScalaType]
    case DataType.Float32 => toFloat32.asInstanceOf[dataType.ScalaType]
    case DataType.Float64 => toFloat64.asInstanceOf[dataType.ScalaType]
    // case DataType.BFloat16 => toBFloat16.asInstanceOf[dataType.ScalaType]
    case DataType.Int8    => toInt8.asInstanceOf[dataType.ScalaType]
    case DataType.Int16   => toInt16.asInstanceOf[dataType.ScalaType]
    case DataType.Int32   => toInt32.asInstanceOf[dataType.ScalaType]
    case DataType.Int64   => toInt64.asInstanceOf[dataType.ScalaType]
    case DataType.UInt8   => toUInt8.asInstanceOf[dataType.ScalaType]
    case DataType.UInt16  => toUInt16.asInstanceOf[dataType.ScalaType]
    case DataType.QInt8   => toQInt8.asInstanceOf[dataType.ScalaType]
    case DataType.QInt16  => toQInt16.asInstanceOf[dataType.ScalaType]
    case DataType.QInt32  => toQInt32.asInstanceOf[dataType.ScalaType]
    case DataType.QUInt8  => toQUInt8.asInstanceOf[dataType.ScalaType]
    case DataType.QUInt16 => toQUInt16.asInstanceOf[dataType.ScalaType]
    case DataType.Bool    => toBool.asInstanceOf[dataType.ScalaType]
  }

  def toByte: Byte = toInt8.value
  def toShort: Short = toInt16.value
  def toInt: Int = toInt32.value
  def toLong: Long = toInt64.value
  def toChar: Char = toUInt16.value
  def toFloat: Float = toFloat32.value
  def toDouble: Double = toFloat64.value
  def toBoolean: Boolean = toBool.value

  // def toFloat16: Float16
  def toFloat32: Float32
  def toFloat64: Float64
  // def toBFloat16: BFloat16
  def toInt8: Int8
  def toInt16: Int16
  def toInt32: Int32
  def toInt64: Int64
  def toUInt8: UInt8
  def toUInt16: UInt16
  def toQInt8: Int8 = toInt8
  def toQInt16: Int16 = toInt16
  def toQInt32: Int32 = toInt32
  def toQUInt8: UInt8 = toUInt8
  def toQUInt16: UInt16 = toUInt16
  def toBool: Bool

  def unary_- : N
  def +(that: N): N
  def -(that: N): N
  def *(that: N): N
  def /(that: N): N
  def %(that: N): N
  def **(that: N): N
}

final case class Float32(value: Float) extends AnyVal with SupportedScalaNumberType[Float32] {
  @inline override def dataType: DataType = DataType.Float32

  override def toFloat32: Float32 = this
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = UInt8(value.toByte)
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: Float32): Boolean = this.value == that.value
  override def <(that: Float32): Boolean = this.value < that.value

  override def unary_- : Float32 = Float32(-this.value)
  override def +(that: Float32): Float32 = Float32(this.value + that.value)
  override def -(that: Float32): Float32 = Float32(this.value - that.value)
  override def *(that: Float32): Float32 = Float32(this.value * that.value)
  override def /(that: Float32): Float32 = Float32(this.value / that.value)
  override def %(that: Float32): Float32 = Float32(this.value % that.value)
  override def **(that: Float32): Float32 = Float32(math.pow(this.value, that.value).toFloat)
}

final case class Float64(value: Double) extends AnyVal with SupportedScalaNumberType[Float64] {
  @inline override def dataType: DataType = DataType.Float64

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = this
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = UInt8(value.toByte)
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: Float64): Boolean = this.value == that.value
  override def <(that: Float64): Boolean = this.value < that.value

  override def unary_- : Float64 = Float64(-this.value)
  override def +(that: Float64): Float64 = Float64(this.value + that.value)
  override def -(that: Float64): Float64 = Float64(this.value - that.value)
  override def *(that: Float64): Float64 = Float64(this.value * that.value)
  override def /(that: Float64): Float64 = Float64(this.value / that.value)
  override def %(that: Float64): Float64 = Float64(this.value % that.value)
  override def **(that: Float64): Float64 = Float64(math.pow(this.value, that.value))
}

final case class Int8(value: Byte) extends AnyVal with SupportedScalaNumberType[Int8] {
  @inline override def dataType: DataType = DataType.Int8

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = this
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = UInt8(value)
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: Int8): Boolean = this.value == that.value
  override def <(that: Int8): Boolean = this.value < that.value

  override def unary_- : Int8 = Int8((-this.value).toByte)
  override def +(that: Int8): Int8 = Int8((this.value + that.value).toByte)
  override def -(that: Int8): Int8 = Int8((this.value - that.value).toByte)
  override def *(that: Int8): Int8 = Int8((this.value * that.value).toByte)
  override def /(that: Int8): Int8 = Int8((this.value / that.value).toByte)
  override def %(that: Int8): Int8 = Int8((this.value % that.value).toByte)
  override def **(that: Int8): Int8 = Int8(math.pow(this.value, that.value).toByte)
}

final case class Int16(value: Short) extends AnyVal with SupportedScalaNumberType[Int16] {
  @inline override def dataType: DataType = DataType.Int16

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = this
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = UInt8(value.toByte)
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: Int16): Boolean = this.value == that.value
  override def <(that: Int16): Boolean = this.value < that.value

  override def unary_- : Int16 = Int16((-this.value).toShort)
  override def +(that: Int16): Int16 = Int16((this.value + that.value).toShort)
  override def -(that: Int16): Int16 = Int16((this.value - that.value).toShort)
  override def *(that: Int16): Int16 = Int16((this.value * that.value).toShort)
  override def /(that: Int16): Int16 = Int16((this.value / that.value).toShort)
  override def %(that: Int16): Int16 = Int16((this.value % that.value).toShort)
  override def **(that: Int16): Int16 = Int16(math.pow(this.value, that.value).toShort)
}

final case class Int32(value: Int) extends AnyVal with SupportedScalaNumberType[Int32] {
  @inline override def dataType: DataType = DataType.Int32

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = this
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = UInt8(value.toByte)
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: Int32): Boolean = this.value == that.value
  override def <(that: Int32): Boolean = this.value < that.value

  override def unary_- : Int32 = Int32(-this.value)
  override def +(that: Int32): Int32 = Int32(this.value + that.value)
  override def -(that: Int32): Int32 = Int32(this.value - that.value)
  override def *(that: Int32): Int32 = Int32(this.value * that.value)
  override def /(that: Int32): Int32 = Int32(this.value / that.value)
  override def %(that: Int32): Int32 = Int32(this.value % that.value)
  override def **(that: Int32): Int32 = Int32(math.pow(this.value, that.value).toInt)
}

final case class Int64(value: Long) extends AnyVal with SupportedScalaNumberType[Int64] {
  @inline override def dataType: DataType = DataType.Int64

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = this
  override def toUInt8: UInt8 = UInt8(value.toByte)
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: Int64): Boolean = this.value == that.value
  override def <(that: Int64): Boolean = this.value < that.value

  override def unary_- : Int64 = Int64(-this.value)
  override def +(that: Int64): Int64 = Int64(this.value + that.value)
  override def -(that: Int64): Int64 = Int64(this.value - that.value)
  override def *(that: Int64): Int64 = Int64(this.value * that.value)
  override def /(that: Int64): Int64 = Int64(this.value / that.value)
  override def %(that: Int64): Int64 = Int64(this.value % that.value)
  override def **(that: Int64): Int64 = Int64(math.pow(this.value, that.value).toLong)
}

final case class UInt8(value: Byte) extends AnyVal with SupportedScalaNumberType[UInt8] {
  @inline override def dataType: DataType = DataType.UInt8

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = this
  override def toUInt16: UInt16 = UInt16(value.toChar)
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: UInt8): Boolean = this.value == that.value
  override def <(that: UInt8): Boolean = this.value < that.value

  override def unary_- : UInt8 = UInt8((-this.value).toByte)
  override def +(that: UInt8): UInt8 = UInt8((this.value + that.value).toByte)
  override def -(that: UInt8): UInt8 = UInt8((this.value - that.value).toByte)
  override def *(that: UInt8): UInt8 = UInt8((this.value * that.value).toByte)
  override def /(that: UInt8): UInt8 = UInt8((this.toInt / that.toInt).toByte)
  override def %(that: UInt8): UInt8 = UInt8((this.toInt % that.toInt).toByte)
  override def **(that: UInt8): UInt8 = UInt8(math.pow(this.toLong, that.toLong).toByte)
}

object UInt8 {
  def apply(value: Short): UInt8 = UInt8(value.toByte)
  def apply(value: Int): UInt8 = UInt8(value.toByte)
  def apply(value: Long): UInt8 = UInt8(value.toByte)
  def apply(value: Char): UInt8 = UInt8(value.toByte)
}

final case class UInt16(value: Char) extends AnyVal with SupportedScalaNumberType[UInt16] {
  @inline override def dataType: DataType = DataType.UInt16

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt8: UInt8 = UInt8(value.toByte)
  override def toUInt16: UInt16 = this
  override def toBool: Bool = Bool(value != 0)

  override def ==(that: UInt16): Boolean = this.value == that.value
  override def <(that: UInt16): Boolean = this.value < that.value

  override def unary_- : UInt16 = UInt16((-this.value).toChar)
  override def +(that: UInt16): UInt16 = UInt16((this.value + that.value).toChar)
  override def -(that: UInt16): UInt16 = UInt16((this.value - that.value).toChar)
  override def *(that: UInt16): UInt16 = UInt16((this.value * that.value).toChar)
  override def /(that: UInt16): UInt16 = UInt16((this.toInt / that.toInt).toChar)
  override def %(that: UInt16): UInt16 = UInt16((this.toInt % that.toInt).toChar)
  override def **(that: UInt16): UInt16 = UInt16(math.pow(this.toLong, that.toLong).toChar)
}

object UInt16 {
  def apply(value: Byte): UInt16 = UInt16(value.toChar)
  def apply(value: Short): UInt16 = UInt16(value.toChar)
  def apply(value: Int): UInt16 = UInt16(value.toChar)
  def apply(value: Long): UInt16 = UInt16(value.toChar)
}
