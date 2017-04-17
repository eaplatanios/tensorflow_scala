package org.platanios.tensorflow.api.types

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedScalaType extends Any {
  @inline private[api] def dataType: DataType
  @inline private[api] def cast(dataType: DataType): dataType.ScalaType

  def ==(x: Boolean): Boolean = this == Bool(x)
  def ==(x: Float): Boolean = this == Float32(x)
  def ==(x: Double): Boolean = this == Float64(x)
  def ==(x: Byte): Boolean = this == Int8(x)
  def ==(x: Short): Boolean = this == Int16(x)
  def ==(x: Int): Boolean = this == Int32(x)
  def ==(x: Long): Boolean = this == Int64(x)
  def ==(x: Char): Boolean = this == UInt16(x)

  def !=(x: Boolean): Boolean = !this.==(x)
  def !=(x: Float): Boolean = !this.==(x)
  def !=(x: Double): Boolean = !this.==(x)
  def !=(x: Byte): Boolean = !this.==(x)
  def !=(x: Short): Boolean = !this.==(x)
  def !=(x: Int): Boolean = !this.==(x)
  def !=(x: Long): Boolean = !this.==(x)
  def !=(x: Char): Boolean = !this.==(x)
}

final case class Bool(value: Boolean) extends AnyVal with SupportedScalaType {
  @inline override private[api] def dataType: DataType = DataType.Bool

  @inline override private[api] def cast(dataType: DataType): dataType.ScalaType = {
    (dataType match {
      // case DataType.Float16 => if (value) Float16(1.0f) else Float16(0.0f)
      case DataType.Float32 => if (value) Float32(1.0f) else Float32(0.0f)
      case DataType.Float64 => if (value) Float64(1.0) else Float64(0.0)
      // case DataType.BFloat16 => if (value) BFloat16(1.0f) else BFloat16(0.0f)
      case DataType.Int8  => if (value) Int8(1) else Int8(0)
      case DataType.Int16 => if (value) Int16(1) else Int16(0)
      case DataType.Int32 => if (value) Int32(1) else Int32(0)
      case DataType.Int64 => if (value) Int64(1) else Int64(0)
      // case DataType.UInt8  => if (value) UInt8(1) else UInt8(0)
      case DataType.UInt16 => if (value) UInt16(1) else UInt16(0)
      case DataType.QInt8  => if (value) Int8(1) else Int8(0)
      case DataType.QInt16 => if (value) Int16(1) else Int16(0)
      case DataType.QInt32 => if (value) Int32(1) else Int32(0)
      // case DataType.QUInt8    => if (value) UInt8(1) else UInt8(0)
      case DataType.QUInt16 => if (value) UInt16(1) else UInt16(0)
      case DataType.Bool => this
    }).asInstanceOf[dataType.ScalaType]
  }
}

private[types] trait SupportedScalaNumberType[T, N <: SupportedScalaNumberType[T, N]]
    extends Any with SupportedScalaType {
  @inline override private[api] def cast(dataType: DataType): dataType.ScalaType = dataType match {
    // case DataType.Float16 => toFloat16.asInstanceOf[dataType.ScalaType]
    case DataType.Float32 => toFloat32.asInstanceOf[dataType.ScalaType]
    case DataType.Float64 => toFloat64.asInstanceOf[dataType.ScalaType]
    // case DataType.BFloat16 => toBFloat16.asInstanceOf[dataType.ScalaType]
    case DataType.Int8  => toInt8.asInstanceOf[dataType.ScalaType]
    case DataType.Int16 => toInt16.asInstanceOf[dataType.ScalaType]
    case DataType.Int32 => toInt32.asInstanceOf[dataType.ScalaType]
    case DataType.Int64 => toInt64.asInstanceOf[dataType.ScalaType]
    // case DataType.UInt8  => toUInt8.asInstanceOf[dataType.ScalaType]
    case DataType.UInt16 => toUInt16.asInstanceOf[dataType.ScalaType]
    case DataType.QInt8  => toQInt8.asInstanceOf[dataType.ScalaType]
    case DataType.QInt16 => toQInt16.asInstanceOf[dataType.ScalaType]
    case DataType.QInt32 => toQInt32.asInstanceOf[dataType.ScalaType]
    // case DataType.QUInt8    => toQUInt8.asInstanceOf[dataType.ScalaType]
    case DataType.QUInt16 => toQUInt16.asInstanceOf[dataType.ScalaType]
  }

  def toByte: Byte = toInt8.value
  def toShort: Short = toInt16.value
  def toInt: Int = toInt32.value
  def toLong: Long = toInt64.value
  def toChar: Char = toUInt16.value
  def toFloat: Float = toFloat32.value
  def toDouble: Double = toFloat64.value

  // def toFloat16: Float16
  def toFloat32: Float32
  def toFloat64: Float64
  // def toBFloat16: BFloat16
  def toInt8: Int8
  def toInt16: Int16
  def toInt32: Int32
  def toInt64: Int64
  //def toUInt8: UInt8
  def toUInt16: UInt16
  def toQInt8: Int8 = toInt8
  def toQInt16: Int16 = toInt16
  def toQInt32: Int32 = toInt32
  // def toQUInt8: UInt8 = toUInt8
  def toQUInt16: UInt16 = toUInt16
}

final case class Float32(value: Float) extends AnyVal with SupportedScalaNumberType[Float, Float32] {
  @inline override private[api] def dataType: DataType = DataType.Float32

  override def toFloat32: Float32 = this
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt16: UInt16 = UInt16(value.toChar)

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}

final case class Float64(value: Double) extends AnyVal with SupportedScalaNumberType[Double, Float64] {
  @inline override private[api] def dataType: DataType = DataType.Float64

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = this
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt16: UInt16 = UInt16(value.toChar)

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}

final case class Int8(value: Byte) extends AnyVal with SupportedScalaNumberType[Byte, Int8] {
  @inline override private[api] def dataType: DataType = DataType.Int8

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = this
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt16: UInt16 = UInt16(value.toChar)

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}

final case class Int16(value: Short) extends AnyVal with SupportedScalaNumberType[Short, Int16] {
  @inline override private[api] def dataType: DataType = DataType.Int16

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = this
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt16: UInt16 = UInt16(value.toChar)

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}

final case class Int32(value: Int) extends AnyVal with SupportedScalaNumberType[Int, Int32] {
  @inline override private[api] def dataType: DataType = DataType.Int32

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = this
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt16: UInt16 = UInt16(value.toChar)

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}

final case class Int64(value: Long) extends AnyVal with SupportedScalaNumberType[Long, Int64] {
  @inline override private[api] def dataType: DataType = DataType.Int64

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = this
  override def toUInt16: UInt16 = UInt16(value.toChar)

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}

final case class UInt16(value: Char) extends AnyVal with SupportedScalaNumberType[Char, UInt16] {
  @inline override private[api] def dataType: DataType = DataType.UInt16

  override def toFloat32: Float32 = Float32(value.toFloat)
  override def toFloat64: Float64 = Float64(value.toDouble)
  override def toInt8: Int8 = Int8(value.toByte)
  override def toInt16: Int16 = Int16(value.toShort)
  override def toInt32: Int32 = Int32(value.toInt)
  override def toInt64: Int64 = Int64(value.toLong)
  override def toUInt16: UInt16 = this

  override def ==(x: Float): Boolean = value == x
  override def ==(x: Double): Boolean = value == x
  override def ==(x: Byte): Boolean = value == x
  override def ==(x: Short): Boolean = value == x
  override def ==(x: Int): Boolean = value == x
  override def ==(x: Long): Boolean = value == x
  override def ==(x: Char): Boolean = value == x
}
