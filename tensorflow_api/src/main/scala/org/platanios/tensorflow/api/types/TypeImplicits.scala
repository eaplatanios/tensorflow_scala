package org.platanios.tensorflow.api.types

import SupportedType._

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
trait TypeImplicits {
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

object TypeImplicits extends TypeImplicits
