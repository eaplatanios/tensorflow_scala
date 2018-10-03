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

/**
  * @author Emmanouil Antonios Platanios
  */

trait AllowedCast[Source, Target] {
  val targetDataType: DataType[Target]
}

object AllowedCast {
  implicit val int8ToFloat32: AllowedCast[Byte, Float]  = new ToFloat[Byte]
  implicit val int8ToFloat64: AllowedCast[Byte, Double] = new ToDouble[Byte]
  implicit val int8ToInt16  : AllowedCast[Byte, Short]  = new ToInt16[Byte]
  implicit val int8ToInt32  : AllowedCast[Byte, Int]    = new ToInt32[Byte]
  implicit val int8ToInt64  : AllowedCast[Byte, Long]   = new ToInt64[Byte]

  implicit val int16ToFloat32: AllowedCast[Short, Float]  = new ToFloat[Short]
  implicit val int16ToFloat64: AllowedCast[Short, Double] = new ToDouble[Short]
  implicit val int16ToInt32  : AllowedCast[Short, Int]    = new ToInt32[Short]
  implicit val int16ToInt64  : AllowedCast[Short, Long]   = new ToInt64[Short]

  implicit val int32ToFloat32: AllowedCast[Int, Float]  = new ToFloat[Int]
  implicit val int32ToFloat64: AllowedCast[Int, Double] = new ToDouble[Int]
  implicit val int32ToInt64  : AllowedCast[Int, Long]   = new ToInt64[Int]

  implicit val int64ToFloat32: AllowedCast[Long, Float]  = new ToFloat[Long]
  implicit val int64ToFloat64: AllowedCast[Long, Double] = new ToDouble[Long]
  implicit val int64ToInt64  : AllowedCast[Long, Long]   = new ToInt64[Long]

  implicit val uint8ToFloat32: AllowedCast[UByte, Float]  = new ToFloat[UByte]
  implicit val uint8ToFloat64: AllowedCast[UByte, Double] = new ToDouble[UByte]
  implicit val uint8ToInt16  : AllowedCast[UByte, Short]  = new ToInt16[UByte]
  implicit val uint8ToInt32  : AllowedCast[UByte, Int]    = new ToInt32[UByte]
  implicit val uint8ToInt64  : AllowedCast[UByte, Long]   = new ToInt64[UByte]
  implicit val uint8ToUInt16 : AllowedCast[UByte, UShort] = new ToUInt16[UByte]
  implicit val uint8ToUInt32 : AllowedCast[UByte, UInt]   = new ToUInt32[UByte]
  implicit val uint8ToUInt64 : AllowedCast[UByte, ULong]  = new ToUInt64[UByte]

  implicit val uint16ToFloat32: AllowedCast[UShort, Float]  = new ToFloat[UShort]
  implicit val uint16ToFloat64: AllowedCast[UShort, Double] = new ToDouble[UShort]
  implicit val uint16ToInt32  : AllowedCast[UShort, Int]    = new ToInt32[UShort]
  implicit val uint16ToInt64  : AllowedCast[UShort, Long]   = new ToInt64[UShort]
  implicit val uint16ToUInt32 : AllowedCast[UShort, UInt]   = new ToUInt32[UShort]
  implicit val uint16ToUInt64 : AllowedCast[UShort, ULong]  = new ToUInt64[UShort]

  implicit val uint32ToFloat32: AllowedCast[UInt, Float]  = new ToFloat[UInt]
  implicit val uint32ToFloat64: AllowedCast[UInt, Double] = new ToDouble[UInt]
  implicit val uint32ToInt64  : AllowedCast[UInt, Long]   = new ToInt64[UInt]
  implicit val uint32ToUInt64 : AllowedCast[UInt, ULong]  = new ToUInt64[UInt]

  implicit val uint64ToFloat32: AllowedCast[ULong, Float]  = new ToFloat[ULong]
  implicit val uint64ToFloat64: AllowedCast[ULong, Double] = new ToDouble[ULong]
  implicit val uint64ToInt64  : AllowedCast[ULong, Long]   = new ToInt64[ULong]
  implicit val uint64ToUInt64 : AllowedCast[ULong, ULong]  = new ToUInt64[ULong]

  private[types] class ToString[Source] extends AllowedCast[Source, String] {
    val targetDataType: DataType[String] = STRING
  }

  private[types] class ToBoolean[Source] extends AllowedCast[Source, Boolean] {
    val targetDataType: DataType[Boolean] = BOOLEAN
  }

  private[types] class ToFloat16[Source] extends AllowedCast[Source, Half] {
    val targetDataType: DataType[Half] = FLOAT16
  }

  private[types] class ToFloat[Source] extends AllowedCast[Source, Float] {
    val targetDataType: DataType[Float] = FLOAT32
  }

  private[types] class ToDouble[Source] extends AllowedCast[Source, Double] {
    val targetDataType: DataType[Double] = FLOAT64
  }

  private[types] class ToBFloat16[Source] extends AllowedCast[Source, TruncatedHalf] {
    val targetDataType: DataType[TruncatedHalf] = BFLOAT16
  }

  private[types] class ToComplex64[Source] extends AllowedCast[Source, ComplexFloat] {
    val targetDataType: DataType[ComplexFloat] = COMPLEX64
  }

  private[types] class ToComplex128[Source] extends AllowedCast[Source, ComplexDouble] {
    val targetDataType: DataType[ComplexDouble] = COMPLEX128
  }

  private[types] class ToInt8[Source] extends AllowedCast[Source, Byte] {
    override val targetDataType: DataType[Byte] = INT8
  }

  private[types] class ToInt16[Source] extends AllowedCast[Source, Short] {
    override val targetDataType: DataType[Short] = INT16
  }

  private[types] class ToInt32[Source] extends AllowedCast[Source, Int] {
    override val targetDataType: DataType[Int] = INT32
  }

  private[types] class ToInt64[Source] extends AllowedCast[Source, Long] {
    override val targetDataType: DataType[Long] = INT64
  }

  private[types] class ToUInt8[Source] extends AllowedCast[Source, UByte] {
    override val targetDataType: DataType[UByte] = UINT8
  }

  private[types] class ToUInt16[Source] extends AllowedCast[Source, UShort] {
    override val targetDataType: DataType[UShort] = UINT16
  }

  private[types] class ToUInt32[Source] extends AllowedCast[Source, UInt] {
    override val targetDataType: DataType[UInt] = UINT32
  }

  private[types] class ToUInt64[Source] extends AllowedCast[Source, ULong] {
    override val targetDataType: DataType[ULong] = UINT64
  }

  private[types] class ToQInt8[Source] extends AllowedCast[Source, QByte] {
    override val targetDataType: DataType[QByte] = QINT8
  }

  private[types] class ToQInt16[Source] extends AllowedCast[Source, QShort] {
    override val targetDataType: DataType[QShort] = QINT16
  }

  private[types] class ToQInt32[Source] extends AllowedCast[Source, QInt] {
    override val targetDataType: DataType[QInt] = QINT32
  }

  private[types] class ToQUInt8[Source] extends AllowedCast[Source, QUByte] {
    override val targetDataType: DataType[QUByte] = QUINT8
  }

  private[types] class ToQUInt16[Source] extends AllowedCast[Source, QUShort] {
    override val targetDataType: DataType[QUShort] = QUINT16
  }
  
  trait FromSource[Source] {
    type T
    val targetDataType: DataType[T]
  }

  object FromSource {
    type Aux[Source, Target] = FromSource[Source] {
      type T = Target
    }

    implicit val int8ToFloat32: FromSource.Aux[Byte, Float]  = new ToFloat[Byte]
    implicit val int8ToFloat64: FromSource.Aux[Byte, Double] = new ToDouble[Byte]
    implicit val int8ToInt16  : FromSource.Aux[Byte, Short]  = new ToInt16[Byte]
    implicit val int8ToInt32  : FromSource.Aux[Byte, Int]    = new ToInt32[Byte]
    implicit val int8ToInt64  : FromSource.Aux[Byte, Long]   = new ToInt64[Byte]

    implicit val int16ToFloat32: FromSource.Aux[Short, Float]  = new ToFloat[Short]
    implicit val int16ToFloat64: FromSource.Aux[Short, Double] = new ToDouble[Short]
    implicit val int16ToInt32  : FromSource.Aux[Short, Int]    = new ToInt32[Short]
    implicit val int16ToInt64  : FromSource.Aux[Short, Long]   = new ToInt64[Short]

    implicit val int32ToFloat32: FromSource.Aux[Int, Float]  = new ToFloat[Int]
    implicit val int32ToFloat64: FromSource.Aux[Int, Double] = new ToDouble[Int]
    implicit val int32ToInt64  : FromSource.Aux[Int, Long]   = new ToInt64[Int]

    implicit val int64ToFloat32: FromSource.Aux[Long, Float]  = new ToFloat[Long]
    implicit val int64ToFloat64: FromSource.Aux[Long, Double] = new ToDouble[Long]
    implicit val int64ToInt64  : FromSource.Aux[Long, Long]   = new ToInt64[Long]

    implicit val uint8ToFloat32: FromSource.Aux[UByte, Float]  = new ToFloat[UByte]
    implicit val uint8ToFloat64: FromSource.Aux[UByte, Double] = new ToDouble[UByte]
    implicit val uint8ToInt16  : FromSource.Aux[UByte, Short]  = new ToInt16[UByte]
    implicit val uint8ToInt32  : FromSource.Aux[UByte, Int]    = new ToInt32[UByte]
    implicit val uint8ToInt64  : FromSource.Aux[UByte, Long]   = new ToInt64[UByte]
    implicit val uint8ToUInt16 : FromSource.Aux[UByte, UShort] = new ToUInt16[UByte]
    implicit val uint8ToUInt32 : FromSource.Aux[UByte, UInt]   = new ToUInt32[UByte]
    implicit val uint8ToUInt64 : FromSource.Aux[UByte, ULong]  = new ToUInt64[UByte]

    implicit val uint16ToFloat32: FromSource.Aux[UShort, Float]  = new ToFloat[UShort]
    implicit val uint16ToFloat64: FromSource.Aux[UShort, Double] = new ToDouble[UShort]
    implicit val uint16ToInt32  : FromSource.Aux[UShort, Int]    = new ToInt32[UShort]
    implicit val uint16ToInt64  : FromSource.Aux[UShort, Long]   = new ToInt64[UShort]
    implicit val uint16ToUInt32 : FromSource.Aux[UShort, UInt]   = new ToUInt32[UShort]
    implicit val uint16ToUInt64 : FromSource.Aux[UShort, ULong]  = new ToUInt64[UShort]

    implicit val uint32ToFloat32: FromSource.Aux[UInt, Float]  = new ToFloat[UInt]
    implicit val uint32ToFloat64: FromSource.Aux[UInt, Double] = new ToDouble[UInt]
    implicit val uint32ToInt64  : FromSource.Aux[UInt, Long]   = new ToInt64[UInt]
    implicit val uint32ToUInt64 : FromSource.Aux[UInt, ULong]  = new ToUInt64[UInt]

    implicit val uint64ToFloat32: FromSource.Aux[ULong, Float]  = new ToFloat[ULong]
    implicit val uint64ToFloat64: FromSource.Aux[ULong, Double] = new ToDouble[ULong]
    implicit val uint64ToInt64  : FromSource.Aux[ULong, Long]   = new ToInt64[ULong]
    implicit val uint64ToUInt64 : FromSource.Aux[ULong, ULong]  = new ToUInt64[ULong]

    private[types] class ToString[Source] extends FromSource[Source] {
      override type T = String
      val targetDataType: DataType[String] = STRING
    }

    private[types] class ToBoolean[Source] extends FromSource[Source] {
      override type T = Boolean
      val targetDataType: DataType[Boolean] = BOOLEAN
    }

    private[types] class ToFloat16[Source] extends FromSource[Source] {
      override type T = Half
      val targetDataType: DataType[Half] = FLOAT16
    }

    private[types] class ToFloat[Source] extends FromSource[Source] {
      override type T = Float
      val targetDataType: DataType[Float] = FLOAT32
    }

    private[types] class ToDouble[Source] extends FromSource[Source] {
      override type T = Double
      val targetDataType: DataType[Double] = FLOAT64
    }

    private[types] class ToBFloat16[Source] extends FromSource[Source] {
      override type T = TruncatedHalf
      val targetDataType: DataType[TruncatedHalf] = BFLOAT16
    }

    private[types] class ToComplex64[Source] extends FromSource[Source] {
      override type T = ComplexFloat
      val targetDataType: DataType[ComplexFloat] = COMPLEX64
    }

    private[types] class ToComplex128[Source] extends FromSource[Source] {
      override type T = ComplexDouble
      val targetDataType: DataType[ComplexDouble] = COMPLEX128
    }

    private[types] class ToInt8[Source] extends FromSource[Source] {
      override type T = Byte
      override val targetDataType: DataType[Byte] = INT8
    }

    private[types] class ToInt16[Source] extends FromSource[Source] {
      override type T = Short
      override val targetDataType: DataType[Short] = INT16
    }

    private[types] class ToInt32[Source] extends FromSource[Source] {
      override type T = Int
      override val targetDataType: DataType[Int] = INT32
    }

    private[types] class ToInt64[Source] extends FromSource[Source] {
      override type T = Long
      override val targetDataType: DataType[Long] = INT64
    }

    private[types] class ToUInt8[Source] extends FromSource[Source] {
      override type T = UByte
      override val targetDataType: DataType[UByte] = UINT8
    }

    private[types] class ToUInt16[Source] extends FromSource[Source] {
      override type T = UShort
      override val targetDataType: DataType[UShort] = UINT16
    }

    private[types] class ToUInt32[Source] extends FromSource[Source] {
      override type T = UInt
      override val targetDataType: DataType[UInt] = UINT32
    }

    private[types] class ToUInt64[Source] extends FromSource[Source] {
      override type T = ULong
      override val targetDataType: DataType[ULong] = UINT64
    }

    private[types] class ToQInt8[Source] extends FromSource[Source] {
      override type T = QByte
      override val targetDataType: DataType[QByte] = QINT8
    }

    private[types] class ToQInt16[Source] extends FromSource[Source] {
      override type T = QShort
      override val targetDataType: DataType[QShort] = QINT16
    }

    private[types] class ToQInt32[Source] extends FromSource[Source] {
      override type T = QInt
      override val targetDataType: DataType[QInt] = QINT32
    }

    private[types] class ToQUInt8[Source] extends FromSource[Source] {
      override type T = QUByte
      override val targetDataType: DataType[QUByte] = QUINT8
    }

    private[types] class ToQUInt16[Source] extends FromSource[Source] {
      override type T = QUShort
      override val targetDataType: DataType[QUShort] = QUINT16
    }
  }

  trait FromTarget[Target] {
    type S
    val targetDataType: DataType[Target]
  }

  object FromTarget {
    type Aux[Target, Source] = FromTarget[Target] {
      type S = Source
    }

    implicit val int8ToFloat32: FromTarget.Aux[Float, Byte]  = new ToFloat[Byte]
    implicit val int8ToFloat64: FromTarget.Aux[Double, Byte] = new ToFloat64[Byte]
    implicit val int8ToInt16  : FromTarget.Aux[Short, Byte]  = new ToInt16[Byte]
    implicit val int8ToInt32  : FromTarget.Aux[Int, Byte]    = new ToInt32[Byte]
    implicit val int8ToInt64  : FromTarget.Aux[Long, Byte]   = new ToInt64[Byte]

    implicit val int16ToFloat32: FromTarget.Aux[Float, Short]  = new ToFloat[Short]
    implicit val int16ToFloat64: FromTarget.Aux[Double, Short] = new ToFloat64[Short]
    implicit val int16ToInt32  : FromTarget.Aux[Int, Short]    = new ToInt32[Short]
    implicit val int16ToInt64  : FromTarget.Aux[Long, Short]   = new ToInt64[Short]

    implicit val int32ToFloat32: FromTarget.Aux[Float, Int]  = new ToFloat[Int]
    implicit val int32ToFloat64: FromTarget.Aux[Double, Int] = new ToFloat64[Int]
    implicit val int32ToInt64  : FromTarget.Aux[Long, Int]   = new ToInt64[Int]

    implicit val int64ToFloat32: FromTarget.Aux[Float, Long]  = new ToFloat[Long]
    implicit val int64ToFloat64: FromTarget.Aux[Double, Long] = new ToFloat64[Long]
    implicit val int64ToInt64  : FromTarget.Aux[Long, Long]   = new ToInt64[Long]

    implicit val uint8ToFloat32: FromTarget.Aux[Float, UByte]  = new ToFloat[UByte]
    implicit val uint8ToFloat64: FromTarget.Aux[Double, UByte] = new ToFloat64[UByte]
    implicit val uint8ToInt16  : FromTarget.Aux[Short, UByte]  = new ToInt16[UByte]
    implicit val uint8ToInt32  : FromTarget.Aux[Int, UByte]    = new ToInt32[UByte]
    implicit val uint8ToInt64  : FromTarget.Aux[Long, UByte]   = new ToInt64[UByte]
    implicit val uint8ToUInt16 : FromTarget.Aux[UShort, UByte] = new ToUInt16[UByte]
    implicit val uint8ToUInt32 : FromTarget.Aux[UInt, UByte]   = new ToUInt32[UByte]
    implicit val uint8ToUInt64 : FromTarget.Aux[ULong, UByte]  = new ToUInt64[UByte]

    implicit val uint16ToFloat32: FromTarget.Aux[Float, UShort]  = new ToFloat[UShort]
    implicit val uint16ToFloat64: FromTarget.Aux[Double, UShort] = new ToFloat64[UShort]
    implicit val uint16ToInt32  : FromTarget.Aux[Int, UShort]    = new ToInt32[UShort]
    implicit val uint16ToInt64  : FromTarget.Aux[Long, UShort]   = new ToInt64[UShort]
    implicit val uint16ToUInt32 : FromTarget.Aux[UInt, UShort]   = new ToUInt32[UShort]
    implicit val uint16ToUInt64 : FromTarget.Aux[ULong, UShort]  = new ToUInt64[UShort]

    implicit val uint32ToFloat32: FromTarget.Aux[Float, UInt]  = new ToFloat[UInt]
    implicit val uint32ToFloat64: FromTarget.Aux[Double, UInt] = new ToFloat64[UInt]
    implicit val uint32ToInt64  : FromTarget.Aux[Long, UInt]   = new ToInt64[UInt]
    implicit val uint32ToUInt64 : FromTarget.Aux[ULong, UInt]  = new ToUInt64[UInt]

    implicit val uint64ToFloat32: FromTarget.Aux[Float, ULong]  = new ToFloat[ULong]
    implicit val uint64ToFloat64: FromTarget.Aux[Double, ULong] = new ToFloat64[ULong]
    implicit val uint64ToInt64  : FromTarget.Aux[Long, ULong]   = new ToInt64[ULong]
    implicit val uint64ToUInt64 : FromTarget.Aux[ULong, ULong]  = new ToUInt64[ULong]

    private[types] class ToString[Source] extends FromTarget[String] {
      override type S = Source
      val targetDataType: DataType[String] = STRING
    }

    private[types] class ToBoolean[Source] extends FromTarget[Boolean] {
      override type S = Source
      val targetDataType: DataType[Boolean] = BOOLEAN
    }

    private[types] class ToFloat16[Source] extends FromTarget[Half] {
      override type S = Source
      val targetDataType: DataType[Half] = FLOAT16
    }

    private[types] class ToFloat[Source] extends FromTarget[Float] {
      override type S = Source
      val targetDataType: DataType[Float] = FLOAT32
    }

    private[types] class ToFloat64[Source] extends FromTarget[Double] {
      override type S = Source
      val targetDataType: DataType[Double] = FLOAT64
    }

    private[types] class ToBFloat16[Source] extends FromTarget[TruncatedHalf] {
      override type S = Source
      val targetDataType: DataType[TruncatedHalf] = BFLOAT16
    }

    private[types] class ToComplex64[Source] extends FromTarget[ComplexFloat] {
      override type S = Source
      val targetDataType: DataType[ComplexFloat] = COMPLEX64
    }

    private[types] class ToComplex128[Source] extends FromTarget[ComplexDouble] {
      override type S = Source
      val targetDataType: DataType[ComplexDouble] = COMPLEX128
    }

    private[types] class ToInt8[Source] extends FromTarget[Byte] {
      override type S = Source
      override val targetDataType: DataType[Byte] = INT8
    }

    private[types] class ToInt16[Source] extends FromTarget[Short] {
      override type S = Source
      override val targetDataType: DataType[Short] = INT16
    }

    private[types] class ToInt32[Source] extends FromTarget[Int] {
      override type S = Source
      override val targetDataType: DataType[Int] = INT32
    }

    private[types] class ToInt64[Source] extends FromTarget[Long] {
      override type S = Source
      override val targetDataType: DataType[Long] = INT64
    }

    private[types] class ToUInt8[Source] extends FromTarget[UByte] {
      override type S = Source
      override val targetDataType: DataType[UByte] = UINT8
    }

    private[types] class ToUInt16[Source] extends FromTarget[UShort] {
      override type S = Source
      override val targetDataType: DataType[UShort] = UINT16
    }

    private[types] class ToUInt32[Source] extends FromTarget[UInt] {
      override type S = Source
      override val targetDataType: DataType[UInt] = UINT32
    }

    private[types] class ToUInt64[Source] extends FromTarget[ULong] {
      override type S = Source
      override val targetDataType: DataType[ULong] = UINT64
    }

    private[types] class ToQInt8[Source] extends FromTarget[QByte] {
      override type S = Source
      override val targetDataType: DataType[QByte] = QINT8
    }

    private[types] class ToQInt16[Source] extends FromTarget[QShort] {
      override type S = Source
      override val targetDataType: DataType[QShort] = QINT16
    }

    private[types] class ToQInt32[Source] extends FromTarget[QInt] {
      override type S = Source
      override val targetDataType: DataType[QInt] = QINT32
    }

    private[types] class ToQUInt8[Source] extends FromTarget[QUByte] {
      override type S = Source
      override val targetDataType: DataType[QUByte] = QUINT8
    }

    private[types] class ToQUInt16[Source] extends FromTarget[QUShort] {
      override type S = Source
      override val targetDataType: DataType[QUShort] = QUINT16
    }
  }
}
