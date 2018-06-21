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

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
sealed trait SupportedType[+T] {
  type D <: DataType

  @inline def dataType: D

  @throws[InvalidDataTypeException]
  @inline def cast(value: Any): T = {
    throw InvalidDataTypeException("The Scala type of this data type is not supported.")
  }
}

object SupportedType {
  type Aux[+T, DD <: DataType] = SupportedType[T] {
    type D = DD
  }

  implicit class SupportedTypeOps[T, D <: DataType](val value: T)(implicit
      evSupported: SupportedType.Aux[T, D]
  ) {
    @inline def dataType: D = evSupported.dataType
    @inline def cast[DV <: DataType](dataType: DV): DV#ScalaType = {
      dataType.evSupportedType.cast(value)
    }
  }

  implicit val stringIsSupportedType: SupportedType.Aux[String, STRING] = new SupportedType[String] {
    override type D = STRING
    @inline override def dataType: STRING = STRING
    @inline override def cast(value: Any): String = value.toString
  }

  implicit val booleanIsSupportedType: SupportedType.Aux[Boolean, BOOLEAN] = new SupportedType[Boolean] {
    override type D = BOOLEAN
    @inline override def dataType: BOOLEAN = BOOLEAN
    @inline override def cast(value: Any): Boolean = value match {
      case value: Boolean => value
      case _ => throw InvalidDataTypeException("Cannot convert the provided value to a boolean.")
    }
  }

  implicit val floatIsSupportedType: SupportedType.Aux[Float, FLOAT32] = new SupportedType[Float] {
    override type D = FLOAT32
    @inline override def dataType: FLOAT32 = FLOAT32
    @inline override def cast(value: Any): Float = value match {
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
  }

  implicit val doubleIsSupportedType: SupportedType.Aux[Double, FLOAT64] = new SupportedType[Double] {
    override type D = FLOAT64
    @inline override def dataType: FLOAT64 = FLOAT64
    @inline override def cast(value: Any): Double = value match {
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
  }

  implicit val byteIsSupportedType: SupportedType.Aux[Byte, INT8] = new SupportedType[Byte] {
    override type D = INT8
    @inline override def dataType: INT8 = INT8
    @inline override def cast(value: Any): Byte = value match {
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
  }

  implicit val shortIsSupportedType: SupportedType.Aux[Short, INT16] = new SupportedType[Short] {
    override type D = INT16
    @inline override def dataType: INT16 = INT16
    @inline override def cast(value: Any): Short = value match {
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
  }

  implicit val intIsSupportedType: SupportedType.Aux[Int, INT32] = new SupportedType[Int] {
    override type D = INT32
    @inline override def dataType: INT32 = INT32
    @inline override def cast(value: Any): Int = value match {
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
  }

  implicit val longIsSupportedType: SupportedType.Aux[Long, INT64] = new SupportedType[Long] {
    override type D = INT64
    @inline override def dataType: INT64 = INT64
    @inline override def cast(value: Any): Long = value match {
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
  }

  implicit val uByteIsSupportedType: SupportedType.Aux[UByte, UINT8] = new SupportedType[UByte] {
    override type D = UINT8
    @inline override def dataType: UINT8 = UINT8
    @inline override def cast(value: Any): UByte = value match {
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
  }

  implicit val uShortIsSupportedType: SupportedType.Aux[UShort, UINT16] = new SupportedType[UShort] {
    override type D = UINT16
    @inline override def dataType: UINT16 = UINT16
    @inline override def cast(value: Any): UShort = value match {
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
  }
}
