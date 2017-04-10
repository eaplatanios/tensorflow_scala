package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

/** Represents the data type of the elements in a tensor.
  *
  * @param  name   Name of the data type (mainly useful for logging purposes).
  * @param  cValue Integer representing this data type in the `TF_DataType` enum of the TensorFlow C API.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed case class DataType private (name: String, private[api] val cValue: Int) {
  /** Size in bytes of each value with this data type. Returns `None` if the size is not available. */
  lazy val byteSize: Option[Int] = {
    if (this == DataType.resource) {
      Some(1)
    } else {
      val nativeLibrarySize = NativeLibrary.dataTypeSize(base.cValue)
      if (nativeLibrarySize == 0)
        None
      else
        Some(nativeLibrarySize)
    }
  }

  /** Returns `true` if this data type represents a boolean data type. */
  def isBoolean: Boolean = base == DataType.boolean

  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
  def isFloatingPoint: Boolean = !isQuantized && DataType.floatingPointDataTypes.contains(base)

  /** Returns `true` if this data type represents a non-quantized integer data type. */
  def isInteger: Boolean = !isQuantized && DataType.integerDataTypes.contains(base)

  /** Returns `true` if this data type represents a complex data types. */
  def isComplex: Boolean = DataType.complexDataTypes.contains(base)

  /** Returns `true` if this data type represents a quantized data type. */
  def isQuantized: Boolean = DataType.quantizedDataTypes.contains(base)

  /** Returns `true` if this data type represents a non-quantized unsigned data type. */
  def isUnsigned: Boolean = !isQuantized && DataType.unsignedDataTypes.contains(base)

  /** Returns `true` if this data type represents a reference data type. */
  def isRef: Boolean = cValue > 100

  /** Returns a reference data type based on this data type. */
  def ref: DataType = if (isRef) this else DataType.fromCValue(cValue + 100)

  /** Returns a non-reference data type based on this data type. */
  def base: DataType = if (isRef) DataType.fromCValue(cValue - 100) else this

  /** Returns a data type that corresponds to this data type's real part. */
  def real: DataType = this match {
    case DataType.complex64 => DataType.float32
    case DataType.complex128 => DataType.float64
    case DataType.complex64Ref => DataType.float32Ref
    case DataType.complex128Ref => DataType.float64Ref
    case _ => this
  }

  /** Returns `true` if the `other` data type can be converted to this data type.
    *
    * The conversion rules are as follows:
    * {{{
    *   DataType(cValue = T)      .isCompatibleWith(DataType(cValue = T))       == true
    *   DataType(cValue = T)      .isCompatibleWith(DataType(cValue = T).asRef) == true
    *   DataType(cValue = T).asRef.isCompatibleWith(DataType(cValue = T))       == false
    *   DataType(cValue = T).asRef.isCompatibleWith(DataType(cValue = T).asRef) == true
    * }}}
    *
    * @param  other Data type to check compatibility with.
    * @return Result of the compatibility check.
    */
  def isCompatibleWith(other: DataType): Boolean = cValue == other.cValue || cValue == other.base.cValue

  override def toString: String = name
}

/** Contains all supported data types. */
object DataType {
  val float16: DataType = DataType(name = "float16", cValue = 19)
  val float32: DataType = DataType(name = "float32", cValue = 1)
  val float64: DataType = DataType(name = "float64", cValue = 2)
  val bfloat16: DataType = DataType(name = "bfloat16", cValue = 14)
  val complex64: DataType = DataType(name = "complex64", cValue = 8)
  val complex128: DataType = DataType(name = "complex128", cValue = 18)
  val int8: DataType = DataType(name = "int8", cValue = 6)
  val int16: DataType = DataType(name = "int16", cValue = 5)
  val int32: DataType = DataType(name = "int32", cValue = 3)
  val int64: DataType = DataType(name = "int64", cValue = 9)
  val uint8: DataType = DataType(name = "uint8", cValue = 4)
  val uint16: DataType = DataType(name = "uint16", cValue = 17)
  val qint8: DataType = DataType(name = "qint8", cValue = 11)
  val qint16: DataType = DataType(name = "qint16", cValue = 15)
  val qint32: DataType = DataType(name = "qint32", cValue = 13)
  val quint8: DataType = DataType(name = "quint8", cValue = 12)
  val quint16: DataType = DataType(name = "quint16", cValue = 16)
  val boolean: DataType = DataType(name = "Bbolean", cValue = 10)
  val string: DataType = DataType(name = "string", cValue = 7)
  val resource: DataType = DataType(name = "resource", cValue = 20)
  val float16Ref: DataType = DataType(name = "float16_ref", cValue = 119)
  val float32Ref: DataType = DataType(name = "float32_ref", cValue = 101)
  val float64Ref: DataType = DataType(name = "float64_ref", cValue = 102)
  val bfloat16Ref: DataType = DataType(name = "bfloat16_ref", cValue = 114)
  val complex64Ref: DataType = DataType(name = "complex64_ref", cValue = 108)
  val complex128Ref: DataType = DataType(name = "complex128_ref", cValue = 118)
  val int8Ref: DataType = DataType(name = "int8_ref", cValue = 106)
  val int16Ref: DataType = DataType(name = "int16_ref", cValue = 105)
  val int32Ref: DataType = DataType(name = "int32_ref", cValue = 103)
  val int64Ref: DataType = DataType(name = "int64_ref", cValue = 109)
  val uint8Ref: DataType = DataType(name = "uint8_ref", cValue = 104)
  val uint16Ref: DataType = DataType(name = "uint16_ref", cValue = 117)
  val qint8Ref: DataType = DataType(name = "qint8_ref", cValue = 111)
  val qint16Ref: DataType = DataType(name = "qint16_ref", cValue = 115)
  val qint32Ref: DataType = DataType(name = "qint32_ref", cValue = 113)
  val quint8Ref: DataType = DataType(name = "quint8_ref", cValue = 112)
  val quint16Ref: DataType = DataType(name = "quint16_ref", cValue = 116)
  val booleanRef: DataType = DataType(name = "boolean_ref", cValue = 110)
  val stringRef: DataType = DataType(name = "string_ref", cValue = 107)
  val resourceRef: DataType = DataType(name = "resource_ref", cValue = 120)

  /** Set of all floating-point data types. */
  private val floatingPointDataTypes = Set(float16, float32, float64, bfloat16)

  /** Set of all integer data types. */
  private val integerDataTypes = Set(int8, int16, int32, int64, uint8, uint16, qint8, qint16, qint32, quint8, quint16)

  /** Set of all complex data types. */
  private val complexDataTypes = Set(complex64, complex128)

  /** Set of all quantized data types. */
  private val quantizedDataTypes = Set(bfloat16, qint8, qint16, qint32, quint8, quint16)

  /** Set of all unsigned data types. */
  private val unsignedDataTypes = Set(uint8, uint16, quint8, quint16)

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
    case float16.cValue => float16
    case float32.cValue => float32
    case float64.cValue => float64
    case bfloat16.cValue => bfloat16
    case complex64.cValue => complex64
    case complex128.cValue => complex128
    case int8.cValue => int8
    case int16.cValue => int16
    case int32.cValue => int32
    case int64.cValue => int64
    case uint8.cValue => uint8
    case uint16.cValue => uint16
    case qint8.cValue => qint8
    case qint16.cValue => qint16
    case qint32.cValue => qint32
    case quint8.cValue => quint8
    case quint16.cValue => quint16
    case boolean.cValue => boolean
    case string.cValue => string
    case resource.cValue => resource
    case float16Ref.cValue => float16Ref
    case float32Ref.cValue => float32Ref
    case float64Ref.cValue => float64Ref
    case bfloat16Ref.cValue => bfloat16Ref
    case complex64Ref.cValue => complex64Ref
    case complex128Ref.cValue => complex128Ref
    case int8Ref.cValue => int8Ref
    case int16Ref.cValue => int16Ref
    case int32Ref.cValue => int32Ref
    case int64Ref.cValue => int64Ref
    case uint8Ref.cValue => uint8Ref
    case uint16Ref.cValue => uint16Ref
    case qint8Ref.cValue => qint8Ref
    case qint16Ref.cValue => qint16Ref
    case qint32Ref.cValue => qint32Ref
    case quint8Ref.cValue => quint8Ref
    case quint16Ref.cValue => quint16Ref
    case booleanRef.cValue => booleanRef
    case stringRef.cValue => stringRef
    case resourceRef.cValue => resourceRef
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
    case "float16" => float16
    case "float32" => float32
    case "float64" => float64
    case "bfloat16" => bfloat16
    case "complex64" => complex64
    case "complex128" => complex128
    case "int8" => int8
    case "int16" => int16
    case "int32" => int32
    case "int64" => int64
    case "uint8" => uint8
    case "uint16" => uint16
    case "qint8" => qint8
    case "qint16" => qint16
    case "qint32" => qint32
    case "quint8" => quint8
    case "quint16" => quint16
    case "boolean" => boolean
    case "string" => string
    case "resource" => resource
    case "float16_ref" => float16Ref
    case "float32_ref" => float32Ref
    case "float64_ref" => float64Ref
    case "bfloat16_ref" => bfloat16Ref
    case "complex64_ref" => complex64Ref
    case "complex128_ref" => complex128Ref
    case "int8_ref" => int8Ref
    case "int16_ref" => int16Ref
    case "int32_ref" => int32Ref
    case "int64_ref" => int64Ref
    case "uint8_ref" => uint8Ref
    case "uint16_ref" => uint16Ref
    case "qint8_ref" => qint8Ref
    case "qint16_ref" => qint16Ref
    case "qint32_ref" => qint32Ref
    case "quint8_ref" => quint8Ref
    case "quint16_ref" => quint16Ref
    case "boolean_ref" => booleanRef
    case "string_ref" => stringRef
    case "resource_ref" => resourceRef
    case value => throw new IllegalArgumentException(
      s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
  }
}
