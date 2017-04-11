package org.platanios.tensorflow.api

import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

/** Represents the data type of the elements in a tensor.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait DataType {
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
    val nativeLibrarySize = NativeLibrary.dataTypeSize(base.cValue)
    if (nativeLibrarySize == 0)
      None
    else
      Some(nativeLibrarySize)
  }

  /** Returns `true` if this data type represents a non-quantized floating-point data type. */
  def isFloatingPoint: Boolean = !isQuantized && DataType.floatingPointDataTypes.contains(base)

  /** Returns `true` if this data type represents a complex data types. */
  def isComplex: Boolean = DataType.complexDataTypes.contains(base)

  /** Returns `true` if this data type represents a non-quantized integer data type. */
  def isInteger: Boolean = !isQuantized && DataType.integerDataTypes.contains(base)

  /** Returns `true` if this data type represents a quantized data type. */
  def isQuantized: Boolean = DataType.quantizedDataTypes.contains(base)

  /** Returns `true` if this data type represents a non-quantized unsigned data type. */
  def isUnsigned: Boolean = !isQuantized && DataType.unsignedDataTypes.contains(base)

  /** Returns `true` if this data type represents a numeric data type. */
  def isNumeric: Boolean = DataType.numericDataTypes.contains(base)

  /** Returns `true` if this data type represents a boolean data type. */
  def isBoolean: Boolean = base == DataType.Boolean

  /** Returns `true` if this data type represents a reference data type. */
  def isRef: Boolean = cValue > 100

  /** Returns a reference data type based on this data type. */
  def ref: DataType = if (isRef) this else DataType.fromCValue(cValue + 100)

  /** Returns a non-reference data type based on this data type. */
  def base: DataType = if (isRef) DataType.fromCValue(cValue - 100) else this

  /** Returns a data type that corresponds to this data type's real part. */
  def real: DataType = this match {
    case DataType.Complex64 => DataType.Float32
    case DataType.Complex128 => DataType.Float64
    case DataType.Complex64Ref => DataType.Float32Ref
    case DataType.Complex128Ref => DataType.Float64Ref
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
  object Float16 extends DataType {
    override val name: String = "Float16"
    override val cValue: Int = 19
    override val byteSize: Int = 2
  }

  object Float32 extends DataType {
    override val name: String = "Float32"
    override val cValue: Int = 1
    override val byteSize: Int = 4
  }

  object Float64 extends DataType {
    override val name: String = "Float64"
    override val cValue: Int = 2
    override val byteSize: Int = 8
  }

  object BFloat16 extends DataType {
    override val name: String = "BFloat16"
    override val cValue: Int = 14
    override val byteSize: Int = 2
  }

  object Complex64 extends DataType {
    override val name: String = "Complex64"
    override val cValue: Int = 8
    override val byteSize: Int = 8
  }

  object Complex128 extends DataType {
    override val name: String = "Complex128"
    override val cValue: Int = 18
    override val byteSize: Int = 16
  }

  object Int8 extends DataType {
    override val name: String = "Int8"
    override val cValue: Int = 6
    override val byteSize: Int = 1
  }

  object Int16 extends DataType {
    override val name: String = "Int16"
    override val cValue: Int = 5
    override val byteSize: Int = 2
  }

  object Int32 extends DataType {
    override val name: String = "Int32"
    override val cValue: Int = 3
    override val byteSize: Int = 4
  }

  object Int64 extends DataType {
    override val name: String = "Int64"
    override val cValue: Int = 9
    override val byteSize: Int = 8
  }

  object UInt8 extends DataType {
    override val name: String = "UInt8"
    override val cValue: Int = 4
    override val byteSize: Int = 1
  }

  object UInt16 extends DataType {
    override val name: String = "UInt16"
    override val cValue: Int = 17
    override val byteSize: Int = 2
  }

  object QInt8 extends DataType {
    override val name: String = "QInt8"
    override val cValue: Int = 11
    override val byteSize: Int = 1
  }

  object QInt16 extends DataType {
    override val name: String = "QInt16"
    override val cValue: Int = 15
    override val byteSize: Int = 2
  }

  object QInt32 extends DataType {
    override val name: String = "QInt32"
    override val cValue: Int = 13
    override val byteSize: Int = 4
  }

  object QUInt8 extends DataType {
    override val name: String = "QUInt8"
    override val cValue: Int = 12
    override val byteSize: Int = 1
  }

  object QUInt16 extends DataType {
    override val name: String = "QUInt16"
    override val cValue: Int = 16
    override val byteSize: Int = 2
  }

  object Boolean extends DataType {
    override val name: String = "Boolean"
    override val cValue: Int = 10
    override val byteSize: Int = 1
  }

  object String extends DataType {
    override val name: String = "String"
    override val cValue: Int = 7
    override val byteSize: Int = -1
  }

  object Resource extends DataType {
    override val name: String = "Resource"
    override val cValue: Int = 20
    override val byteSize: Int = 1
  }

  object Float16Ref extends DataType {
    override val name: String = "Float16Ref"
    override val cValue: Int = 119
    override val byteSize: Int = 2
  }

  object Float32Ref extends DataType {
    override val name: String = "Float32Ref"
    override val cValue: Int = 101
    override val byteSize: Int = 4
  }

  object Float64Ref extends DataType {
    override val name: String = "Float64Ref"
    override val cValue: Int = 102
    override val byteSize: Int = 8
  }

  object BFloat16Ref extends DataType {
    override val name: String = "BFloat16Ref"
    override val cValue: Int = 114
    override val byteSize: Int = 2
  }

  object Complex64Ref extends DataType {
    override val name: String = "Complex64Ref"
    override val cValue: Int = 108
    override val byteSize: Int = 8
  }

  object Complex128Ref extends DataType {
    override val name: String = "Complex128Ref"
    override val cValue: Int = 118
    override val byteSize: Int = 16
  }

  object Int8Ref extends DataType {
    override val name: String = "Int8Ref"
    override val cValue: Int = 106
    override val byteSize: Int = 1
  }

  object Int16Ref extends DataType {
    override val name: String = "Int16Ref"
    override val cValue: Int = 105
    override val byteSize: Int = 2
  }

  object Int32Ref extends DataType {
    override val name: String = "Int32Ref"
    override val cValue: Int = 103
    override val byteSize: Int = 4
  }

  object Int64Ref extends DataType {
    override val name: String = "Int64Ref"
    override val cValue: Int = 109
    override val byteSize: Int = 8
  }

  object UInt8Ref extends DataType {
    override val name: String = "UInt8Ref"
    override val cValue: Int = 104
    override val byteSize: Int = 1
  }

  object UInt16Ref extends DataType {
    override val name: String = "UInt16Ref"
    override val cValue: Int = 117
    override val byteSize: Int = 2
  }

  object QInt8Ref extends DataType {
    override val name: String = "QInt8Ref"
    override val cValue: Int = 111
    override val byteSize: Int = 1
  }

  object QInt16Ref extends DataType {
    override val name: String = "QInt16Ref"
    override val cValue: Int = 115
    override val byteSize: Int = 2
  }

  object QInt32Ref extends DataType {
    override val name: String = "QInt32Ref"
    override val cValue: Int = 113
    override val byteSize: Int = 4
  }

  object QUInt8Ref extends DataType {
    override val name: String = "QUInt8Ref"
    override val cValue: Int = 112
    override val byteSize: Int = 1
  }

  object QUInt16Ref extends DataType {
    override val name: String = "QUInt16Ref"
    override val cValue: Int = 116
    override val byteSize: Int = 2
  }

  object BooleanRef extends DataType {
    override val name: String = "BooleanRef"
    override val cValue: Int = 110
    override val byteSize: Int = 1
  }

  object StringRef extends DataType {
    override val name: String = "StringRef"
    override val cValue: Int = 107
    override val byteSize: Int = -1
  }

  object ResourceRef extends DataType {
    override val name: String = "ResourceRef"
    override val cValue: Int = 120
    override val byteSize: Int = 1
  }

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
    case Float16Ref.cValue => Float16Ref
    case Float32Ref.cValue => Float32Ref
    case Float64Ref.cValue => Float64Ref
    case BFloat16Ref.cValue => BFloat16Ref
    case Complex64Ref.cValue => Complex64Ref
    case Complex128Ref.cValue => Complex128Ref
    case Int8Ref.cValue => Int8Ref
    case Int16Ref.cValue => Int16Ref
    case Int32Ref.cValue => Int32Ref
    case Int64Ref.cValue => Int64Ref
    case UInt8Ref.cValue => UInt8Ref
    case UInt16Ref.cValue => UInt16Ref
    case QInt8Ref.cValue => QInt8Ref
    case QInt16Ref.cValue => QInt16Ref
    case QInt32Ref.cValue => QInt32Ref
    case QUInt8Ref.cValue => QUInt8Ref
    case QUInt16Ref.cValue => QUInt16Ref
    case BooleanRef.cValue => BooleanRef
    case StringRef.cValue => StringRef
    case ResourceRef.cValue => ResourceRef
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
    case "Float16Ref" => Float16Ref
    case "Float32Ref" => Float32Ref
    case "Float64Ref" => Float64Ref
    case "BFloat16Ref" => BFloat16Ref
    case "Complex64Ref" => Complex64Ref
    case "Complex128Ref" => Complex128Ref
    case "Int8Ref" => Int8Ref
    case "Int16Ref" => Int16Ref
    case "Int32Ref" => Int32Ref
    case "Int64Ref" => Int64Ref
    case "UInt8Ref" => UInt8Ref
    case "UInt16Ref" => UInt16Ref
    case "QInt8Ref" => QInt8Ref
    case "QInt16Ref" => QInt16Ref
    case "QInt32Ref" => QInt32Ref
    case "QUInt8Ref" => QUInt8Ref
    case "QUInt16Ref" => QUInt16Ref
    case "BooleanRef" => BooleanRef
    case "StringRef" => StringRef
    case "ResourceRef" => ResourceRef
    case value => throw new IllegalArgumentException(
      s"Data type name '$value' is not recognized in Scala (TensorFlow version ${NativeLibrary.version}).")
  }

  /** Returns the [[DataType]] of the provided value.
    *
    * @param  value Value whose data type to return.
    * @return Data type of the provided value.
    * @throws IllegalArgumentException If the data type of the provided value is not supported as a TensorFlow
    *                                  [[DataType]].
    */
  @throws[IllegalArgumentException]
  private[api] def dataTypeOf(value: Any): DataType = {
    value match {
      // Array[Byte] is a DataType.String scalar
      case value: Array[Byte] =>
        if (value.length == 0)
          throw new IllegalArgumentException("Cannot create a tensor of size 0.")
        DataType.String
      case value: Array[_] =>
        if (value.length == 0)
          throw new IllegalArgumentException("Cannot create a tensor of size 0.")
        dataTypeOf(value(0)) // TODO: What if different array elements have different values?
      case _: Float => DataType.Float32
      case _: Double => DataType.Float64
      case _: Int => DataType.Int32
      case _: Long => DataType.Int64
      case _: Boolean => DataType.Boolean
      case _ => throw new IllegalArgumentException(s"Cannot create a tensor of type '${value.getClass.getName}'.")
    }
  }
}
