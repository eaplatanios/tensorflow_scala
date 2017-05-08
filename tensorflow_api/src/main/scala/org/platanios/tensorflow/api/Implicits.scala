package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.OpSpecification

import spire.math.{UByte, UShort}

/**
  * @author Emmanouil Antonios Platanios
  */
trait Implicits extends LowPriorityImplicits {
//  implicit val SupportedTypeBoolean = types.SupportedType.SupportedTypeBoolean
//  implicit val SupportedTypeString = types.SupportedType.SupportedTypeString
//  implicit val SupportedTypeFloat = types.SupportedType.SupportedTypeFloat
//  implicit val SupportedTypeDouble = types.SupportedType.SupportedTypeDouble
//  implicit val SupportedTypeByte = types.SupportedType.SupportedTypeByte
//  implicit val SupportedTypeShort = types.SupportedType.SupportedTypeShort
//  implicit val SupportedTypeInt = types.SupportedType.SupportedTypeInt
//  implicit val SupportedTypeLong = types.SupportedType.SupportedTypeLong
//  implicit val SupportedTypeUByte = types.SupportedType.SupportedTypeUByte
//  implicit val SupportedTypeUShort = types.SupportedType.SupportedTypeUShort
//
//  @inline implicit def valueToSupportedScalaType(value: Boolean): Bool = Bool(value)
//  @inline implicit def valueToSupportedScalaType(value: String): Str = Str(value)
//  @inline implicit def valueToSupportedScalaType(value: Float): Float32 = Float32(value)
//  @inline implicit def valueToSupportedScalaType(value: Double): Float64 = Float64(value)
//  @inline implicit def valueToSupportedScalaType(value: Byte): Int8 = Int8(value)
//  @inline implicit def valueToSupportedScalaType(value: Short): Int16 = Int16(value)
//  @inline implicit def valueToSupportedScalaType(value: Int): Int32 = Int32(value)
//  @inline implicit def valueToSupportedScalaType(value: Long): Int64 = Int64(value)
//  @inline implicit def valueToSupportedScalaType(value: Char): UInt16 = UInt16(value)
//
//  @inline implicit def supportedScalaTypeToValue(value: Bool): Boolean = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Str): String = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Float32): Float = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Float64): Double = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Int8): Byte = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Int16): Short = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Int32): Int = value.value
//  @inline implicit def supportedScalaTypeToValue(value: Int64): Long = value.value
//  @inline implicit def supportedScalaTypeToValue(value: UInt16): Char = value.value

  implicit def intToIndex(index: Int): Index = Indexer.intToIndex(index)
  implicit def intToIndexerConstructionWithOneNumber(n: Int): IndexerConstructionWithOneNumber =
    Indexer.intToIndexerConstructionWithOneNumber(n)
  implicit def indexerConstructionWithOneNumberToIndex(construction: IndexerConstructionWithOneNumber): Index =
    Indexer.indexerConstructionWithOneNumberToIndex(construction)
  implicit def indexerConstructionWithTwoNumbersToSlice(construction: IndexerConstructionWithTwoNumbers): Slice =
    Indexer.indexerConstructionWithTwoNumbersToSlice(construction)
  implicit def indexerConstructionWithThreeNumbersToSlice(construction: IndexerConstructionWithThreeNumbers): Slice =
    Indexer.indexerConstructionWithThreeNumbersToSlice(construction)

  implicit def deviceImplicitConversion(device: String): OpSpecification => String = Op.deviceImplicitConversion(device)

  implicit def opOutputToInitialValueFunction(opOutput: Op.Output): () => Op.Output = () => opOutput
  implicit def variableToOpOutput[T: SupportedType](variable: Variable[T]): Op.Output = variable.toOpOutput
}

trait LowPriorityImplicits {
  implicit def tensorToOpOutput[T: SupportedType](tensor: Tensor[T]): Op.Output = ops.Basic.constant(tensor)

  implicit def shapeToTensor(shape: Shape): Tensor[Int] = shape.toTensor()
  implicit def shapeToOpOutput(shape: Shape): Op.Output = ops.Basic.constant(shape.toTensor())

  implicit def scalaValueToTensor(value: Boolean): Tensor[Boolean] = Tensor.fill(dataType = DataType.Bool)(value)
  implicit def scalaValueToTensor(value: String): Tensor[String] = Tensor.fill(dataType = DataType.Str)(value)
  implicit def scalaValueToTensor(value: Float): Tensor[Float] = Tensor.fill(dataType = DataType.Float32)(value)
  implicit def scalaValueToTensor(value: Double): Tensor[Double] = Tensor.fill(dataType = DataType.Float64)(value)
  implicit def scalaValueToTensor(value: Byte): Tensor[Byte] = Tensor.fill(dataType = DataType.Int8)(value)
  implicit def scalaValueToTensor(value: Short): Tensor[Short] = Tensor.fill(dataType = DataType.Int16)(value)
  implicit def scalaValueToTensor(value: Int): Tensor[Int] = Tensor.fill(dataType = DataType.Int32)(value)
  implicit def scalaValueToTensor(value: Long): Tensor[Long] = Tensor.fill(dataType = DataType.Int64)(value)
  implicit def scalaValueToTensor(value: UShort): Tensor[UShort] = Tensor.fill(dataType = DataType.UInt16)(value)

  implicit def scalaValueToOpOutput(value: Boolean): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: String): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: Float): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: Double): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: Byte): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: Short): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: Int): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: Long): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
  implicit def scalaValueToOpOutput(value: UShort): Op.Output = ops.Basic.constant(scalaValueToTensor(value))
}
