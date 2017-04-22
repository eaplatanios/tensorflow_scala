package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.OpSpecification

/**
  * @author Emmanouil Antonios Platanios
  */
trait Implicits extends LowPriorityImplicits {
  @inline implicit def valueToSupportedScalaType(value: Boolean): Bool = Bool(value)
  @inline implicit def valueToSupportedScalaType(value: String): Str = Str(value)
  @inline implicit def valueToSupportedScalaType(value: Float): Float32 = Float32(value)
  @inline implicit def valueToSupportedScalaType(value: Double): Float64 = Float64(value)
  @inline implicit def valueToSupportedScalaType(value: Byte): Int8 = Int8(value)
  @inline implicit def valueToSupportedScalaType(value: Short): Int16 = Int16(value)
  @inline implicit def valueToSupportedScalaType(value: Int): Int32 = Int32(value)
  @inline implicit def valueToSupportedScalaType(value: Long): Int64 = Int64(value)
  @inline implicit def valueToSupportedScalaType(value: Char): UInt16 = UInt16(value)

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
  implicit def variableToOpOutput(variable: Variable): Op.Output = variable.value
}

trait LowPriorityImplicits {
  implicit def scalaValueToTensor(value: Boolean): Tensor = Tensor.fill(dataType = DataType.Bool)(Bool(value))
  implicit def scalaValueToTensor(value: String): Tensor = Tensor.fill(dataType = DataType.Str)(Str(value))
  implicit def scalaValueToTensor(value: Float): Tensor = Tensor.fill(dataType = DataType.Float32)(Float32(value))
  implicit def scalaValueToTensor(value: Double): Tensor = Tensor.fill(dataType = DataType.Float64)(Float64(value))
  implicit def scalaValueToTensor(value: Byte): Tensor = Tensor.fill(dataType = DataType.Int8)(Int8(value))
  implicit def scalaValueToTensor(value: Short): Tensor = Tensor.fill(dataType = DataType.Int16)(Int16(value))
  implicit def scalaValueToTensor(value: Int): Tensor = Tensor.fill(dataType = DataType.Int32)(Int32(value))
  implicit def scalaValueToTensor(value: Long): Tensor = Tensor.fill(dataType = DataType.Int64)(Int64(value))
  implicit def scalaValueToTensor(value: Char): Tensor = Tensor.fill(dataType = DataType.UInt16)(UInt16(value))
}
