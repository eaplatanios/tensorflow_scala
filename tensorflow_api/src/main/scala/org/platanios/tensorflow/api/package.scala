package org.platanios.tensorflow

import spire.math.{UByte, UShort}

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api {
  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = Tensor.RowMajorOrder

  //region Data Types

  type Float16 = Float
  type Float32 = Float
  type Float64 = Double
  type BFloat16 = Float
  // type Complex64 = Complex[Float]
  // type Complex128 = Complex[Double]
  type Int8 = Byte
  type Int16 = Short
  type Int32 = Int
  type Int64 = Long
  type UInt8 = UByte
  type UInt16 = UShort
  type QInt8 = Byte
  type QInt16 = Short
  type QInt32 = Int
  type QUInt8 = UByte
  type QUInt16 = UShort

  def UInt8(number: Byte): UInt8 = UByte(number)
  def UInt16(number: Short): UInt16 = UShort(number)
  // def Complex[@specialized(Float, Double) T](real: T, imag: T): Complex[T] = Complex(real = real, imag = imag)

  // implicit def intToComplex(n: Int): Complex[Double] = n
  // implicit def longToComplex(n: Long): Complex[Double] = n
  // implicit def floatToComplex(n: Float): Complex[Float] = n
  // implicit def doubleToComplex(n: Double): Complex[Double] = n

  //endregion Data Types

  type Op = ops.Op
  val Op = ops.Op

  private[api] val COLOCATION_OPS_ATTRIBUTE_NAME = "_class"
  private[api] val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  private[api] val VALID_OP_NAME_REGEX: Regex = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[api] val VALID_NAME_SCOPE_REGEX: Regex = "^[A-Za-z0-9_.\\-/]*$".r

  trait Closeable {
    def close(): Unit
  }

  def using[T <: Closeable, R](resource: T)(block: T => R): R = {
    try {
      block(resource)
    } finally {
      if (resource != null)
        resource.close()
    }
  }

  //region Op Creation Implicits

  import org.platanios.tensorflow.api.ops.OpCreationContext

  private[api] val defaultGraph: Graph = Graph()
  private[api] implicit val opCreationContext: DynamicVariable[OpCreationContext] =
    new DynamicVariable[OpCreationContext](OpCreationContext(graph = defaultGraph))
  private[api] implicit def dynamicVariableToOpCreationContext(
      context: DynamicVariable[OpCreationContext]): OpCreationContext = context.value

  //endregion Op Creation Implicits

  //region Indexer Implicits

  val --- : Indexer = Indexer.---
  val :: : Slice = Slice.::
  implicit def intToIndex(index: Int): Index = Indexer.intToIndex(index)
  implicit def intToIndexerConstructionWithOneNumber(n: Int): IndexerConstructionWithOneNumber =
    Indexer.intToIndexerConstructionWithOneNumber(n)
  implicit def indexerConstructionWithOneNumberToIndex(construction: IndexerConstructionWithOneNumber): Index =
    Indexer.indexerConstructionWithOneNumberToIndex(construction)
  implicit def indexerConstructionWithTwoNumbersToSlice(construction: IndexerConstructionWithTwoNumbers): Slice =
    Indexer.indexerConstructionWithTwoNumbersToSlice(construction)
  implicit def indexerConstructionWithThreeNumbersToSlice(construction: IndexerConstructionWithThreeNumbers): Slice =
    Indexer.indexerConstructionWithThreeNumbersToSlice(construction)

  //endregion Indexer Implicits
}
