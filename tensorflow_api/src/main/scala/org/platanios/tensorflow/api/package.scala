package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api {
  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = Tensor.RowMajorOrder

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

  // TODO: Need to be able to reset and close this session.
  private[api] val defaultSession: Session = Session(defaultGraph)
}
