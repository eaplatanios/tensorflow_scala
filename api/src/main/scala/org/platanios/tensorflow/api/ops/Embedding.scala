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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.Indexer._
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.types.INT32

import scala.language.postfixOps

/** Contains functions for constructing ops related to embeddings.
  *
  * @author Emmanouil Antonios Platanios
  */
private[ops] trait Embedding {
  /** $OpDocEmbeddingEmbeddingLookup
    *
    * @group EmbeddingOps
    * @param  parameters        Embedding map, which is either a single tensor, a list of `P` tensors with the same
    *                           shape, except for their first dimension, representing sharded embedding tensors, or a
    *                           `PartitionedVariable`, created by partitioning along the first dimension.
    * @param  ids               `INT32` or `INT64` tensor to be looked up in `parameters`.
    * @param  partitionStrategy Partitioning strategy to use if `parameters.numPartitions > 1`.
    * @param  transformFn       If provided, this function is applied to each partitioned tensor of retrieved
    *                           embeddings, colocated with the embeddings. The shape of the argument to this function
    *                           will be the same as that of `parameters`, except for the size of the first dimension.
    *                           The first dimension of the result's shape must have the same size as that of the
    *                           argument's. Note that, if `maxNorm` is provided, then norm-based clipping is performed
    *                           before the `transformFn` is applied.
    * @param  maxNorm           If provided, embedding values are l2-normalized to this value.
    * @param  name              Name prefix used for the created op.
    * @return Obtained embeddings for the provided `ids`.
    */
  def embeddingLookup(
      parameters: EmbeddingMap,
      ids: Output,
      partitionStrategy: PartitionStrategy = ModStrategy,
      transformFn: Output => Output = null,
      maxNorm: Output = null,
      name: String = "EmbeddingLookup"
  ): Output = {
    Op.createWithNameScope(name) {
      if (parameters.numPartitions == 1 && (ids.rank == 1 || transformFn == null)) {
        Op.colocateWith(Set(parameters.partitionParameters(0).colocationOp)) {
          var result = parameters.partitionParameters(0).gather(ids)
          if (maxNorm != null)
            result = Embedding.clipByNorm(result, ids, maxNorm)
          if (transformFn != null)
            result = transformFn(result)
          result
        }
      } else {
        // Flatten the ids. There are two cases where we need to do this:
        //   - There is more than one parameter tensors.
        //   - There is a `transformFn` and ids is not statically known to be 1-D.
        // In this case, we must flatten because `transformFn` expects a flat tensor of embeddings.
        val flattenedIds = ids.reshape(Shape(-1))
        val originalIds = Math.range(0, flattenedIds.size)
        // Create `partitionAssignments` and set `newIds` depending on the strategy.
        val transformedIds = partitionStrategy.transformIds(
          flattenedIds, parameters.partitionParameters, parameters.numPartitions)
        var partitionAssignments = transformedIds._1
        val newIds = transformedIds._2
        // Cast partition assignments to `INT32` for use in `dynamicPartition`.
        // There really should not be more than 2^32 partitions.
        partitionAssignments = partitionAssignments.cast(INT32)

        // Partition list of ids based on assignments into `parameters.numPartitions` separate lists.
        val gatherIds = DataFlow.dynamicPartition(newIds, partitionAssignments, parameters.numPartitions)
        // Similarly, partition the original indices.
        val partitionIndices = DataFlow.dynamicPartition(originalIds, partitionAssignments, parameters.numPartitions)

        // Do `parameters.numPartitions` separate lookups, finding embeddings for `plist(p)` in `parameters(p)`.
        val partitionedResult = parameters.partitionParameters.zip(gatherIds).map {
          case (params, paramIds) => Op.colocateWith(Set(params.colocationOp)) {
            var result = params.gather(paramIds)
            // If `transformFn` is provided, the `clipByNorm` precedes the transform and hence must be co-located.
            // See below for the counterpart if `transformFn` is not provided.
            if (maxNorm != null)
              result = Embedding.clipByNorm(result, paramIds, maxNorm)
            if (transformFn != null && maxNorm != null)
              result = transformFn(Embedding.clipByNorm(result, paramIds, maxNorm))
            else if (transformFn != null)
              result = transformFn(result)
            result
          }
        }

        // Stitch these back together.
        var result = DataFlow.dynamicStitch(partitionIndices, partitionedResult)

        // Determine the static element shape.
        val elementStaticShape = {
          if (transformFn == null) {
            var shape = parameters.partitionParameters(0).staticShape(1 ::)
            parameters.partitionParameters.tail.foreach(p => shape = shape.mergeWith(p.staticShape(1 ::)))
            shape
          } else {
            result.shape(1 ::)
          }
        }

        // Compute the dynamic element shape.
        val elementDynamicShape = {
          if (elementStaticShape.isFullyDefined) {
            elementStaticShape.toOutput()
          } else if (transformFn == null) {
            // It's important that we compute the shape on the right device to avoid data copies.
            Op.colocateWith(Set(parameters.partitionParameters(0).colocationOp)) {
              parameters.partitionParameters(0).dynamicShape(1 ::)
            }
          } else {
            Basic.shape(result)(1 ::)
          }
        }

        // Reshape to reverse the flattening of the ids.
        result = result.reshape(Basic.concatenate(Seq(Basic.shape(ids), elementDynamicShape)))

        // Normally the reshape is sufficient, but setting shape explicitly teaches shape inference that
        // `parameters.partitionParameters(1 ::).shape` matters (in the case that `transformFn` is `null`).
        result.setShape(ids.shape.concatenateWith(elementStaticShape))
        if (transformFn == null) {
          // If `transformFn` was provided, the `clipByNorm` was done above.
          result = Embedding.clipByNorm(result, ids, maxNorm)
        }

        result
      }
    }
  }

  /** $OpDocEmbeddingSparseEmbeddingLookup
    *
    * @group EmbeddingOps
    * @param  parameters        Embedding map, which is either a single tensor, a list of `P` tensors with the same
    *                           shape, except for their first dimension, representing sharded embedding tensors, or a
    *                           `PartitionedVariable`, created by partitioning along the first dimension.
    * @param  sparseIds         `NxM` sparse tensor containing `INT64` ids, where `N` typically corresponds to the batch
    *                           size and `M` is arbitrary.
    * @param  sparseWeights     Either a sparse tensor containing `FLOAT32` or `FLOAT64` weight values, or None `null`
    *                           to indicate all weights should be taken to be equal to 1. If specified, `sparseWeights`
    *                           must have exactly the same shape and indices as `sparseIds`.
    * @param  partitionStrategy Partitioning strategy to use if `parameters.numPartitions > 1`.
    * @param  combiner          Combination/reduction strategy to use for the obtained embeddings.
    * @param  maxNorm           If provided, embedding values are l2-normalized to this value.
    * @param  name              Name prefix used for the created op.
    * @return Obtained embeddings for the provided `ids`.
    */
  def sparseEmbeddingLookup(
      parameters: EmbeddingMap,
      sparseIds: SparseOutput,
      sparseWeights: SparseOutput = null,
      partitionStrategy: PartitionStrategy = ModStrategy,
      combiner: Combiner = SumSqrtNCombiner,
      maxNorm: Output = null,
      name: String = "SparseEmbeddingLookup"
  ): Output = {
    val ignoreWeights = sparseWeights == null
    if (!ignoreWeights) {
      sparseIds.indices.shape.assertIsCompatibleWith(sparseWeights.indices.shape)
      sparseIds.values.shape.assertIsCompatibleWith(sparseWeights.values.shape)
      if (sparseIds.denseShape != null && sparseWeights.denseShape != null)
        sparseIds.denseShape.shape.assertIsCompatibleWith(sparseWeights.denseShape.shape)
      else if (sparseIds.denseShape != sparseWeights.denseShape)
        throw InvalidShapeException("The dense shapes of 'sparseIds' and 'sparseWeights' must be compatible.")
    }
    Op.createWithNameScope(name) {
      val segmentIds = sparseIds.indices(::, 0).cast(INT32)
      val (ids, idx) = Basic.unique(sparseIds.values, 0)
      var embeddings = embeddingLookup(parameters, ids, partitionStrategy, maxNorm = maxNorm)
      if (ignoreWeights) {
        combiner.combine(embeddings, idx, segmentIds)
      } else {
        embeddings = embeddings.gather(idx)
        val weights = sparseWeights.values.cast(embeddings.dataType)
        // Reshape weights to allow broadcasting.
        val weightsStaticShape = weights.shape
        val weightsDynamicShape = Basic.shape(weights)
        val ones = Basic.fill(weightsDynamicShape.dataType, Basic.expandDims(Basic.rank(embeddings) - 1, 0))(1)
        val broadcastedWeightsShape = Basic.concatenate(Seq(weightsDynamicShape, ones), 0)
        val reshapedWeights = weights.reshape(broadcastedWeightsShape)
        // Set the weight shape, since after reshaping to `broadcastedWeightsShape`, the shape becomes unknown.
        if (embeddings.shape.rank != -1)
          reshapedWeights.setShape(
            weightsStaticShape.concatenateWith(Shape.fromSeq((0 until embeddings.shape.rank - 1).map(_ => 1))))
        val weightedEmbeddings = embeddings * reshapedWeights
        combiner.combineWeighted(weightedEmbeddings, reshapedWeights, segmentIds)
      }
    }
  }

  /** Partitioning strategy for the embeddings map. */
  sealed trait PartitionStrategy {
    /** Transforms the provided ids based on this partition strategy and returns the partition assignments and the
      * new/transformed ids. */
    def transformIds(ids: Output, parameters: Seq[EmbeddingParameters], numPartitions: Int): (Output, Output)
  }

  /** Each id is assigned to partition `p = id % parameters.numPartitions`. For instance, 13 ids are split across 5
    * partitions as: `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`.*/
  case object ModStrategy extends PartitionStrategy {
    override def transformIds(
        ids: Output, parameters: Seq[EmbeddingParameters], numPartitions: Int): (Output, Output) = {
      val numPartitionsOutput = numPartitions: Output
      val partitionAssignments = ids % numPartitionsOutput
      val newIds = ids.truncateDivide(numPartitionsOutput)
      (partitionAssignments, newIds)
    }
  }

  /** Ids are assigned to partitions in a contiguous manner. In this case, 13 ids are split across 5 partitions as:
    * `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`. */
  case object DivStrategy extends PartitionStrategy {
    override def transformIds(
        ids: Output, parameters: Seq[EmbeddingParameters], numPartitions: Int): (Output, Output) = {
      val numPartitionsOutput = numPartitions: Output
      // We compute `numTotalIds` as the sum of the first dimension size of `parameters`, and then we assign to
      // partitions based on a constant number of ids per partition. We optimize if we already know the full shape
      // statically.
      val numTotalIds: Output = {
        if (parameters.forall(p => p.staticShape.rank != -1 && p.staticShape(0) != -1)) {
          parameters.map(_.staticShape(0)).sum
        } else {
          val axis0Sizes = parameters.map(p => {
            if (p.staticShape.rank != -1 && p.staticShape(0) != -1)
              Basic.constant(p.staticShape(0))
            else
              Op.colocateWith(Set(p.colocationOp))(p.dynamicShape(0))
          })
          Math.sum(Basic.stack(axis0Sizes).cast(ids.dataType))
        }
      }
      val idsPerPartition = numTotalIds.truncateDivide(numPartitionsOutput)
      val extras = numTotalIds % numPartitionsOutput
      val partitionAssignments = Math.maximum(
        ids.truncateDivide(idsPerPartition + 1),
        (ids - extras).truncateDivide(idsPerPartition))
      val newIds = Math.select(
        partitionAssignments < extras,
        ids % (idsPerPartition + 1),
        (ids - extras) % idsPerPartition)
      (partitionAssignments, newIds)
    }
  }

  /** Method for combining sparse embeddings. */
  sealed trait Combiner {
    @inline def combine(parameters: Output, indices: Output, segmentIndices: Output): Output
    @inline def combineWeighted(parameters: Output, weights: Output, segmentIndices: Output): Output
  }

  /** Combines sparse embeddings by using a weighted sum. */
  case object SumCombiner extends Combiner {
    @inline override def combine(
        parameters: Output, indices: Output, segmentIndices: Output): Output = {
      Math.sparseSegmentSum(parameters, indices, segmentIndices)
    }

    @inline def combineWeighted(parameters: Output, weights: Output, segmentIndices: Output): Output = {
      Math.segmentSum(parameters, segmentIndices)
    }
  }

  /** Combines sparse embeddings by using a weighted sum divided by the total weight. */
  case object MeanCombiner extends Combiner {
    @inline override def combine(parameters: Output, indices: Output, segmentIndices: Output): Output = {
      Math.sparseSegmentMean(parameters, indices, segmentIndices)
    }

    @inline def combineWeighted(parameters: Output, weights: Output, segmentIndices: Output): Output = {
      val embeddings = Math.segmentSum(parameters, segmentIndices)
      val weightsSum = Math.segmentSum(weights, segmentIndices)
      Math.divide(embeddings, weightsSum)
    }
  }

  /** Combines sparse embeddings by using a weighted sum divided by the square root of the sum of the
      squares of the weights. */
  case object SumSqrtNCombiner extends Combiner {
    @inline override def combine(parameters: Output, indices: Output, segmentIndices: Output): Output = {
      Math.sparseSegmentSumSqrtN(parameters, indices, segmentIndices)
    }

    @inline def combineWeighted(parameters: Output, weights: Output, segmentIndices: Output): Output = {
      val embeddings = Math.segmentSum(parameters, segmentIndices)
      val weightsSquaredSum = Math.segmentSum(weights.square(), segmentIndices)
      Math.divide(embeddings, weightsSquaredSum.sqrt())
    }
  }
}

object Embedding extends Embedding {
  /** If `maxNorm` is not `null`, this method clips `parameters` to a maximum l2-norm of `maxNorm`. */
  private[Embedding] def clipByNorm(
      parameters: Output, indices: Output, maxNorm: Output = null, name: String = "ClipNorm"): Output = {
    if (maxNorm == null)
      parameters
    else if (parameters.rank != -1 && indices.rank != -1)
      parameters.clipByNorm(maxNorm, indices.rank until parameters.rank)
    else
      parameters.clipByNorm(maxNorm, Math.range(Basic.rank(indices), Basic.rank(parameters)))
  }

  case class OutputParameters(parameters: Output) extends EmbeddingParameters {
    @inline override def colocationOp: Op = parameters.op
    @inline override def staticShape: Shape = parameters.shape
    @inline override def dynamicShape: Output = Basic.shape(parameters)

    override def gather(indices: Output, name: String = "Gather"): Output = {
      Basic.gather(parameters, indices, name = name)
    }
  }

  case class VariableParameters(parameters: Variable) extends EmbeddingParameters {
    @inline override def colocationOp: Op = parameters.op
    @inline override def staticShape: Shape = parameters.shape
    @inline override def dynamicShape: Output = Basic.shape(parameters.value)

    override def gather(indices: Output, name: String = "Gather"): Output = {
      parameters.gather(indices, name = name)
    }
  }

  /** @define OpDocEmbeddingEmbeddingLookup
    *   The `embeddingLookup` op looks up `ids` in a list of embedding tensors.
    *
    *   This function is used to perform parallel lookups on the embedding map in `parameters`. It is a generalization
    *   of the `gather` op, where `parameters` is interpreted as a partitioning of a large embedding tensor.
    *   `parameters` may be a `PartitionedVariable` as returned when creating a variable with a partitioner.
    *
    *   If `parameters` consists of more than 1 partition, each element `id` of `ids` is partitioned between the
    *   elements of `parameters` according to the `partitionStrategy`. In all strategies, if the id space does not
    *   evenly divide the number of partitions, each of the first `(maxId + 1) % parameters.numPartitions` partitions
    *   will be assigned one more id.
    *
    *   If `partitionStrategy` is [[Embedding.ModStrategy]], we assign each id to partition
    *   `p = id % parameters.numPartitions`. For instance, 13 ids are split across 5 partitions as:
    *   `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`.
    *
    *   If `partitionStrategy` is [[Embedding.DivStrategy]], we assign ids to partitions in a contiguous manner. In this
    *   case, 13 ids are split across 5 partitions as:
    *   `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`.
    *
    *   The results of the lookup are concatenated into a dense tensor. The returned tensor has shape
    *   `ids.shape + parameters.partitionParameters(0).shape(1 ::)`.
    *
    * @define OpDocEmbeddingSparseEmbeddingLookup
    *   The `sparseEmbeddingLookup` op looks up and computes embeddings for the given sparse ids and weights.
    *
    *   The op assumes that there is at least one id for each row in the dense tensor represented by `sparseIds` (i.e.,
    *   there are no rows with empty features), and that all the indices of `sparseIds` are in canonical row-major
    *   order. It also assumes that all id values lie in the range `[0, p0)`, where `p0` is the sum of the size of
    *   `parameters` along dimension 0.
    *
    *   The op returns a dense tensor representing the combined embeddings for the provided sparse ids. For each row in
    *   the dense tensor represented by `sparseIds`, the op looks up the embeddings for all ids in that row, multiplies
    *   them by the corresponding weight, and combines them using the provided `combiner`.
    *
    *   In other words, if `shape(combinedParameters) = [p0, p1, ..., pm]` and
    *   `shape(sparseIds) = shape(sparseWeights) = [d0, d1, ..., dn]`, then
    *   `shape(output) = [d0, d1, ..., dn-1, p1, ..., pm]`.
    *
    *   For instance, if `parameters` is a `10x20` matrix, and `sparseIds` and `sparseWeights` are as follows:
    *
    *     - [0, 0]: id 1, weight 2.0
    *     - [0, 1]: id 3, weight 0.5
    *     - [1, 0]: id 0, weight 1.0
    *     - [2, 3]: id 1, weight 3.0
    *
    *   and we are using the `MeanCombiner`, then the output will be a `3x20` matrix, where:
    *
    *     - output(0, ::) = (parameters(1, ::) * 2.0 + parameters(3, ::) * 0.5) / (2.0 + 0.5)
    *     - output(1, ::) = parameters(0, ::) * 1.0
    *     - output(2, ::) = parameters(1, ::) * 3.0
    */
  private[ops] trait Documentation
}

case class EmbeddingMap(partitionParameters: Seq[EmbeddingParameters]) {
  val numPartitions: Int = partitionParameters.size
}

/** Trait for specifying supported embedding parameter types. */
trait EmbeddingParameters {
  /** Returns the op that generates these parameters (to be used for colocating other ops with it). */
  @inline def colocationOp: Op

  /** Returns the static shape of this parameters tensor. */
  @inline def staticShape: Shape

  /** Returns the dynamic shape of this parameters tensor. */
  @inline def dynamicShape: Output

  /** Gathers the embeddings corresponding to `indices` from `parameters`. */
  def gather(indices: Output, name: String = "Gather"): Output
}
