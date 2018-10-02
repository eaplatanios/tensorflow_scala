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
import org.platanios.tensorflow.api.types.{FLOAT32, INT64}

/**
  * @author Emmanouil Antonios Platanios
  */
package object metrics {
  private[api] trait API {
    type Metric[T, R] = metrics.Metric[T, R]
    type MapMetric[S, T, R] = metrics.MapMetric[S, T, R]
    type Mean = metrics.Mean
    type Accuracy = metrics.Accuracy
    type ConfusionMatrix = metrics.ConfusionMatrix
    type GroupedPrecision = metrics.GroupedPrecision
    type PrecisionAtK = metrics.PrecisionAtK

    val Metric          : metrics.Metric.type           = metrics.Metric
    val MapMetric       : metrics.MapMetric.type        = metrics.MapMetric
    val Mean            : metrics.Mean.type             = metrics.Mean
    val Accuracy        : metrics.Accuracy.type         = metrics.Accuracy
    val ConfusionMatrix : metrics.ConfusionMatrix.type  = metrics.ConfusionMatrix
    val GroupedPrecision: metrics.GroupedPrecision.type = metrics.GroupedPrecision
    val PrecisionAtK    : metrics.PrecisionAtK.type     = metrics.PrecisionAtK
  }

  // TODO: [SPARSE] Add versions for the following utilities.

  /** Filters all but `selectedID` out of `ids`.
    *
    * @param  ids        Tensor containing the IDs to filter.
    * @param  selectedID Scalar containing the ID to select.
    * @return Sparse tensor with the same shape as `ids`, but containing only the entries equal to `selectedID`.
    */
  private[metrics] def selectID(
      ids: Output[Long],
      selectedID: Output[Long]
  ): SparseOutput[Long] = {
    // The shape of filled IDs is the same as `ids` with the last axis size collapsed to 1.
    val idsShape = Basic.shape(ids, INT64)
    val idsLastAxis = Basic.size(idsShape, INT64) - 1L
    val filledSelectedIDShape = Math.reducedShape(idsShape, Basic.reshape(idsLastAxis, Shape(1)))

    // Intersect `ids` with the selected ID.
    val filledSelectedID = Basic.fill(INT64, filledSelectedIDShape)(selectedID)
    val result = Sets.setIntersection(filledSelectedID, ids)
    SparseOutput(result.indices, result.values, idsShape)
  }

  /** Calculates true positives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary true positives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        Tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of true positives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Tensor containing the number of true positives.
    */
  private[metrics] def sparseTruePositives(
      labels: Output[Long],
      predictionIDs: Output[Long],
      labelID: Option[Output[Long]] = None,
      weights: Option[Output[Float]] = None,
      name: String = "SparseTruePositives"
  ): Output[Float] = {
    Op.nameScope(name) {
      val numTruePositives = labelID match {
        case None =>
          Sets.setSize(Sets.setIntersection(predictionIDs, labels)).toFloat32
        case Some(selectedID) =>
          val filteredPredictionIDs = selectID(predictionIDs, selectedID)
          val filteredLabels = selectID(labels, selectedID)
          Sets.setSize(Sets.setIntersection(filteredPredictionIDs, filteredLabels)).toFloat32
      }
      weights match {
        case None => numTruePositives
        case Some(w) =>
          Op.createWith(controlDependencies = Set(Metric.weightsAssertBroadcastable(numTruePositives, w))) {
            Math.multiply(numTruePositives, w)
          }
      }
    }
  }

  /** Calculates streaming true positives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary true positives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        Tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of true positives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Streaming metric instance for computing a tensor containing the number of true positives.
    */
  private[metrics] def streamingSparseTruePositives(
      labels: Output[Long],
      predictionIDs: Output[Long],
      labelID: Option[Output[Long]] = None,
      weights: Option[Output[Float]] = None,
      name: String = "StreamingSparseTruePositives"
  ): Metric.StreamingInstance[Output[Float]] = {
    Op.nameScope(name) {
      val numTruePositives = sparseTruePositives(labels, predictionIDs, labelID, weights)
      val batchNumTruePositives = Math.sum(numTruePositives)
      val accumulator = Metric.variable(s"$name/Accumulator", FLOAT32, Shape())
      val value = accumulator.value
      val update = accumulator.assignAdd(batchNumTruePositives)
      val reset = accumulator.initializer
      Metric.StreamingInstance(value, update, reset, Set(accumulator))
    }
  }

  /** Calculates false positives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary false positives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        Tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of false positives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Tensor containing the number of false positives.
    */
  private[metrics] def sparseFalsePositives(
      labels: Output[Long],
      predictionIDs: Output[Long],
      labelID: Option[Output[Long]] = None,
      weights: Option[Output[Float]] = None,
      name: String = "SparseFalsePositives"
  ): Output[Float] = {
    Op.nameScope(name) {
      val numFalsePositives = labelID match {
        case None =>
          Sets.setSize(Sets.setDifference(predictionIDs, labels, aMinusB = true)).toFloat32
        case Some(selectedID) =>
          val filteredPredictionIDs = selectID(predictionIDs, selectedID)
          val filteredLabels = selectID(labels, selectedID)
          Sets.setSize(Sets.setDifference(filteredPredictionIDs, filteredLabels, aMinusB = true)).toFloat32
      }
      weights match {
        case None => numFalsePositives
        case Some(w) =>
          Op.createWith(controlDependencies = Set(Metric.weightsAssertBroadcastable(numFalsePositives, w))) {
            Math.multiply(numFalsePositives, w)
          }
      }
    }
  }

  /** Calculates streaming false positives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary false positives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        Tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of false positives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Streaming metric instance for computing a tensor containing the number of false positives.
    */
  private[metrics] def streamingSparseFalsePositives(
      labels: Output[Long],
      predictionIDs: Output[Long],
      labelID: Option[Output[Long]] = None,
      weights: Option[Output[Float]] = None,
      name: String = "StreamingSparseFalsePositives"
  ): Metric.StreamingInstance[Output[Float]] = {
    Op.nameScope(name) {
      val numFalsePositives = sparseFalsePositives(labels, predictionIDs, labelID, weights)
      val batchNumFalsePositives = Math.sum(numFalsePositives)
      val accumulator = Metric.variable(s"$name/Accumulator", FLOAT32, Shape())
      val value = accumulator.value
      val update = accumulator.assignAdd(batchNumFalsePositives)
      val reset = accumulator.initializer
      Metric.StreamingInstance(value, update, reset, Set(accumulator))
    }
  }

  /** Calculates false negatives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary false negatives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        Tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of false negatives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Tensor containing the number of false negatives.
    */
  private[metrics] def sparseFalseNegatives(
      labels: Output[Long],
      predictionIDs: Output[Long],
      labelID: Option[Output[Long]] = None,
      weights: Option[Output[Float]] = None,
      name: String = "SparseFalseNegatives"
  ): Output[Float] = {
    Op.nameScope(name) {
      val numTruePositives = labelID match {
        case None =>
          Sets.setSize(Sets.setDifference(predictionIDs, labels, aMinusB = false)).toFloat32
        case Some(selectedID) =>
          val filteredPredictionIDs = selectID(predictionIDs, selectedID)
          val filteredLabels = selectID(labels, selectedID)
          Sets.setSize(Sets.setDifference(filteredPredictionIDs, filteredLabels, aMinusB = false)).toFloat32
      }
      weights match {
        case None => numTruePositives
        case Some(w) =>
          Op.createWith(controlDependencies = Set(Metric.weightsAssertBroadcastable(numTruePositives, w))) {
            Math.multiply(numTruePositives, w)
          }
      }
    }
  }

  /** Calculates streaming false negatives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary false negatives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        Tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of false negatives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Streaming metric instance for computing a tensor containing the number of false negatives.
    */
  private[metrics] def streamingSparseFalseNegatives(
      labels: Output[Long],
      predictionIDs: Output[Long],
      labelID: Option[Output[Long]] = None,
      weights: Option[Output[Float]] = None,
      name: String = "StreamingSparseFalseNegatives"
  ): Metric.StreamingInstance[Output[Float]] = {
    Op.nameScope(name) {
      val numFalseNegatives = sparseFalseNegatives(labels, predictionIDs, labelID, weights)
      val batchNumFalseNegatives = Math.sum(numFalseNegatives)
      val accumulator = Metric.variable(s"$name/Accumulator", FLOAT32, Shape())
      val value = accumulator.value
      val update = accumulator.assignAdd(batchNumFalseNegatives)
      val reset = accumulator.initializer
      Metric.StreamingInstance(value, update, reset, Set(accumulator))
    }
  }
}
