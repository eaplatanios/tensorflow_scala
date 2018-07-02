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
import org.platanios.tensorflow.api.types.{FLOAT64, INT64}

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

    val Metric          : metrics.Metric.type           = metrics.Metric
    val MapMetric       : metrics.MapMetric.type        = metrics.MapMetric
    val Mean            : metrics.Mean.type             = metrics.Mean
    val Accuracy        : metrics.Accuracy.type         = metrics.Accuracy
    val ConfusionMatrix : metrics.ConfusionMatrix.type  = metrics.ConfusionMatrix
    val GroupedPrecision: metrics.GroupedPrecision.type = metrics.GroupedPrecision
  }

  // TODO: [SPARSE] Add versions for the following utilities.

  /** Filters all but `selectedID` out of `ids`.
    *
    * @param  ids        `INT64` tensor containing the IDs to filter.
    * @param  selectedID `INT32` scalar containing the ID to select.
    * @return Sparse tensor with the same shape as `ids`, but containing only the entries equal to `selectedID`.
    */
  private[metrics] def selectID(ids: Output, selectedID: Output): SparseOutput = {
    // The shape of filled IDs is the same as `ids` with the last axis size collapsed to 1.
    val idsShape = Basic.shape(ids, dataType = INT64)
    val idsLastAxis = Basic.size(idsShape) - 1
    val filledSelectedIDShape = Math.reducedShape(idsShape, Basic.reshape(idsLastAxis, Shape(1)))

    // Intersect `ids` with the selected ID.
    val filledSelectedID = Basic.fill(INT64, filledSelectedIDShape)(selectedID.toInt64)
    val result = Sets.setIntersection(filledSelectedID, ids)
    SparseOutput(result.indices, result.values, idsShape)
  }

  /** Calculates true positives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary true positives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        `INT64` tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher `INT64` tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of true positives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return `FLOAT64` tensor containing the number of true positives.
    */
  private[metrics] def sparseTruePositivesAtK(
      labels: Output,
      predictionIDs: Output,
      labelID: Option[Output] = None,
      weights: Option[Output] = None,
      name: String = "SparseTruePositivesAtK"
  ): Output = {
    Op.createWithNameScope(name) {
      val numTruePositives = labelID match {
        case None =>
          Sets.setSize(Sets.setIntersection(predictionIDs, labels)).toFloat64
        case Some(selectedID) =>
          val filteredPredictionIDs = selectID(predictionIDs, selectedID)
          val filteredLabels = selectID(labels, selectedID)
          Sets.setSize(Sets.setIntersection(filteredPredictionIDs, filteredLabels)).toFloat64
      }
      weights match {
        case None => numTruePositives
        case Some(w) =>
          Op.createWith(controlDependencies = Set(Metric.weightsAssertBroadcastable(numTruePositives, w))) {
            Math.multiply(numTruePositives, w.toFloat64)
          }
      }
    }
  }

  /** Calculates streaming true positives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary true positives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        `INT64` tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher `INT64` tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of true positives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Streaming metric instance for computing a `FLOAT64` tensor containing the number of true positives.
    */
  private[metrics] def streamingSparseTruePositivesAtK(
      labels: Output,
      predictionIDs: Output,
      labelID: Option[Output] = None,
      weights: Option[Output] = None,
      name: String = "StreamingSparseTruePositivesAtK"
  ): Metric.StreamingInstance[Output] = {
    Op.createWithNameScope(name) {
      val numTruePositives = sparseTruePositivesAtK(labels, predictionIDs, labelID, weights)
      val batchNumTruePositives = Math.sum(numTruePositives)
      val accumulator = Metric.variable("Accumulator", FLOAT64, Shape())
      val value = accumulator.value
      val update = accumulator.assignAdd(batchNumTruePositives)
      val reset = accumulator.initializer
      Metric.StreamingInstance(value, update, reset, Set(accumulator))
    }
  }

  /** Calculates false negatives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary false negatives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        `INT64` tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher `INT64` tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of false negatives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return `FLOAT64` tensor containing the number of false negatives.
    */
  private[metrics] def sparseFalseNegativesAtK(
      labels: Output,
      predictionIDs: Output,
      labelID: Option[Output] = None,
      weights: Option[Output] = None,
      name: String = "SparseFalseNegativesAtK"
  ): Output = {
    Op.createWithNameScope(name) {
      val numTruePositives = labelID match {
        case None =>
          Sets.setSize(Sets.setDifference(predictionIDs, labels, aMinusB = false)).toFloat64
        case Some(selectedID) =>
          val filteredPredictionIDs = selectID(predictionIDs, selectedID)
          val filteredLabels = selectID(labels, selectedID)
          Sets.setSize(Sets.setIntersection(filteredPredictionIDs, filteredLabels)).toFloat64
      }
      weights match {
        case None => numTruePositives
        case Some(w) =>
          Op.createWith(controlDependencies = Set(Metric.weightsAssertBroadcastable(numTruePositives, w))) {
            Math.multiply(numTruePositives, w.toFloat64)
          }
      }
    }
  }

  /** Calculates streaming false negatives for the recall@k and the precision@k metrics.
    *
    * If `labelID` is specified, the constructed op calculates binary false negatives for `labelID` only.
    * If `labelID` is not specified, then it calculates metrics for `k` predicted vs `n` labels.
    *
    * @param  labels        `INT64` tensor with shape `[D1, ... DN, numLabels]`, where `N >= 1` and `numLabels` is the
    *                       number of target classes for the associated prediction. Commonly, `N = 1` and `labels` has
    *                       shape `[batchSize, numLabels]`. `[D1, ..., DN]` must match the shape of `predictionIDs`.
    * @param  predictionIDs 1-D or higher `INT64` tensor with its last dimension corresponding to the top `k` predicted
    *                       classes. For rank `n`, the first `n-1` dimensions must match the shape of `labels`.
    * @param  labelID       Optional label for which we want to compute the number of false negatives.
    * @param  weights       Optional weights tensor with rank is either `0`, or `n-1`, where `n` is the rank of
    *                       `labels`. If the latter, it must be broadcastable to `labels` (i.e., all dimensions must be
    *                       either `1`, or the same as the corresponding `labels` dimension).
    * @param  name          Namescope to use for all created ops.
    * @return Streaming metric instance for computing a `FLOAT64` tensor containing the number of false negatives.
    */
  private[metrics] def streamingSparseFalseNegativesAtK(
      labels: Output,
      predictionIDs: Output,
      labelID: Option[Output] = None,
      weights: Option[Output] = None,
      name: String = "StreamingSparseFalseNegativesAtK"
  ): Metric.StreamingInstance[Output] = {
    Op.createWithNameScope(name) {
      val numFalseNegatives = sparseFalseNegativesAtK(labels, predictionIDs, labelID, weights)
      val batchNumFalseNegatives = Math.sum(numFalseNegatives)
      val accumulator = Metric.variable("Accumulator", FLOAT64, Shape())
      val value = accumulator.value
      val update = accumulator.assignAdd(batchNumFalseNegatives)
      val reset = accumulator.initializer
      Metric.StreamingInstance(value, update, reset, Set(accumulator))
    }
  }
}
