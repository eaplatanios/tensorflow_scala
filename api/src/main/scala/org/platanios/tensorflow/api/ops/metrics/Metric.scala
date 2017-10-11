/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.ops.metrics

import org.platanios.tensorflow.api.core.Graph.Keys.{OutputCollectionKey, VariableCollectionKey}
import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.{Initializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.types.{DataType, FLOAT32, FLOAT64}

/** Trait representing evaluation metrics that support both eager computation, as well as computation in a streaming
  * manner.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Metric {
  /** Name of this metric. */
  val name: String

  /** Computes the value of this metric for the provided predictions and targets, optionally weighted by `weights`.
    *
    * @param  predictions Tensor containing the predictions.
    * @param  targets     Tensor containing the desired targets.
    * @param  weights     Tensor containing weights for the predictions.
    * @param  name        Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  def compute(predictions: Output, targets: Output, weights: Output = null, name: String = name): Output

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  predictions Tensor containing the predictions.
    * @param  targets     Tensor containing the desired targets.
    * @param  weights     Tensor containing weights for the predictions.
    * @param  name        Name prefix for the created ops.
    * @return Tuple containing: (i) output representing the current value of the metric, (ii) op used to reset its
    *         value, and (iii) op used to update its current value and obtain the new value.
    */
  def streaming(
      predictions: Output, targets: Output, weights: Output = null, name: String = name): (Output, Op, Output)
}

object Metric {
  /** Key to collect the subset of `Variable` objects that are used for computing and storing metric values. */
  object METRIC_VARIABLES extends VariableCollectionKey {
    override def name: String = "metric_variables"
  }

  /** Key to collect the subset of tensors that are used for updating metric values. */
  object METRIC_UPDATES extends OutputCollectionKey {
    override def name: String = "metric_updates"
  }

  /** Creates a new variable and adds it to the `LOCAL_VARIABLES` graph collection. */
  private[metrics] def localVariable(
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = ZerosInitializer,
      collections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES)): Variable = {
    Variable.getVariable(name = name, trainable = false, collections = collections + Graph.Keys.LOCAL_VARIABLES)
  }

  /** Divides two values, returning 0 if the denominator is <= 0. */
  private[metrics] def safeDiv(numerator: Output, denominator: Output, name: String = "SafeDiv"): Output = {
    Math.select(Math.greater(denominator, 0), Math.divide(numerator, denominator), 0, name)
  }

  /** Divides two scalar values, returning 0 if the denominator is 0. */
  private[metrics] def safeScalarDiv(numerator: Output, denominator: Output, name: String = "SafeScalarDiv"): Output = {
    val zeros = Basic.zerosLike(denominator)
    Math.select(Math.equal(denominator, zeros), zeros, Math.divide(numerator, denominator), name)
  }

  /** The `matchAxes` op matches the dimension sizes of `predictions`, `targets`, and `weights`.
    *
    * Squeezes either `predictions` or `targets`, such that the resulting ranks differ by `expectedRankDiff`.
    * Squeezes `weights`, or expands its axes, such that the resulting `predictions` and `weights` ranks differ by
    * `expectedRankDiff`.
    *
    * In the common case where we expect shapes to match, `expectedRankDiff` defaults to 0, and we squeeze the last
    * axis of the larger rank if the ranks of `predictions` and `targets` differ by 1. However, if for example,
    * `targets` contains class IDs and `predictions` contains 1 probability per class, then we expect `predictions` to
    * have 1 more axis than `targets`, and so `expectedRankDiff` would be 1. In this case, we would squeeze `targets`
    * if `rank(predictions) - rank(targets) == 0`, and `predictions` if `rank(predictions) - rank(targets) == 2`.
    *
    * This method will use static shape information, if available. Otherwise, it will add graph ops, which could
    * result in a performance penalty.
    *
    * @param  predictions      Predictions tensor.
    * @param  targets          Targets tensor.
    * @param  weights          Weights tensor.
    * @param  expectedRankDiff Expected rank difference.
    * @return Tuple containing the processed `predictions`, `targets`, and `weights`.
    */
  private[metrics] def matchAxes(
      predictions: Output, targets: Output = null, weights: Output = null, expectedRankDiff: Int = 0
  ): (Output, Output, Output) = {
    var matchedPredictions = predictions
    var matchedTargets = targets
    if (targets != null) {
      if (predictions.rank != -1 && targets.rank != -1) {
        // Use static rank.
        val rankDiff = predictions.rank - targets.rank
        if (rankDiff == expectedRankDiff + 1)
          matchedPredictions = Basic.squeeze(matchedPredictions, Seq(-1))
        else if (rankDiff == expectedRankDiff - 1)
          matchedTargets = Basic.squeeze(matchedTargets, Seq(-1))
      } else {
        // Use dynamic rank.
        val rankDiff = Basic.rank(predictions) - Basic.rank(targets)
        if (predictions.rank == -1 || predictions.shape(-1) == -1 || predictions.shape(-1) == 1) {
          matchedPredictions = ControlFlow.cond(
            Math.equal(rankDiff, expectedRankDiff + 1),
            () => Basic.squeeze(matchedPredictions, Seq(-1)),
            () => matchedPredictions)
        }
        if (targets.rank == -1 || targets.shape(-1) == -1 || targets.shape(-1) == 1) {
          matchedTargets = ControlFlow.cond(
            Math.equal(rankDiff, expectedRankDiff - 1),
            () => Basic.squeeze(matchedTargets, Seq(-1)),
            () => matchedTargets)
        }
      }
    }
    if (weights == null || weights.rank == 0) {
      (matchedPredictions, matchedTargets, weights)
    } else {
      var matchedWeights = weights
      if (predictions.rank != -1 && weights.rank != -1) {
        // Use static rank.
        val rankDiff = predictions.rank - weights.rank
        if (rankDiff == expectedRankDiff + 1)
          matchedWeights = Basic.expandDims(matchedWeights, -1)
        else if (rankDiff == expectedRankDiff - 1)
          matchedWeights = Basic.squeeze(matchedWeights, Seq(-1))
      } else {
        // Use dynamic rank.
        val rankDiff = Basic.rank(predictions) - Basic.rank(weights)
        // If weights are scalar, do nothing. Otherwise, try to add or remove an axis to match predictions.
        matchedWeights = ControlFlow.cond(
          Math.equal(matchedWeights, 0),
          () => matchedWeights,
          () => ControlFlow.cond(
            Math.equal(rankDiff, -1),
            if (weights.rank == -1 || weights.shape(-1) == -1 || weights.shape(-1) == 1)
              () => Basic.expandDims(matchedWeights, -1)
            else
              () => matchedWeights,
            ControlFlow.cond(
              Math.equal(rankDiff, 1),
              () => Basic.squeeze(matchedWeights, Seq(-1)),
              () => matchedWeights)))
      }
      (matchedPredictions, matchedTargets, matchedWeights)
    }
  }
}
