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

package org.platanios.tensorflow.api.ops.metrics

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.{Math, Op, Output}
import org.platanios.tensorflow.api.ops.metrics.Metric.{METRIC_RESETS, METRIC_UPDATES, METRIC_VALUES, METRIC_VARIABLES}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.FLOAT32

/** Accuracy metric.
  *
  * The accuracy metric calculates how often a set of predictions matches a corresponding set of targets. The metric
  * creates two local variables, `total` and `count` that are used to compute the frequency with which the predictions
  * match the targets. This frequency is ultimately returned as an idempotent operation that simply divides `total` by
  * `count`.
  *
  * For estimation of the metric over a stream of data, the function creates an `update` operation that updates these
  * variables and returns the accuracy. Internally, an `isCorrect` operation computes a tensor with elements equal to 1
  * where the corresponding elements of the predictions and the targets match, and 0 otherwise. `update` increments
  * `total` with the reduced sum of the product of `isCorrect` and `weights`, and increments `count` with the reduced
  * sum of `weights`.
  *
  * If `weights` is `None`, the weights default to 1. Use weights of `0` to mask values.
  *
  * @param  namescope            Name prefix for the created ops.
  * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  *
  * @author Emmanouil Antonios Platanios
  */
class Accuracy(
    val namescope: String,
    protected val defaultWeights: Option[Tensor[FLOAT32]] = None,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
) extends Metric[(Output, Output), Output] {
  private[this] val meanMetric = {
    Mean(name, defaultWeights, variablesCollections, valuesCollections, updatesCollections, resetsCollections)
  }

  /** Name of this metric. */
  override def name: String = namescope

  /** Weights to multiply the provided values with when computing the value of this metric. */
  override def weights: Option[Tensor[FLOAT32]] = defaultWeights

  /** Computes the value of this metric for the provided predictions and targets, optionally weighted by `weights`.
    *
    * @param  values  Tuple containing the predictions tensor and the targets tensor.
    * @param  weights Optional tensor containing weights for the values.
    * @param  name    Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  override def compute(
      values: (Output, Output),
      weights: Option[Output] = None,
      name: String = s"$name/Compute"
  ): Output = {
    var (matchedPredictions, matchedTargets, matchedWeights) = Metric.matchAxes(values._1, Some(values._2), weights)
    matchedPredictions.shape.assertIsCompatibleWith(matchedTargets.get.shape)
    matchedPredictions = matchedPredictions.cast(matchedTargets.get.dataType)
    val isCorrect = Math.equal(matchedPredictions, matchedTargets.get)
    meanMetric.compute(isCorrect, matchedWeights, name)
  }

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  values  Tuple containing the predictions tensor and the targets tensor.
    * @param  weights Optional tensor containing weights for the predictions.
    * @param  name    Name prefix for the created ops.
    * @return Tuple containing: (i) an output representing the current value of the metric, (ii) an op used to update
    *         its current value and obtain the new value, and (iii) an op used to reset its value.
    */
  override def streaming(
      values: (Output, Output),
      weights: Option[Output] = None,
      name: String = s"$name/Streaming"
  ): Metric.StreamingInstance[Output] = {
    var (matchedPredictions, matchedTargets, matchedWeights) = Metric.matchAxes(values._1, Some(values._2), weights)
    matchedPredictions.shape.assertIsCompatibleWith(matchedTargets.get.shape)
    matchedPredictions = matchedPredictions.cast(matchedTargets.get.dataType)
    val isCorrect = Math.equal(matchedPredictions, matchedTargets.get)
    meanMetric.streaming(isCorrect, matchedWeights, name)
  }
}

object Accuracy {
  /** Creates a new accuracy metric.
    *
    * @param  namescope            Name prefix for the created ops.
    * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
    * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
    * @param  valuesCollections    Graph collections in which to add the metric values.
    * @param  updatesCollections   Graph collections in which to add the metric updates.
    * @param  resetsCollections    Graph collections in which to add the metric resets.
    * @return New accuracy metric.
    */
  def apply(
      namescope: String,
      defaultWeights: Option[Tensor[FLOAT32]] = None,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
  ): Accuracy = {
    new Accuracy(
      namescope, defaultWeights, variablesCollections, valuesCollections, updatesCollections, resetsCollections)
  }
}
