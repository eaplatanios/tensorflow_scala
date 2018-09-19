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
import org.platanios.tensorflow.api.ops.{NN, Op, Output}
import org.platanios.tensorflow.api.ops.metrics.Metric.{METRIC_RESETS, METRIC_UPDATES, METRIC_VALUES, METRIC_VARIABLES}
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

/** Precision@K metric.
  *
  * This metric computes the precision@k of predictions with respect to sparse labels. If `labelID` is specified, we
  * calculate the precision by considering only the entries in the batch for which `labelID` is in the top-k highest
  * `predictions`, and computing the fraction of them for which `labelID` is indeed a correct label. If `labelID` is
  * not specified, we calculate precision as how often, on average, a class among the top-k classes with the highest
  * predicted values of a batch entry is correct and can be found in the label for that entry.
  *
  * The metric creates two local variables, `truePositivesAtK` and `falsePositivesAtK` that are used to compute the
  * precision of some provided predictions and targets. The predictions must be in the form of predicted label indices.
  * The precision is ultimately returned as an idempotent operation that simply divides `truePositivesAtK` by
  * `truePositivesAtK + falsePositivesAtK`.
  *
  * For estimation of the metric over a stream of data, the function creates an `update` operation that updates these
  * variables and returns the precision.
  *
  * If `weights` is `None`, the weights default to 1. Use weights of `0` to mask values.
  *
  * @param  nameScope            Name prefix for the created ops.
  * @param  k                    Value for k.
  * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
  * @param  labelID              Optional label for which we want to compute the precision.
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  *
  * @author Emmanouil Antonios Platanios
  */
class PrecisionAtK(
    val nameScope: String,
    val k: Int,
    protected val defaultWeights: Option[Tensor[Float]] = None,
    val labelID: Option[Int] = None,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
) extends Metric[(Output, Output), Output] {
  private[this] val groupedPrecisionMetric = {
    GroupedPrecision(
      s"$name/GroupedPrecisionAt$k", defaultWeights, labelID, variablesCollections, valuesCollections,
      updatesCollections, resetsCollections)
  }

  /** Name of this metric. */
  override def name: String = nameScope

  /** Weights to multiply the provided values with when computing the value of this metric. */
  override def weights: Option[Tensor[Float]] = defaultWeights

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
    val (predictions, targets) = values
    var ops = Set(predictions.op, targets.op)
    val computedWeights = getWeights(weights)
    computedWeights.foreach(ops += _.op)
    Op.createWithNameScope(name, ops) {
      val (_, topKIndices) = NN.topK(predictions, k)
      groupedPrecisionMetric.compute((topKIndices, targets), computedWeights, name)
    }
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
    val (predictions, targets) = values
    var ops = Set(predictions.op, targets.op)
    val computedWeights = getWeights(weights)
    computedWeights.foreach(ops += _.op)
    Op.createWithNameScope(name, ops) {
      val (_, topKIndices) = NN.topK(predictions, k)
      groupedPrecisionMetric.streaming((topKIndices, targets), computedWeights, name)
    }
  }
}

object PrecisionAtK {
  /** Creates a new precision@K metric.
    *
    * @param  nameScope            Name prefix for the created ops.
    * @param  k                    Value for k.
    * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
    * @param  labelID              Optional label for which we want to compute the precision.
    * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
    * @param  valuesCollections    Graph collections in which to add the metric values.
    * @param  updatesCollections   Graph collections in which to add the metric updates.
    * @param  resetsCollections    Graph collections in which to add the metric resets.
    * @return New mean metric.
    */
  def apply(
      nameScope: String,
      k: Int,
      defaultWeights: Option[Tensor[Float]] = None,
      labelID: Option[Int] = None,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
  ): PrecisionAtK = {
    new PrecisionAtK(
      nameScope, k, defaultWeights, labelID, variablesCollections, valuesCollections, updatesCollections,
      resetsCollections)
  }
}
