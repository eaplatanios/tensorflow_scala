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
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.{Math, Op, Output}
import org.platanios.tensorflow.api.ops.metrics.Metric._
import org.platanios.tensorflow.api.ops.variables.{Variable, VariableScope}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.FLOAT32

/** Grouped precision metric.
  *
  * The metric creates two local variables, `truePositives` and `falsePositives` that are used to compute the precision
  * of some provided predictions and targets. The predictions must be in the form of predicted label indices. The
  * precision is ultimately returned as an idempotent operation that simply divides `truePositives` by
  * `truePositives + falsePositives`.
  *
  * For estimation of the metric over a stream of data, the function creates an `update` operation that updates these
  * variables and returns the precision.
  *
  * If `weights` is `None`, the weights default to 1. Use weights of `0` to mask values.
  *
  * @param  nameScope            Name prefix for the created ops.
  * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
  * @param  labelID              Optional label for which we want to compute the precision.
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  *
  * @author Emmanouil Antonios Platanios
  */
class GroupedPrecision(
    val nameScope: String,
    protected val defaultWeights: Option[Tensor[FLOAT32]] = None,
    val labelID: Option[Int] = None,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
) extends Metric[(Output, Output), Output] {
  /** Name of this metric. */
  override def name: String = nameScope

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
    val (predictions, targets) = values
    var ops = Set(predictions.op, targets.op)
    val computedWeights = getWeights(weights)
    computedWeights.foreach(ops += _.op)
    Op.createWithNameScope(name, ops) {
      val reshapedTargets = Metric.maybeExpandTargets(predictions, targets)
      val predictedIndices = predictions.toInt64
      val truePositives = Math.sum(sparseTruePositives(
        reshapedTargets, predictedIndices, labelID.map(_.toTensor.toOutput), computedWeights))
      val falsePositives = Math.sum(sparseFalsePositives(
        reshapedTargets, predictedIndices, labelID.map(_.toTensor.toOutput), computedWeights))
      // TODO: [DISTRIBUTE] Add support for aggregation across towers.
      val value = safeDiv(truePositives, truePositives + falsePositives, name = "Value")
      valuesCollections.foreach(Op.currentGraph.addToCollection(value, _))
      value
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
    VariableScope.scope(name) {
      Op.createWithNameScope(name, ops) {
        val reshapedTargets = Metric.maybeExpandTargets(predictions, targets)
        val predictedIndices = predictions.toInt64
        val truePositives = VariableScope.scope(name)(streamingSparseTruePositives(
          reshapedTargets, predictedIndices, labelID.map(_.toTensor.toOutput), computedWeights))
        val falsePositives = VariableScope.scope(name)(streamingSparseFalsePositives(
          reshapedTargets, predictedIndices, labelID.map(_.toTensor.toOutput), computedWeights))
        val tp = truePositives.value
        val fp = falsePositives.value
        val tpUpdate = truePositives.update
        val fpUpdate = falsePositives.update
        val tpReset = truePositives.reset
        val fpReset = falsePositives.reset
        val tpVariables = truePositives.variables
        val fpVariables = falsePositives.variables
        // TODO: [DISTRIBUTE] Add support for aggregation across towers.
        val value = safeDiv(tp, tp + fp, name = "Value")
        val update = safeDiv(tpUpdate, tpUpdate + fpUpdate, name = "Update")
        val reset = ControlFlow.group(Set(tpReset, fpReset), name = "Reset")
        valuesCollections.foreach(Op.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(Op.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(Op.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, tpVariables ++ fpVariables)
      }
    }
  }
}

object GroupedPrecision {
  /** Creates a new grouped precision metric.
    *
    * @param  nameScope            Name prefix for the created ops.
    * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
    * @param  labelID              Optional label for which we want to compute the precision.
    * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
    * @param  valuesCollections    Graph collections in which to add the metric values.
    * @param  updatesCollections   Graph collections in which to add the metric updates.
    * @param  resetsCollections    Graph collections in which to add the metric resets.
    * @return New grouped precision metric.
    */
  def apply(
      nameScope: String,
      defaultWeights: Option[Tensor[FLOAT32]] = None,
      labelID: Option[Int] = None,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
  ): GroupedPrecision = {
    new GroupedPrecision(
      nameScope, defaultWeights, labelID, variablesCollections, valuesCollections, updatesCollections,
      resetsCollections)
  }
}
