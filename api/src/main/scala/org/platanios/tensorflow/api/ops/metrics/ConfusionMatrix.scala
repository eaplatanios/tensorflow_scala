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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.metrics.Metric.{METRIC_RESETS, METRIC_UPDATES, METRIC_VALUES, METRIC_VARIABLES}
import org.platanios.tensorflow.api.ops.variables.{Variable, VariableScope, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Checks, Math, Op, Output, SparseOutput}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

/** Confusion matrix classification metric.
  *
  * The confusion matrix columns represent the prediction labels and the rows represent the target labels. The confusion
  * matrix is always a 2-D array of shape `[n, n]`, where `n` is the number of valid labels for a given classification
  * task. Both predictions and targets must be 1-D arrays of the same shape in order for this function to work.
  *
  * If `numClasses` is `null`, then `numClasses` will be set to the one plus the maximum value in either predictions or
  * targets. Class labels are expected to start at 0. E.g., if `numClasses` was three, then the possible labels would be
  * `[0, 1, 2]`.
  *
  * If `weights` is not `null`, then each prediction contributes its corresponding weight to the total value of the
  * confusion matrix cell.
  *
  * For example, for predictions `[1, 2, 4]` and targets `[2, 2, 4]`, the confusion matrix is equal to:
  * {{{
  *   [[0, 0, 0, 0, 0],
  *    [0, 0, 1, 0, 0],
  *    [0, 0, 1, 0, 0],
  *    [0, 0, 0, 0, 0],
  *    [0, 0, 0, 0, 1]]
  * }}}
  *
  * Note that the possible labels are assumed to be `[0, 1, 2, 3, 4]`, resulting in a 5x5 confusion matrix.
  *
  * @param  namescope            Name prefix for the created ops.
  * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
  * @param  numClasses           Number of classes over which the confusion matrix is computed.
  * @param  dataType             Data type for the confusion matrix.
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  *
  * @author Emmanouil Antonios Platanios
  */
class ConfusionMatrix(
    val namescope: String,
    protected val defaultWeights: Option[Tensor[FLOAT32]] = None,
    val numClasses: Int = -1,
    val dataType: DataType = FLOAT64,
    val variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
) extends Metric[(Output, Output), Output] {
  private[this] val numClassesOutput: Output = if (numClasses != -1) Basic.constant(numClasses) else null

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
    val predictions = values._1
    val targets = values._2
    val computedWeights = getWeights(weights)
    // Flatten the inputs if their rank is bigger than 1.
    val flattenedPredictions = if (predictions.rank > 1) Basic.reshape(predictions, -1) else predictions
    val flattenedTargets = if (targets.rank > 1) Basic.reshape(targets, -1) else targets
    val flattenedWeights = computedWeights.map(w => if (w != null && w.rank > 1) Basic.reshape(w, -1) else w)
    var ops = Set(predictions.op, targets.op)
    flattenedWeights.foreach(ops += _.op)
    if (numClassesOutput != null)
      ops += numClassesOutput
    Op.createWithNameScope(name, ops) {
      var (matchedPredictions, Some(matchedTargets), matchedWeights) =
        Metric.matchAxes(flattenedPredictions, Some(flattenedTargets), flattenedWeights)
      matchedPredictions = matchedPredictions.cast(INT64)
      matchedTargets = matchedTargets.cast(INT64)
      // Sanity checks: underflow or overflow can cause memory corruption.
      matchedPredictions = ControlFlow.withControlDependencies(
        Set(Checks.assertNonNegative(matchedPredictions, "'predictions' contains negative values")), matchedPredictions)
      matchedTargets = ControlFlow.withControlDependencies(
        Set(Checks.assertNonNegative(matchedTargets, "'targets' contains negative values")), matchedTargets)
      val inferredNumClasses = {
        if (numClassesOutput == null) {
          Math.add(Math.maximum(matchedPredictions.max(), matchedTargets.max()), 1)
        } else {
          val castedNumClasses = numClassesOutput.cast(INT64)
          matchedTargets = ControlFlow.withControlDependencies(
            Set(Checks.assertLess(matchedTargets, castedNumClasses)), "Some 'targets' are out of bounds.")
          matchedPredictions = ControlFlow.withControlDependencies(
            Set(Checks.assertLess(matchedPredictions, castedNumClasses)), "Some 'predictions' are out of bounds.")
          castedNumClasses
        }
      }
      matchedWeights = matchedWeights.map(_.cast(dataType))
      val computedWeights = matchedWeights.getOrElse(Basic.onesLike(matchedPredictions, dataType))
      val denseShape = Basic.stack(Seq(inferredNumClasses, inferredNumClasses)).cast(INT64)
      val indices = Basic.transpose(Basic.stack(Seq(matchedTargets, matchedPredictions)))
      val confusionMatrix = SparseOutput(indices, computedWeights, denseShape)
      val zeros = Basic.fill(dataType, denseShape.cast(INT32))(0)
      val value = confusionMatrix.addDense(zeros)
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
    Op.createWithNameScope(name) {
      VariableScope.scope(name) {
        val accumulator = Metric.variable(
          s"$name/Accumulator", dataType, Shape(numClasses, numClasses), ZerosInitializer,
          variablesCollections)
        val value = compute(values, weights, name = "Value")
        val update = accumulator.assignAdd(value, name = "Update")
        val reset = accumulator.initializer
        valuesCollections.foreach(Op.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(Op.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(Op.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(accumulator.value, update, reset, Set(accumulator))
      }
    }
  }
}

object ConfusionMatrix {
  /** Creates a new confusion matrix metric.
    *
    * @param  namescope            Name prefix for the created ops.
    * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
    * @param  numClasses           Number of classes over which the confusion matrix is computed.
    * @param  dataType             Data type for the confusion matrix.
    * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
    * @param  valuesCollections    Graph collections in which to add the metric values.
    * @param  updatesCollections   Graph collections in which to add the metric updates.
    * @param  resetsCollections    Graph collections in which to add the metric resets.
    * @return New confusion matrix metric.
    */
  def apply(
      namescope: String,
      defaultWeights: Option[Tensor[FLOAT32]] = None,
      numClasses: Int = -1, dataType: DataType = FLOAT64,
      variablesCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[Op]] = Set(METRIC_RESETS)
  ): ConfusionMatrix = {
    new ConfusionMatrix(
      namescope, defaultWeights, numClasses, dataType, variablesCollections, valuesCollections, updatesCollections,
      resetsCollections)
  }
}
