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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.ZerosInitializer
import org.platanios.tensorflow.api.ops.{Basic, Checks, Math, Op, Output, SparseOutput}
import org.platanios.tensorflow.api.types.{DataType, FLOAT64, INT32, INT64}

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
  * @author Emmanouil Antonios Platanios
  */
case class ConfusionMatrix(
    numClasses: Int = -1,
    dataType: DataType = FLOAT64,
    override val name: String = "ConfusionMatrix")
    extends Metric {
  private[this] val numClassesOutput: Output = if (numClasses != -1) Basic.constant(numClasses) else null

  override def compute(predictions: Output, targets: Output, weights: Output = null, name: String = name): Output = {
    // Flatten the inputs if their rank is bigger than 1.
    val flattenedPredictions = if (predictions.rank > 1) Basic.reshape(predictions, -1) else predictions
    val flattenedTargets = if (targets.rank > 1) Basic.reshape(targets, -1) else targets
    val flattenedWeights = if (weights != null && weights.rank > 1) Basic.reshape(weights, -1) else weights
    var values = Set(predictions.op, targets.op)
    if (flattenedWeights != null)
      values += flattenedWeights
    if (numClassesOutput != null)
      values += numClassesOutput
    Op.createWithNameScope(name, values) {
      var (matchedPredictions, matchedTargets, matchedWeights) =
        Metric.matchAxes(flattenedPredictions, flattenedTargets, flattenedWeights)
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
      if (matchedWeights != null)
        matchedWeights = matchedWeights.cast(dataType)
      else
        matchedWeights = Basic.onesLike(matchedPredictions, dataType)
      val denseShape = Basic.stack(Seq(inferredNumClasses, inferredNumClasses)).cast(INT64)
      val indices = Basic.transpose(Basic.stack(Seq(matchedTargets, matchedPredictions)))
      val confusionMatrix = SparseOutput(indices, matchedWeights, denseShape)
      val zeros = Basic.fill(dataType, denseShape.cast(INT32))(0)
      confusionMatrix + zeros
    }
  }

  override def streaming(
      predictions: Output, targets: Output, weights: Output = null, name: String = name): (Output, Op, Output) = {
    val accumulator = Metric.localVariable(
      s"$name/Accumulator", dataType, Shape(numClasses, numClasses), ZerosInitializer)
    val value = compute(predictions, targets, weights, name = s"$name/Value")
    val update = accumulator.assignAdd(value, name = s"$name/Update")
    val reset = accumulator.initializer
    (accumulator.value, reset, update)
  }
}
