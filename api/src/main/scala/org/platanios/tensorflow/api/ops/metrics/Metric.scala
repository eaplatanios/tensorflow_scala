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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.Graph.Keys.{OpCollectionKey, OutputCollectionKey, VariableCollectionKey}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.{Initializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Checks, Math, Op, Output, Sets}
import org.platanios.tensorflow.api.types.DataType

/** Trait representing evaluation metrics that support both eager computation, as well as computation in a streaming
  * manner.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Metric[T, R] {
  /** Name of this metric. */
  val name: String

  /** Computes the value of this metric for the provided values, optionally weighted by `weights`.
    *
    * @param  values  Values.
    * @param  weights Tensor containing weights for the values.
    * @param  name    Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  def compute(values: T, weights: Output = null, name: String = name): R

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  values  Values.
    * @param  weights Tensor containing weights for the predictions.
    * @param  name    Name prefix for the created ops.
    * @return Tuple containing: (i) an output representing the current value of the metric, (ii) an op used to update
    *         its current value and obtain the new value, and (iii) an op used to reset its value.
    */
  def streaming(values: T, weights: Output = null, name: String = name): Metric.StreamingInstance[R]

  override def toString: String = name
}

object Metric {
  case class StreamingInstance[R](value: R, update: R, reset: Op, variables: Set[Variable])

  /** Key to collect the subset of `Variable` objects that are used for computing and storing metric values. */
  object METRIC_VARIABLES extends VariableCollectionKey {
    override def name: String = "metric_variables"
  }

  /** Key to collect the subset of tensors that are used for obtaining metric values. */
  object METRIC_VALUES extends OutputCollectionKey {
    override def name: String = "metric_values"
  }

  /** Key to collect the subset of tensors that are used for updating metric values. */
  object METRIC_UPDATES extends OutputCollectionKey {
    override def name: String = "metric_updates"
  }

  /** Key to collect the subset of tensors that are used for resetting metric values. */
  object METRIC_RESETS extends OpCollectionKey {
    override def name: String = "metric_resets"
  }

  /** Creates a new variable and adds it to the `LOCAL_VARIABLES` graph collection. */
  private[metrics] def localVariable(
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = ZerosInitializer,
      collections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES)): Variable = {
    Variable.getVariable(
      name = name, dataType = dataType, shape = shape, trainable = false,
      collections = collections + Graph.Keys.LOCAL_VARIABLES)
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

  /** Broadcast `weights` to the same shape as `values`.
    *
    * This method a version of `weights` following the same broadcasting rules as `multiply(weights, values)`, but
    * limited to the weights shapes allowed by `weightsAssertBroadcastable`. When computing a weighted average, use this
    * function to broadcast `weights` before summing them. For example, `sum(w * v) / sum(weightsBroadcast(w, v))`.
    *
    * @param  values  Values tensor.
    * @param  weights Weights tensor.
    * @param  name    Name prefix for the created ops.
    * @return Broadcasted weights.
    */
  private[metrics] def weightsBroadcast(values: Output, weights: Output, name: String = "BroadcastWeights"): Output = {
    Op.createWithNameScope(name, Set(values.op, weights.op)) {
      // Try static check for exact match.
      if (values.shape.isFullyDefined && weights.shape.isFullyDefined && values.shape.isCompatibleWith(weights.shape)) {
        weights
      } else {
        Op.createWith(controlDependencies = Set(weightsAssertBroadcastable(values, weights))) {
          Math.multiply(weights, Basic.onesLike(values), name = "Broadcast")
        }
      }
    }
  }

  /** Asserts that `weights` can be broadcast to the same shape as `values`.
    *
    * In the `metrics` package, we support limited weight broadcasting. We let weights be either scalar, or the same
    * rank as the target values, with each dimension being equal to either 1, or to the corresponding `values`
    * dimension.
    *
    * @param  values  Values tensor.
    * @param  weights Weights tensor.
    * @param  name    Name prefix for the created ops.
    * @return Assertion op for the assertion that `weights` can be broadcast to the same shape as `values`.
    */
  private[this] def weightsAssertBroadcastable(
      values: Output, weights: Output, name: String = "AssertBroadcastable"): Op = {
    Op.createWithNameScope(name, Set(values.op, weights.op)) {
      val valuesRank = Basic.rank(values, name = "Values/Rank")
      val valuesShape = Basic.shape(values, name = "Values/Shape")
      val valuesRankStatic = Output.constantValue(valuesRank)
      val valuesShapeStatic = Output.constantValueAsShape(valuesShape)
      val weightsRank = Basic.rank(weights, name = "Weights/Rank")
      val weightsShape = Basic.shape(weights, name = "Weights/Shape")
      val weightsRankStatic = Output.constantValue(weightsRank)
      val weightsShapeStatic = Output.constantValueAsShape(weightsShape)
      // Try static checks.
      (valuesRankStatic, valuesShapeStatic, weightsRankStatic, weightsShapeStatic) match {
        case (Some(_), _, Some(wR), _) if wR.scalar == 0 => ControlFlow.noOp("StaticScalarCheckSuccess")
        case (Some(vR), _, Some(wR), _) if vR != wR => throw InvalidShapeException(
          "'weights' can not be broadcasted to 'values'. " +
              s"values.rank = ${vR.scalar}, " +
              s"weights.rank = ${wR.scalar}, " +
              s"values.shape = ${values.shape}, " +
              s"weights.shape = ${weights.shape}.")
        case (_, Some(vS), _, Some(wS)) =>
          vS.asArray.zip(wS.asArray).zipWithIndex.foreach(
            s => if (s._1._2 != 1 && s._1._2 != s._1._1)
              throw InvalidShapeException(
                s"'weights' can not be broadcasted to 'values'. Mismatch at axis ${s._2}. " +
                    s"values.shape = $vS, " +
                    s"weights.shape = $wS."))
          ControlFlow.noOp("StaticShapeCheckSuccess")
        case _ =>
          // Dynamic checks.
          val isScalar = Math.equal(0, weightsRank, name = "IsScalar")
          val isValidShape = ControlFlow.cond(
            isScalar,
            () => isScalar,
            () => weightsHaveValidNonScalarShape(valuesRank, valuesShape, weightsRank, weightsShape),
            name = "IsValidShape")
          Checks.assert(isValidShape, Seq(
            "'weights' can not be broadcasted to 'values'. ",
            "values.shape = ", values.name, valuesShape,
            "weights.shape = ", weights.name, weightsShape,
            "isScalar = ", isScalar), name = "IsValidShapeAssertion")
      }
    }
  }

  /** Returns a boolean tensor indicating whether `weightsShape` has valid non-scalar dimensions for broadcasting to
    * `valuesShape`. */
  private[this] def weightsHaveValidNonScalarShape(
      valuesRank: Output, valuesShape: Output, weightsRank: Output, weightsShape: Output,
      name: String = "WeightsHaveValidNonScalarShape"): Output = {
    Op.createWithNameScope(name, Set(valuesRank.op, valuesShape.op, weightsRank.op, weightsShape.op)) {
      val isSameRank = Math.equal(valuesRank, weightsRank, name = "IsSameRank")
      ControlFlow.cond(
        isSameRank,
        () => weightsHaveValidDimensions(valuesShape, weightsShape),
        () => isSameRank)
    }
  }

  /** Returns a boolean tensor indicating whether `weightsShape` has valid dimensions for broadcasting to
    * `valuesShape`. */
  private[this] def weightsHaveValidDimensions(
      valuesShape: Output, weightsShape: Output, name: String = "WeightsHaveValidDimensions"): Output = {
    Op.createWithNameScope(name, Set(valuesShape.op, weightsShape.op)) {
      val reshapedValuesShape = Basic.expandDims(valuesShape, -1)
      val reshapedWeightsShape = Basic.expandDims(weightsShape, -1)
      val validDimensions = Basic.concatenate(Seq(reshapedValuesShape, Basic.onesLike(reshapedValuesShape)), 1)
      val invalidDimensions = Sets.setDifference(reshapedWeightsShape, validDimensions)
      val numInvalidDimensions = Basic.size(invalidDimensions.values, name = "NumInvalidDimensions")
      Math.equal(0, numInvalidDimensions)
    }
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
  def matchAxes(
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
            () => ControlFlow.cond(
              Math.equal(rankDiff, 1),
              () => Basic.squeeze(matchedWeights, Seq(-1)),
              () => matchedWeights)))
      }
      (matchedPredictions, matchedTargets, matchedWeights)
    }
  }
}
