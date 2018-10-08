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
import org.platanios.tensorflow.api.core.Graph.Keys.{OpCollectionKey, OutputCollectionKey, VariableCollectionKey}
import org.platanios.tensorflow.api.core.exception.{InvalidShapeException, ShapeMismatchException}
import org.platanios.tensorflow.api.core.types.{TF, IsNotQuantized}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Checks, Math, Op, Output, Sets, UntypedOp}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables.{Initializer, Variable, ZerosInitializer}
import org.platanios.tensorflow.api.tensors.Tensor

/** Trait representing evaluation metrics that support both eager computation, as well as computation in a streaming
  * manner.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Metric[T, R] {
  /** Name of this metric. */
  def name: String

  /** Weights to multiply the provided values with when computing the value of this metric. */
  def weights: Option[Tensor[Float]] = None

  /** Computes the value of this metric for the provided values, optionally weighted by `weights`.
    *
    * @param  values  Values.
    * @param  weights Optional tensor containing weights for the values.
    * @param  name    Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  def compute(
      values: T,
      weights: Option[Output[Float]] = None,
      name: String = s"$name/Compute"
  ): R

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  values  Values.
    * @param  weights Optional tensor containing weights for the predictions.
    * @param  name    Name prefix for the created ops.
    * @return Tuple containing: (i) an output representing the current value of the metric, (ii) an op used to update
    *         its current value and obtain the new value, and (iii) an op used to reset its value.
    */
  def streaming(
      values: T,
      weights: Option[Output[Float]] = None,
      name: String = s"$name/Streaming"
  ): Metric.StreamingInstance[R]

  protected def getWeights(
      providedWeights: Option[Output[Float]]
  ): Option[Output[Float]] = {
    providedWeights
        .map(w => this.weights.map(w * _).getOrElse(w))
        .orElse(this.weights.map(_.toOutput))
  }

  override def toString: String = name
}

object Metric {
  case class StreamingInstance[R](
      value: R,
      update: R,
      reset: UntypedOp,
      variables: Set[Variable[Any]])

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
  def variable[T: TF](
      name: String,
      shape: Shape = null,
      initializer: Initializer = ZerosInitializer,
      collections: Set[Graph.Key[Variable[Any]]] = Set.empty
  ): Variable[T] = {
    // TODO: [DISTRIBUTE] Add support for the distribute API.
    Variable.getVariable[T](
      name, shape, initializer, trainable = false,
      collections = collections ++ Set(Metric.METRIC_VARIABLES, Graph.Keys.LOCAL_VARIABLES))
  }

  /** Divides two values, returning 0 if the denominator is <= 0. */
  def safeDiv[T: TF : IsNotQuantized](
      numerator: Output[T],
      denominator: Output[T],
      name: String = "SafeDiv"
  ): Output[T] = {
    val zero = Basic.zeros[T](Shape())
    Math.select(
      condition = Math.greater(denominator, zero),
      x = Math.divide(numerator, denominator),
      y = zero,
      name = name)
  }

  /** Divides two scalar values, returning 0 if the denominator is 0. */
  def safeScalarDiv[T: TF : IsNotQuantized](
      numerator: Output[T],
      denominator: Output[T],
      name: String = "SafeScalarDiv"
  ): Output[T] = {
    val zeros = Basic.zerosLike(denominator)
    Math.select(
      condition = Math.equal(denominator, zeros),
      x = zeros,
      y = Math.divide(numerator, denominator),
      name = name)
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
  private[metrics] def weightsBroadcast[T: TF](
      values: Output[T],
      weights: Output[Float],
      name: String = "BroadcastWeights"
  ): Output[Float] = {
    Op.nameScope(name) {
      // Try static check for exact match.
      if (values.shape.isFullyDefined &&
          weights.shape.isFullyDefined &&
          values.shape.isCompatibleWith(weights.shape)
      ) {
        weights
      } else {
        Op.createWith(controlDependencies = Set(weightsAssertBroadcastable(values, weights))) {
          Math.multiply(weights, Basic.onesLike(values).castTo[Float], name = "Broadcast")
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
  private[metrics] def weightsAssertBroadcastable[T: TF](
      values: Output[T],
      weights: Output[Float],
      name: String = "AssertBroadcastable"
  ): UntypedOp = {
    Op.nameScope(name) {
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
        case (Some(_), _, Some(wR), _) if wR.scalar == 0 =>
          ControlFlow.noOp("StaticScalarCheckSuccess")
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
  private def weightsHaveValidNonScalarShape(
      valuesRank: Output[Int],
      valuesShape: Output[Long],
      weightsRank: Output[Int],
      weightsShape: Output[Long],
      name: String = "WeightsHaveValidNonScalarShape"
  ): Output[Boolean] = {
    Op.nameScope(name) {
      val isSameRank = Math.equal(valuesRank, weightsRank, name = "IsSameRank")
      ControlFlow.cond(
        isSameRank,
        () => weightsHaveValidDimensions(valuesShape, weightsShape),
        () => isSameRank)
    }
  }

  /** Returns a boolean tensor indicating whether `weightsShape` has valid dimensions for broadcasting to
    * `valuesShape`. */
  private def weightsHaveValidDimensions(
      valuesShape: Output[Long],
      weightsShape: Output[Long],
      name: String = "WeightsHaveValidDimensions"
  ): Output[Boolean] = {
    Op.nameScope(name) {
      val reshapedValuesShape = Basic.expandDims(valuesShape, -1)
      val reshapedWeightsShape = Basic.expandDims(weightsShape, -1)
      val validDimensions = Basic.concatenate(Seq(reshapedValuesShape, Basic.onesLike(reshapedValuesShape)), 1)
      val invalidDimensions = Sets.setDifference(reshapedWeightsShape, validDimensions)
      val numInvalidDimensions = Basic.size(invalidDimensions.values, name = "NumInvalidDimensions")
      Math.equal(0L, numInvalidDimensions)
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
  def matchAxes[P: TF, T: TF](
      predictions: Output[P],
      targets: Option[Output[T]] = None,
      weights: Option[Output[Float]] = None,
      expectedRankDiff: Int = 0
  ): (Output[P], Option[Output[T]], Option[Output[Float]]) = {
    var matchedPredictions = predictions
    var matchedTargets = targets
    targets match {
      case None => ()
      case Some(t) =>
        if (predictions.rank != -1 && t.rank != -1) {
          // Use static rank.
          val rankDiff = predictions.rank - t.rank
          if (rankDiff == expectedRankDiff + 1)
            matchedPredictions = Basic.squeeze(matchedPredictions, Seq(-1))
          else if (rankDiff == expectedRankDiff - 1)
            matchedTargets = matchedTargets.map(Basic.squeeze(_, Seq(-1)))
        } else {
          // Use dynamic rank.
          val rankDiff = Basic.rank(predictions) - Basic.rank(t)
          if (predictions.rank == -1 || predictions.shape(-1) == -1 || predictions.shape(-1) == 1) {
            matchedPredictions = ControlFlow.cond(
              Math.equal(rankDiff, expectedRankDiff + 1),
              () => Basic.squeeze(matchedPredictions, Seq(-1)),
              () => matchedPredictions)
          }
          if (t.rank == -1 || t.shape(-1) == -1 || t.shape(-1) == 1) {
            matchedTargets = matchedTargets.map(mt => {
              ControlFlow.cond(
                Math.equal(rankDiff, expectedRankDiff - 1),
                () => Basic.squeeze(mt, Seq(-1)),
                () => mt)
            })
          }
        }
    }
    weights match {
      case None =>
        (matchedPredictions, matchedTargets, weights)
      case Some(w) if w.rank == 0 =>
        (matchedPredictions, matchedTargets, weights)
      case Some(w) =>
        var matchedWeights = weights
        if (predictions.rank != -1 && w.rank != -1) {
          // Use static rank.
          val rankDiff = predictions.rank - w.rank
          if (rankDiff == expectedRankDiff + 1)
            matchedWeights = matchedWeights.map(Basic.expandDims(_, -1))
          else if (rankDiff == expectedRankDiff - 1)
            matchedWeights = matchedWeights.map(Basic.squeeze(_, Seq(-1)))
        } else {
          // Use dynamic rank.
          val rankDiff = Basic.rank(predictions) - Basic.rank(w)
          // If weights are scalar, do nothing. Otherwise, try to add or remove an axis to match predictions.
          matchedWeights = matchedWeights.map(mw => {
            ControlFlow.cond(
              Math.equal(mw, 0.0f),
              () => mw,
              () => ControlFlow.cond(
                Math.equal(rankDiff, -1),
                if (w.rank == -1 || w.shape(-1) == -1 || w.shape(-1) == 1)
                  () => Basic.expandDims(mw, -1)
                else
                  () => mw,
                () => ControlFlow.cond(
                  Math.equal(rankDiff, 1),
                  () => Basic.squeeze(mw, Seq(-1)),
                  () => mw)))
          })
        }
        (matchedPredictions, matchedTargets, matchedWeights)
    }
  }

  /** If necessary, this method, expands `targets` along the last aixs to match the shape of `predictions`.
    *
    * @param  predictions Tensor with shape `[D1, ..., DN, numLabels]`.
    * @param  targets     Tensor with shape `[D1, ..., DN, numLabels]` or `[D1, ..., DN]`. The latter implies
    *                     `numLabels = 1`, in which case the result is an expanded `targets` with shape
    *                     `[D1, ..., DN, 1]`.
    * @return Potentially reshaped `targets` with the same shape as the provided `predictions`.
    * @throws ShapeMismatchException If the targets shape is neither equal to `[D1, ..., DN, numLabels]` or
    *                                `[D1, ..., DN]`.
    */
  @throws[ShapeMismatchException]
  def maybeExpandTargets[P: TF, T: TF](
      predictions: Output[P],
      targets: Output[T]
  ): Output[T] = {
    // TODO: [SPARSE] Support sparse `targets`.
    Op.nameScope("MaybeExpandTargets") {
      if (predictions.rank > -1 && targets.rank > -1) {
        // We first try to use static shape information.
        if (predictions.rank == targets.rank)
          targets
        else if (predictions.rank == targets.rank + 1)
          Basic.expandDims(targets, -1)
        else
          throw ShapeMismatchException(
            s"Unexpected targets shape '${targets.shape}', for predictions shape '${predictions.shape}'.")
      } else {
        // Otherwise, we use dynamic shape information.
        ControlFlow.cond(
          Math.equal(Basic.rank(predictions), Basic.rank(targets) + 1),
          () => Basic.expandDims(targets, -1),
          () => targets)
      }
    }
  }
}
