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

package org.platanios.tensorflow.api.ops.training.optimizers

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.{IsIntOrLong, IsNotQuantized, TF, FLOAT32}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.{Variable, ZerosInitializer}

import scala.language.postfixOps

// TODO: [OPTIMIZERS] Add support for sparse updates.
// TODO: [OPTIMIZERS] [QUANTIZATION] Add support for BFLOAT16 and simulated quantized bits.

/** Optimizer that implements the Adafactor optimization algorithm.
  *
  * Adafactor is most similar to the Adam optimization algorithm, presented in
  * [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
  *
  * The major differences are:
  *
  *   1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary parameters to maintain the
  *      second-moment estimator, instead of AB. This is advantageous on memory-limited systems. In addition, `beta1`
  *      (momentum) is set to zero by default, saving an additional auxiliary parameter per weight. Variables with
  *      `>= 3` dimensions are treated as collections of two-dimensional matrices -- factorization is over the last
  *      two dimensions.
  *   2. Adafactor incorporates "update-clipping" -- a scale-invariant analog of gradient clipping. This adds stability.
  *   3. Adafactor does not require an external "learning rate". By default, it incorporates a relative-update-scale
  *      schedule, corresponding to inverse-square-root learning-rate-decay in Adam. We hope this works well for most
  *      applications.
  *
  * The Adafactor update for step `t` is as follows:
  * {{{
  *   variable -= absoluteUpdateScale * clip(gradient / gradientScale)
  * }}}
  * where:
  * {{{
  *   absoluteUpdateScale = relativeUpdateScale * parameterScale
  *   relativeUpdateScale = min((t + 1) ^ -0.5, 1e-2)
  *   parameterScale = max(sqrt(mean(square(variable))), epsilon2)
  *   clip(x) = x / max(1.0, sqrt(mean(square(x))))
  *   gradientScale = sqrt(v_t)
  * }}}
  * and where the second moment estimator is maintained in a manner similar to Adam. It is initialized as:
  * {{{
  *   // If the variable is 2-dimensional:
  *   vr_0 = zeros[Float](Shape(numRows))
  *   vc_0 = zeros[Float](Shape(numColumns))
  *   // Otherwise:
  *   v = zeros[Float](variable.shape)
  * }}}
  * and its update rule is defined as:
  * {{{
  *   decayRate = 1 - (t + 1) ^ -0.8
  *   gradSquared = square(gradient) + epsilon1
  *   // If the variable is 2-dimensional:
  *   vr_t = decayRate * vr_{t-1} + (1 - decayRate) * mean(gradSquared, axes = 1)
  *   vc_t = decayRate * vc_{t-1} + (1 - decayRate) * mean(gradSquared, axes = 0)
  *   // Otherwise:
  *   v_t = decayRate * v_{t-1} + (1 - decayRate) * gradSquared
  * }}}
  *
  * For variables with `> 2` dimensions, we factorize the second-moment accumulator over the final 2 dimensions.
  *
  * Note that this optimizer does not support sparse updates and so all sparse updates are automatically converted to
  * dense ones.
  *
  * For more information on this algorithm, please refer to this [paper](https://arxiv.org/abs/1804.04235)
  * ([PDF](https://arxiv.org/pdf/1804.04235.pdf)).
  *
  * @param  learningRate             Learning rate. Must be `> 0`. If used with `decay`, then this argument
  *                                  specifies the initial value of the learning rate. In the Adafactor optimizer, the
  *                                  learning rate represents the `relativeUpdateScale`, if `multiplyByParameterScale`
  *                                  is `true`, and the `absoluteUpdateScale` otherwise. If not provided, then a
  *                                  reasonable default will be set based on the current iteration.
  * @param  decayRate                Decay rate of the second moment estimator. This should be set to a function such
  *                                  that: `1 - 1 / (iteration + 1) <= decayRate(iteration) < 1.0`. If not provided,
  *                                  then a reasinable default will be set based on the current iteration.
  * @param  decay                    Learning rate decay method to use for each update.
  * @param  multiplyByParameterScale If `true`, then the `absoluteUpdateScale` is computed adaptively, as described
  *                                  above. Otherwise, it is externally supplied as the provided `learningRate`.
  * @param  beta1                    Exponential decay rate for the first moment estimates. It must be in `[0, 1]`. The
  *                                  optimizer uses extra memory if this is non-zero.
  * @param  clippingThreshold        Optional update clipping threshold. Must be `>= 1.0`.
  * @param  factored                 If `true`, the second moment estimator is factored and thus the optimizer uses
  *                                  less memory.
  * @param  epsilon1                 Small constant used for numerical stability of the squared gradient. This epsilon
  *                                  corresponds to "epsilon hat" in the Kingma and Ba paper (in the formula just before
  *                                  Section 2.1), and not to the epsilon in Algorithm 1 of the paper.
  * @param  epsilon2                 Small constant used for numerical stability of the parameter scale.
  * @param  useLocking               If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
  *                                  behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag   Optional summary tag name to use for the learning rate value. If `null`, no summary
  *                                  is created for the learning rate. Otherwise, a scalar summary is created which can
  *                                  be monitored using TensorBoard.
  * @param  name                     Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class Adafactor protected (
    val learningRate: Option[Float] = None,
    val decayRate: Option[Float] = None,
    val decay: Schedule[Float] = FixedSchedule[Float](),
    val multiplyByParameterScale: Boolean = true,
    val beta1: Float = 0.0f,
    val clippingThreshold: Option[Float] = Some(1.0f),
    val factored: Boolean = true,
    val epsilon1: Float = 1e-30f,
    val epsilon2: Float = 1e-3f,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "Adafactor"
) extends Optimizer {
  override val ignoreDuplicateSparseIndices: Boolean = true

  protected var learningRateTensor     : Output[Float]         = _
  protected var decayRateTensor        : Output[Float]         = _
  protected var beta1Tensor            : Output[Float]         = _
  protected var clippingThresholdTensor: Option[Output[Float]] = None
  protected var epsilon1Tensor         : Output[Float]         = _
  protected var epsilon2Tensor         : Output[Float]         = _

  /** Decides whether to use a factored second moment estimate based on the variable shape. */
  protected def shouldUseFactoredSecondMomentEstimate(variableShape: Shape): Boolean = {
    factored && variableShape.rank > 1
  }

  protected def getLearningRate: Output[Float] = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    learningRateTensor
  }

  protected def getDecayRate: Output[Float] = {
    if (decayRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    decayRateTensor
  }

  protected def getBeta1: Output[Float] = {
    if (beta1Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    beta1Tensor
  }

  protected def getEpsilon1: Output[Float] = {
    if (epsilon1Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    epsilon1Tensor
  }

  protected def getEpsilon2: Output[Float] = {
    if (epsilon2Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    epsilon2Tensor
  }

  protected def getBetaPowerAccumulators: (Variable[Float], Variable[Float]) = {
    (getNonSlotVariable[Float]("Beta1Power", Op.currentGraph),
        getNonSlotVariable[Float]("Beta2Power", Op.currentGraph))
  }

  override def createSlots(variables: Seq[Variable[Any]]): Unit = {
    variables.foreach(v => {
      // Create slot for the first moment.
      if (beta1 > 0.0f) {
        zerosSlot("M", v, name)(TF.fromDataType(v.dataType))
      }

      if (shouldUseFactoredSecondMomentEstimate(v.shape)) {
        getSlot("Vr", v, FLOAT32, ZerosInitializer, v.shape(0 :: -1), name)
        getSlot("Vc", v, FLOAT32, ZerosInitializer, v.shape(0 :: -2) ++ v.shape(-1 ::), name)
      } else {
        getSlot("V", v, FLOAT32, ZerosInitializer, v.shape, name)
      }
    })
  }

  override def prepare[I: TF : IsIntOrLong](
      iteration: Option[Variable[I]]
  ): Unit = {
    val iterationWithDefault = iteration.map(_.value.toInt).getOrElse(Basic.constant[Int](0))
    val learningRateWithDefault = learningRate.map(l => Basic.constant(l, name = "LearningRate")).getOrElse(
      Op.nameScope("LearningRate") {
        Adafactor.defaultLearningRate(multiplyByParameterScale, iterationWithDefault)
      })
    learningRateTensor = decay(learningRateWithDefault, iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    decayRateTensor = decayRate.map(dr => Basic.constant(dr, name = "DecayRate")).getOrElse(
      Op.nameScope("DecayRate") {
        Adafactor.defaultDecayRate(iterationWithDefault)
      })
    beta1Tensor = Basic.constant(beta1, name = "Beta1")
    clippingThresholdTensor = clippingThreshold.map(Basic.constant(_, name = "ClippingThreshold"))
    epsilon1Tensor = Basic.constant(epsilon1, name = "Epsilon1")
    epsilon2Tensor = Basic.constant(epsilon2, name = "Epsilon2")
  }

  override def applyDense[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    Op.nameScope(s"$name/ApplyDense") {
      val gradFloat = gradient.toFloat
      val gradSquared = Math.square(gradFloat) + getEpsilon1
      val gradSquaredMean = Math.mean(gradSquared)
      var decayRate = getDecayRate
      var updateScale = getLearningRate

      if (multiplyByParameterScale) {
        // We estimate the scale of the parameters from the current values. We include a minimum value of 0.001 to give
        // the optimizer a chance to escape 0 if it was zero-initialized. Instead of using the value, we could impute
        // the scale from the shape, as initializers do.
        val parameterScale = Math.maximum(Math.sqrt(Math.mean(Math.square(variable.value.toFloat))), getEpsilon2)
        updateScale *= parameterScale
      }

      // The following two lines are a hack to make things dependent on the gradient tensor. This confounds the XLA
      // rewriter and keeps it from fusing computations across different variables. This fusion is a bad for HBM usage,
      // since it causes the gradients to persist in memory.
      decayRate += gradSquaredMean * 1e-30f
      updateScale += gradSquaredMean * 1e-30f

      val mixingRate = 1.0f - decayRate
      var updates = Set.empty[UntypedOp]

      var update = {
        if (shouldUseFactoredSecondMomentEstimate(variable.shape)) {
          val gradSquaredRowMean = Math.mean(gradSquared, axes = -1)
          val gradSquaredColMean = Math.mean(gradSquared, axes = -2)
          val vr = getSlot[T, Float]("Vr", variable)
          val vc = getSlot[T, Float]("Vc", variable)
          val newVr = decayRate * vr + mixingRate * gradSquaredRowMean
          val newVc = decayRate * vc + mixingRate * gradSquaredColMean
          updates += Variable.assign(vr.handle, newVr, "Vr/Update")
          updates += Variable.assign(vc.handle, newVc, "Vr/Update")
          val longTermMean = Math.mean(newVr, axes = -1, keepDims = true)
          val rFactor = Math.rsqrt(newVr / longTermMean).expandDims(-1)
          val cFactor = Math.rsqrt(newVc).expandDims(-2)
          gradFloat * rFactor * cFactor
        } else {
          val v = getSlot[T, Float]("V", variable)
          val newV = decayRate * v + mixingRate * gradSquared
          updates += Variable.assign(v.handle, newV, "V/Update")
          gradFloat * Math.rsqrt(newV)
        }
      }

      clippingThresholdTensor.foreach(threshold => {
        update /= Math.maximum(1.0f, Math.sqrt(Math.mean(Math.square(update))) / threshold)
      })

      update *= updateScale

      if (beta1 > 0.0f) {
        val m = getSlot[T, T]("M", variable)
        val newM = getBeta1 * m.toFloat + (1.0f - getBeta1) * update
        update = newM
        updates += Variable.assign(m.handle, newM.castTo[T], "M/Update")
      }

      val newValue = variable.value.toFloat - update
      updates += Variable.assign(variable.handle, newValue.castTo[T], "Update")

      ControlFlow.group(updates)
    }
  }

  override def applySparse[T: TF : IsNotQuantized, I: TF : IsIntOrLong](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
    Op.nameScope(s"$name/ApplySparse") {
      applyDense(gradient.toOutput, variable, iteration)
    }
  }
}

object Adafactor {
  def apply(
      learningRate: Option[Float] = None,
      decayRate: Option[Float] = None,
      decay: Schedule[Float] = FixedSchedule[Float](),
      multiplyByParameterScale: Boolean = true,
      beta1: Float = 0.0f,
      clippingThreshold: Option[Float] = Some(1.0f),
      factored: Boolean = true,
      epsilon1: Float = 1e-30f,
      epsilon2: Float = 1e-3f,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "Adafactor"
  ): Adafactor = {
    new Adafactor(
      learningRate, decayRate, decay, multiplyByParameterScale, beta1, clippingThreshold,
      factored, epsilon1, epsilon2, useLocking, learningRateSummaryTag, name)
  }

  def defaultLearningRate(
      multiplyByParameterScale: Boolean,
      iteration: Output[Int]
  ): Output[Float] = {
    val learningRate = Math.minimum(Math.rsqrt(iteration.toFloat + 1.0f), Basic.constant[Float](0.01f))
    if (multiplyByParameterScale)
      learningRate * 0.05f
    else
      learningRate
  }

  def defaultDecayRate(
      iteration: Output[Int]
  ): Output[Float] = {
    defaultDecayRatePow(iteration, 0.8f)
  }

  /** Second-moment decay rate where the memory-length grows as `iteration ^ exponent`. */
  def defaultDecayRatePow(
      iteration: Output[Int],
      exponent: Output[Float]
  ): Output[Float] = {
    Basic.constant[Float](1.0f) - Math.pow(iteration.toFloat + 1.0f, -exponent)
  }

  /** Second-moment decay rate similar to that of Adam, subsuming the correction factor. */
  def defaultDecayRateAdam(
      iteration: Output[Int],
      beta2: Output[Float]
  ): Output[Float] = {
    beta2 * (1.0f - Math.pow(beta2, iteration.toFloat)) / (1.0f - Math.pow(beta2, iteration.toFloat + 1.0f))
  }
}
