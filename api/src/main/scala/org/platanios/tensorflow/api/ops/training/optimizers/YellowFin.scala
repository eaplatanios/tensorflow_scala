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

import org.platanios.tensorflow.api.core.{NewAxis, Shape}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.ExponentialMovingAverage
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{FLOAT32, INT32}

/** Optimizer that implements the YellowFin algorithm.
  *
  * Please refer to [Zhang et. al., 2017](https://arxiv.org/abs/1706.03471) for details.
  *
  * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument specifies the
  *                                initial value of the learning rate.
  * @param  decay                  Learning rate decay method to use for each update.
  * @param  momentum               Momentum. Must be `>= 0`.
  * @param  beta                   Smoothing parameter for estimations.
  * @param  curvatureWindowWidth   Curvature window width. Must be `> 1`.
  * @param  zeroDebias             If `true`, the moving averages will be zero-debiased.
  * @param  sparsityDebias         The gradient norm and curvature are biased towards larger values when computed for
  *                                sparse gradients. This is useful when the model is very sparse, e.g. LSTMs with word
  *                                embeddings. For non-sparse CNNs, turning it off could slightly accelerate the
  *                                algorithm's speed.
  * @param  useNesterov            Boolean value indicating whether to use Nesterov acceleration or not. For details,
  *                                refer to [Sutskever et. al., 2013](http://proceedings.mlr.press/v28/sutskever13.pdf).
  * @param  useLocking             If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
  *                                behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag Optional summary tag name to use for the learning rate value. If `null`, no summary
  *                                is created for the learning rate. Otherwise, a scalar summary is created which can be
  *                                monitored using TensorBoard.
  * @param  name                   Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class YellowFin protected (
    override val learningRate: Double = 1.0,
    override val decay: Schedule = FixedSchedule,
    override val momentum: Double = 0.0,
    val beta: Float = 0.999f,
    val curvatureWindowWidth: Int = 20,
    val zeroDebias: Boolean = true,
    val sparsityDebias: Boolean = true,
    override val useNesterov: Boolean = false,
    override val useLocking: Boolean = false,
    override val learningRateSummaryTag: String = null,
    override val name: String = "YellowFin"
) extends GradientDescent(learningRate, decay, momentum, useNesterov, useLocking, learningRateSummaryTag, name) {
  protected var learningRateVariable      : Variable = _
  protected var learningRateFactorVariable: Variable = _
  protected var momentumVariable          : Variable = _

  protected var movingAverage  : ExponentialMovingAverage = _
  protected var curvatureWindow: Variable                 = _
  protected var betaTensor     : Output                   = _
  protected var step           : Variable                 = _
  protected var incrementStepOp: Op                       = _
  protected var doTune         : Output                   = _

  override protected def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(learningRateTensor, variable.dataType)
  }

  override protected def getMomentum(variable: Variable): Output = {
    if (momentumTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(momentumTensor, variable.dataType)
  }

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    variables.foreach(v => zerosSlot("Momentum", v, name))
  }

  /** Creates an op that applies the provided gradients to the provided variables.
    *
    * @param  gradientsAndVariables Sequence with gradient-variable pairs.
    * @param  iteration             Optional `Variable` to increment by one after the variables have been updated.
    * @param  name                  Name for the created op.
    * @return Created op.
    */
  override def applyGradients(
      gradientsAndVariables: Seq[(OutputLike, Variable)],
      iteration: Option[Variable] = None,
      name: String = this.name
  ): Op = {
    // We first apply the gradient descent updates (with momentum).
    val applyGradientsOp = super.applyGradients(gradientsAndVariables, iteration, name)

    Op.createWithNameScope(name) {
      // We then apply the YellowFin update ops that tune the momentum and the learning rate.
      val yellowFinUpdateOp = Op.createWith(controlDependencies = Set(applyGradientsOp)) {
        yellowFinUpdate(gradientsAndVariables)
      }

      ControlFlow.group(Set(applyGradientsOp, yellowFinUpdateOp, incrementStepOp))
    }
  }

  override def prepare(iteration: Option[Variable]): Unit = {
    movingAverage = ExponentialMovingAverage(beta, zeroDebias = zeroDebias)
    val lr = Tensor(learningRate)
    val mu = Tensor(momentum)
    learningRateVariable = Variable.getVariable(
      "LearningRate", FLOAT32, Shape(), ConstantInitializer(lr), trainable = false)
    momentumVariable = Variable.getVariable(
      "Momentum", FLOAT32, Shape(), ConstantInitializer(mu), trainable = false)
    learningRateFactorVariable = Variable.getVariable(
      "LearningRateFactor", FLOAT32, Shape(), OnesInitializer, trainable = false)
    learningRateTensor = decay(learningRateVariable.value * learningRateFactorVariable.value, iteration)
    momentumTensor = momentumVariable.value
    betaTensor = Basic.constant(beta, momentumVariable.dataType, name = "Beta")
    step = Variable.getVariable("Step", INT32, Shape(), ZerosInitializer, trainable = false)
    incrementStepOp = step.assignAdd(1, "IncrementStep").op
    doTune = Math.greater(step.value, 0)
    curvatureWindow = Variable.getVariable(
      "CurvatureWindow", FLOAT32, Shape(curvatureWindowWidth), ZerosInitializer, trainable = false)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    GradientDescent.resourceApplyMomentumDense(
      variable = variable,
      accumulator = getSlot("Momentum", variable),
      stepSize = getLearningRate(variable, iteration),
      gradient = gradient,
      momentum = getMomentum(variable),
      useLocking = useLocking,
      useNesterov = useNesterov)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    GradientDescent.resourceApplyMomentumSparse(
      variable = variable,
      accumulator = getSlot("Momentum", variable),
      stepSize = getLearningRate(variable, iteration),
      gradient = gradient.values,
      indices = gradient.indices,
      momentum = getMomentum(variable),
      useLocking = useLocking,
      useNesterov = useNesterov)
  }

  protected def yellowFinUpdate(gradientsAndVariables: Seq[(OutputLike, Variable)]): Op = {
    val gradSquared = gradientsAndVariables.map(gv => Op.colocateWith(Set(gv._2.op))(Math.square(gv._1)))
    val gradNormSquared = gradSquared.map(Math.sum(_))
    val gradNormSquaredAvgOp = movingAverage.computeForValues(gradNormSquared.toSet)
    val (gradNormSquaredSum, gradNormSquaredAvg) = Op.createWith(controlDependencies = Set(gradNormSquaredAvgOp)) {
      val sum = Math.addN(gradNormSquared)
      val avg = Math.addN(gradNormSquared.map(movingAverage.average(_).get.read()))
      (sum, avg)
    }
    val gradients = gradientsAndVariables.map(_._1)
    val sparsityAvg = gradientsSparsity(gradients)
    val (hMin, hMax) = curvatureRange(gradNormSquaredSum, sparsityAvg)
    val gradVar = gradientsVariance(gradients, gradNormSquaredAvg, sparsityAvg)
    val distAvg = distanceToOptimum(gradNormSquaredSum, gradNormSquaredAvg, sparsityAvg)

    // Single-step: minimizes the surrogate for the expected squared distance from the optimum of a local quadratic
    // approximation after a single step while keeping all directions in the robust region.

    def getMomentum: Output = {
      // We have the equation x^2 D^2 + (1-x)^4 * C / h_min^2, where x = sqrt(mu). We substitute x, which is sqrt(mu),
      // with x = y + 1. It gives y^3 + py = q, where p = (D^2 h_min^2)/(2*C) and q = -p. We use the Vieta's
      // substitution (http://mathworld.wolfram.com/VietasSubstitution.html) to compute the root. There is only one real
      // solution y (which is in [0, 1]).
      val p = Math.square(distAvg) * Math.square(hMin) / (2.0f * gradVar)
      val w3 = (-Math.sqrt(Math.square(p) + 4.0f * Math.pow(p, 3) / 27.0f) - p) / 2.0f
      val w = Math.sign(w3) * Math.pow(Math.abs(w3), 1.0f / 3.0f)
      val y = w - p / (3.0f * w)
      val cubicRoot = y + 1.0f
      val dr = hMax / hMin
      Math.maximum(Math.square(cubicRoot), Math.square((Math.sqrt(dr) - 1.0f) / (Math.sqrt(dr) + 1.0f)))
    }

    val momentum = Basic.identity(ControlFlow.cond(
      doTune,
      () => getMomentum,
      () => momentumVariable.value,
      name = "MomentumTuneCond"))

    def getLearningRate: Output = {
      Math.square(1.0f - Math.sqrt(momentum)) / hMin
    }

    val learningRate = Op.createWith(controlDependencies = Set(momentum.op)) {
      Basic.identity(ControlFlow.cond(
        doTune,
        () => getLearningRate,
        () => learningRateVariable.value,
        name = "LearningRateTuneCond"))
    }

    // Tune the learning rate and the momentum.
    val updateOps = Op.createWith(controlDependencies = Set(momentum.op, learningRate.op)) {
      val updatedMu = Math.add(betaTensor * momentumVariable.value, (1.0f - betaTensor) * momentum)
      val updatedLR = Math.add(betaTensor * learningRateVariable.value, (1.0f - betaTensor) * learningRate)
      val momentumUpdate = momentumVariable.assign(updatedMu, name = "MomentumUpdate")
      val learningRateUpdate = learningRateVariable.assign(updatedLR, name = "LearningRateUpdate")
      Set(momentumUpdate.op, learningRateUpdate.op)
    }

    ControlFlow.group(updateOps)
  }

  protected def gradientsSparsity(gradients: Seq[OutputLike]): Option[Output] = {
    if (sparsityDebias) {
      Op.createWithNameScope("GradientsSparsity") {
        // If the sparse mini-batch gradient has 10 percent of its entries non-zero, its sparsity is 0.1. The norms of
        // dense gradients averaged over the full dataset are roughly estimated from the norms of mini-batch sparse
        // gradient norm * sqrt(sparsity). An extension may only correct the sparse blob.
        val nonZeroCount = Math.addN(gradients.map(Math.countNonZeroSparse(_))).cast(gradients.head.dataType)
        val totalCount = Math.addN(gradients.map(Basic.size(_))).cast(gradients.head.dataType)
        val sparsity = nonZeroCount / totalCount
        val sparsityAvgOp = movingAverage.computeForValues(Set(sparsity))
        val sparsityAvg = Op.createWith(controlDependencies = Set(sparsityAvgOp)) {
          movingAverage.average(sparsity).get.read()
        }
        Some(sparsityAvg)
      }
    } else {
      None
    }
  }

  protected def curvatureRange(gradNormSquaredSum: Output, sparsityAvg: Option[Output]): (Output, Output) = {
    Op.createWithNameScope("CurvatureRange") {
      val windowWidthTensor = Basic.constant(curvatureWindowWidth)
      // We use log-smoothing for the curvature range.
      val updatedCurvatureWindow = curvatureWindow.assignScatter(
        step.value % windowWidthTensor, Math.log(gradNormSquaredSum))
      // Note here that the steps start from 0.
      val validWindow = Basic.slice(
        updatedCurvatureWindow, Tensor(0), Math.minimum(windowWidthTensor, step.value + 1)(NewAxis))
      val hMinT = Math.min(validWindow)
      val hMaxT = Math.max(validWindow)
      Op.createWith(controlDependencies = Set(hMinT.op, hMaxT.op)) {
        val avgOp = movingAverage.computeForValues(Set(hMinT, hMaxT))
        Op.createWith(controlDependencies = Set(avgOp)) {
          var hMin = Math.exp(movingAverage.average(hMinT).get.read())
          var hMax = Math.exp(movingAverage.average(hMaxT).get.read())
          sparsityAvg.foreach(sAvg => {
            hMin *= sAvg
            hMax *= sAvg
          })
          (hMin, hMax)
        }
      }
    }
  }

  protected def gradientsVariance(
      gradients: Seq[OutputLike],
      gradNormSquaredAvg: Output,
      sparsityAvg: Option[Output]
  ): Output = {
    Op.createWithNameScope("GradientsVariance") {
      val grads = gradients.map(_.toOutput)
      val gradAvgOp = movingAverage.computeForValues(grads.toSet)
      val gradAvgSquared = Op.createWith(controlDependencies = Set(gradAvgOp)) {
        grads.map(g => Math.square(movingAverage.average(g).get.read()))
      }
      var gradVar = Math.maximum(
        Basic.constant(1e-6f, gradNormSquaredAvg.dataType),
        gradNormSquaredAvg - Math.addN(gradAvgSquared.map(Math.sum(_))))
      sparsityAvg.foreach(gradVar *= _)
      gradVar
    }
  }

  protected def distanceToOptimum(
      gradNormSquaredSum: Output,
      gradNormSquaredAvg: Output,
      sparsityAvg: Option[Output]
  ): Output = {
    val gradNorm = Math.sqrt(gradNormSquaredSum)
    val gradNormAvgOp = movingAverage.computeForValues(Set(gradNorm))
    val dist = Op.createWith(controlDependencies = Set(gradNormAvgOp)) {
        movingAverage.average(gradNorm).get.read() / gradNormSquaredAvg
    }
    val distAvgOp = movingAverage.computeForValues(Set(dist))
    val distAvg = Op.createWith(controlDependencies = Set(distAvgOp)) {
      var distAvg = movingAverage.average(dist).get.read()
      sparsityAvg.foreach(sAvg => distAvg /= Math.sqrt(sAvg))
      distAvg
    }
    distAvg
  }
}

object YellowFin {
  def apply(
      learningRate: Double = 1.0,
      decay: Schedule = FixedSchedule,
      momentum: Double = 0.0,
      beta: Float = 0.999f,
      curvatureWindowWidth: Int = 20,
      zeroDebias: Boolean = true,
      sparsityDebias: Boolean = true,
      useNesterov: Boolean = false,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "YellowFin"
  ): YellowFin = {
    new YellowFin(
      learningRate, decay, momentum, beta, curvatureWindowWidth, zeroDebias, sparsityDebias, useNesterov,
      useLocking, learningRateSummaryTag, name)
  }
}
