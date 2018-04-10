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

import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, OutputIndexedSlices, Summary}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.Variable

/** Optimizer that implements the AMSGrad optimization algorithm, presented in
  * [On the Convergence of Adam and Beyond](https://openreview.net/pdf?id=ryQu7f-RZ).
  *
  * Initialization:
  * {{{
  *   m_0 = 0     // Initialize the 1st moment vector
  *   v_0 = 0     // Initialize the 2nd moment vector
  *   v_hat_0 = 0 // Initialize the 2nd moment max vector
  *   t = 0       // Initialize the time step
  * }}}
  *
  * The Adam update for step `t` is as follows:
  * {{{
  *   learningRate_t = initialLearningRate * sqrt(beta1 - beta2^t) / (1 - beta1^t)
  *   m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
  *   v_t = beta2 * v_{t-1} + (1 - beta2) * gradient * gradient
  *   v_hat_t = max(v_t, v_hat_{t-1})
  *   variable -= learningRate_t * m_t / (sqrt(v_hat_t) + epsilon)
  * }}}
  *
  * The default value of `1e-8` for epsilon might not be a good default in general. For example, when training an
  * Inception network on ImageNet a current good choice is `1.0` or `0.1`.
  *
  * The sparse implementation of this algorithm (used when the gradient is an indexed slices object, typically because
  * of `tf.gather` or an embedding lookup in the forward pass) does apply momentum to variable slices even if they were
  * not used in the forward pass (meaning they have a gradient equal to zero). Momentum decay (`beta1`) is also applied
  * to the entire momentum accumulator. This means that the sparse behavior is equivalent to the dense behavior (in
  * contrast to some momentum implementations which ignore momentum unless a variable slice was actually used).
  *
  * For more information on this algorithm, please refer to this [paper](https://openreview.net/pdf?id=ryQu7f-RZ).
  *
  * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument specifies the
  *                                initial value of the learning rate.
  * @param  decay                  Learning rate decay method to use for each update.
  * @param  beta1                  Exponential decay rate for the first moment estimates.
  * @param  beta2                  Exponential decay rate for the second moment estimates.
  * @param  useNesterov            If `true`, Nesterov momentum is used for the updates.
  * @param  epsilon                Small constant used for numerical stability. This epsilon corresponds to
  *                                "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1),
  *                                and not to the epsilon in Algorithm 1 of the paper.
  * @param  useLocking             If `true`, the gradient descent updates will be protected by a lock. Otherwise, the
  *                                behavior is undefined, but may exhibit less contention.
  * @param  learningRateSummaryTag Optional summary tag name to use for the learning rate value. If `null`, no summary
  *                                is created for the learning rate. Otherwise, a scalar summary is created which can be
  *                                monitored using TensorBoard.
  * @param  name                   Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class AMSGrad protected (
    val learningRate: Double = 0.001,
    val decay: Schedule = FixedSchedule,
    val beta1: Double = 0.9,
    val beta2: Double = 0.999,
    val useNesterov: Boolean = false,
    val epsilon: Double = 1e-8,
    val useLocking: Boolean = false,
    val learningRateSummaryTag: String = null,
    val name: String = "Adam"
) extends Optimizer {
  protected var learningRateTensor: Output = _
  protected var beta1Tensor       : Output = _
  protected var beta2Tensor       : Output = _
  protected var epsilonTensor     : Output = _

  protected def getLearningRate(variable: Variable, iteration: Option[Variable]): Output = {
    if (learningRateTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(learningRateTensor, variable.dataType)
  }

  protected def getBeta1(variable: Variable): Output = {
    if (beta1Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(beta1Tensor, variable.dataType)
  }

  protected def getBeta2(variable: Variable): Output = {
    if (beta2Tensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(beta2Tensor, variable.dataType)
  }

  protected def getEpsilon(variable: Variable): Output = {
    if (epsilonTensor == null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(epsilonTensor, variable.dataType)
  }

  protected def getBetaPowerAccumulators: (Variable, Variable) = {
    (getNonSlotVariable("Beta1Power", Op.currentGraph), getNonSlotVariable("Beta2Power", Op.currentGraph))
  }

  override protected def createSlots(variables: Seq[Variable]): Unit = {
    // Create slots for the first and second moments.
    variables.foreach(v => {
      zerosSlot("M", v, name)
      zerosSlot("V", v, name)
      zerosSlot("Vhat", v, name)
    })
    // We create the 'beta1' and 'beta2' accumulators on the same device as the first variable. We sort the variables
    // list to make sure this device is consistent across workers (these need to go on the same parameter server,
    // otherwise some updates are silently ignored).
    val firstVariable = variables.minBy(_.name)
    getOrCreateNonSlotVariable("Beta1Power", beta1, Set(firstVariable.op))
    getOrCreateNonSlotVariable("Beta2Power", beta2, Set(firstVariable.op))
  }

  override def prepare(iteration: Option[Variable]): Unit = {
    learningRateTensor = decay(Basic.constant(learningRate, name = "LearningRate"), iteration)
    if (learningRateSummaryTag != null)
      Summary.scalar(learningRateSummaryTag, learningRateTensor)
    beta1Tensor = Basic.constant(beta1, name = "Beta1")
    beta2Tensor = Basic.constant(beta2, name = "Beta2")
    epsilonTensor = Basic.constant(epsilon, name = "Epsilon")
  }

  override def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op = {
    val m = getSlot("M", variable)
    val v = getSlot("V", variable)
    val vHat = getSlot("Vhat", variable)
    val (beta1Power, beta2Power) = getBetaPowerAccumulators
    val beta1 = getBeta1(variable)
    val beta2 = getBeta2(variable)
    val epsilon = getEpsilon(variable)
    var learningRate = getLearningRate(variable, iteration)
    learningRate = learningRate * Math.sqrt(1 - beta2Power.cast(variable.dataType))
    learningRate = learningRate / (1 - beta1Power.cast(variable.dataType))

    // m_t = beta1 * m + (1 - beta1) * gradient
    val mScaledGradient = gradient * (1 - beta1)
    val mT = m.assign((m.value * beta1) + mScaledGradient)

    // v_t = beta2 * v + (1 - beta2) * gradient * gradient
    val vScaledGradient = gradient * gradient * (1 - beta2)
    val vT = v.assign((v.value * beta2) + vScaledGradient)

    val vHatT = vHat.assign(Math.maximum(vT, vHat))
    val vHatTSqrt = Math.sqrt(vHatT)
    val update = variable.assignSub(learningRate * mT / Math.add(vHatTSqrt, epsilon))
    ControlFlow.group(Set(update.op, mT.op, vT.op))
  }

  override protected def finish(updateOps: Set[Op], nameScope: String): Op = {
    // Update the power accumulators.
    val (beta1Power, beta2Power) = getBetaPowerAccumulators
    val updateBetaPowerOps = Op.createWith(controlDependencies = updateOps) {
      Op.colocateWith(Set(beta1Power.op)) {
        val updateBeta1Power = beta1Power.assign(beta1Power.value * beta1Tensor)
        val updateBeta2Power = beta2Power.assign(beta2Power.value * beta2Tensor)
        Set(updateBeta1Power.op, updateBeta2Power.op)
      }
    }
    ControlFlow.group(updateOps ++ updateBetaPowerOps, nameScope)
  }

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    val m = getSlot("M", variable)
    val v = getSlot("V", variable)
    val vHat = getSlot("Vhat", variable)
    val (beta1Power, beta2Power) = getBetaPowerAccumulators
    val beta1 = getBeta1(variable)
    val beta2 = getBeta2(variable)
    val epsilon = getEpsilon(variable)
    var learningRate = getLearningRate(variable, iteration)
    learningRate = learningRate * Math.sqrt(1 - beta2Power.cast(variable.dataType))
    learningRate = learningRate / (1 - beta1Power.cast(variable.dataType))

    // m_t = beta1 * m + (1 - beta1) * gradient
    val mScaledGradient = gradient.values * (1 - beta1)
    var mT = m.assign(m.value * beta1)
    mT = Op.createWith(controlDependencies = Set(mT.op)) {
      m.assignScatterAdd(gradient.indices, mScaledGradient)
    }

    // v_t = beta2 * v + (1 - beta2) * gradient * gradient
    val vScaledGradient = gradient.values * gradient.values * (1 - beta2)
    var vT = v.assign(v.value * beta2)
    vT = Op.createWith(controlDependencies = Set(vT.op)) {
      v.assignScatterAdd(gradient.indices, vScaledGradient)
    }

    val vHatT = vHat.assign(Math.maximum(vT, vHat))
    val vHatTSqrt = Math.sqrt(vHatT)
    val update = variable.assignSub(learningRate * mT / Math.add(vHatTSqrt, epsilon))
    ControlFlow.group(Set(update.op, mT.op, vT.op))
  }
}

object AMSGrad {
  def apply(
      learningRate: Double = 0.001,
      decay: Schedule = FixedSchedule,
      beta1: Double = 0.9,
      beta2: Double = 0.999,
      useNesterov: Boolean = false,
      epsilon: Double = 1e-8,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "Adam"
  ): AMSGrad = {
    new AMSGrad(learningRate, decay, beta1, beta2, useNesterov, epsilon, useLocking, learningRateSummaryTag, name)
  }
}
