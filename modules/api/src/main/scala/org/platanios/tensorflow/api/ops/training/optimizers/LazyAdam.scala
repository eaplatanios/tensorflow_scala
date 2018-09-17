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
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, OutputIndexedSlices}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.optimizers.schedules.{FixedSchedule, Schedule}
import org.platanios.tensorflow.api.ops.variables.Variable

/** Optimizer that implements a variant of the Adam optimization algorithm that handles sparse updates more efficiently.
  *
  * The original Adam algorithm maintains two moving-average accumulators for each trainable variable; the accumulators
  * are updated at every step. This class provides lazier handling of gradient updates for sparse variables. It only
  * updates the moving-average accumulators for sparse variable indices that appear in the current batch, rather than
  * updating the accumulators for all indices. Compared with the original Adam optimizer, it can provide large
  * improvements in model training throughput for some applications. However, it provides slightly different semantics
  * than the original Adam algorithm, and may lead to different empirical results.
  *
  * Initialization:
  * {{{
  *   m_0 = 0  // Initialize the 1st moment vector
  *   v_0 = 0  // Initialize the 2nd moment vector
  *   t = 0    // Initialize the time step
  * }}}
  *
  * The Adam update for step `t` is as follows:
  * {{{
  *   learningRate_t = initialLearningRate * sqrt(beta1 - beta2^t) / (1 - beta1^t)
  *   m_t = beta1 * m_{t-1} + (1 - beta1) * gradient
  *   v_t = beta2 * v_{t-1} + (1 - beta2) * gradient * gradient
  *   variable -= learningRate_t * m_t / (sqrt(v_t) + epsilon)
  * }}}
  *
  * The default value of `1e-8` for epsilon might not be a good default in general. For example, when training an
  * Inception network on ImageNet a current good choice is `1.0` or `0.1`. Note that since the Adam optimizer uses the
  * formulation just before Section 2.1 of the [Kingma and Ba paper](https://arxiv.org/abs/1412.6980) rather than the
  * formulation in Algorithm 1, the "epsilon" referred to here is "epsilon hat" in the paper.
  *
  * The sparse implementation of this algorithm (used when the gradient is an indexed slices object, typically because
  * of `tf.gather` or an embedding lookup in the forward pass) does not apply momentum to variable slices if they are
  * not used in the forward pass (meaning they have a gradient equal to zero). Momentum decay (`beta1`) is also not
  * applied to the entire momentum accumulator. This means that the sparse behavior is not equivalent to the dense
  * behavior.
  *
  * For more information on the original Adam algorithm, please refer to this [paper](https://arxiv.org/abs/1412.6980)
  * ([PDF](https://arxiv.org/pdf/1412.6980.pdf)).
  *
  * The original Adam algorithm (described above) maintains two moving-average accumulators for each trainable variable,
  * which are updated at every step. This class provides a lazier handling of gradient updates for sparse variables. It
  * only updates moving-average accumulators for sparse variable indices that appear in the current batch, rather than
  * updating the accumulators for all indices. Compared with the original Adam optimizer, it can provide large
  * improvements in model training throughput for some applications. However, it provides slightly different semantics
  * than the original Adam algorithm, and may lead to different empirical results.
  *
  * @param  learningRate           Learning rate. Must be `> 0`. If used with `decay`, then this argument
  *                                specifies the initial value of the learning rate.
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
  *                                is created for the learning rate. Otherwise, a scalar summary is created which can
  *                                be monitored using TensorBoard.
  * @param  name                   Name for this optimizer.
  *
  * @author Emmanouil Antonios Platanios
  */
class LazyAdam protected (
    override val learningRate: Float = 0.001f,
    override val decay: Schedule = FixedSchedule,
    override val beta1: Float = 0.9f,
    override val beta2: Float = 0.999f,
    override val useNesterov: Boolean = false,
    override val epsilon: Float = 1e-8f,
    override val useLocking: Boolean = false,
    override val learningRateSummaryTag: String = null,
    override val name: String = "LazyAdam"
) extends Adam(
  learningRate, decay, beta1, beta2, useNesterov, epsilon, useLocking, learningRateSummaryTag, name
) {
  override val ignoreDuplicateSparseIndices: Boolean = true

  override def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op = {
    val m = getSlot("M", variable)
    val v = getSlot("V", variable)
    val (beta1Power, beta2Power) = getBetaPowerAccumulators
    val beta1 = getBeta1(variable)
    val beta2 = getBeta2(variable)
    val epsilon = getEpsilon(variable)
    var learningRate = getLearningRate(variable, iteration)
    learningRate = learningRate * Math.sqrt(1 - beta2Power.cast(variable.dataType))
    learningRate = learningRate / (1 - beta1Power.cast(variable.dataType))

    // m_t = beta1 * m + (1 - beta1) * gradient
    val mT = m.assignScatter(
      gradient.indices, beta1 * Basic.gather(m.value, gradient.indices) + (1 - beta1) * gradient.values)

    // v_t = beta2 * v + (1 - beta2) * gradient * gradient
    val vT = v.assignScatter(
      gradient.indices, beta2 * Basic.gather(v.value, gradient.indices) + (1 - beta2) * Math.square(gradient.values))

    // variable -= learning_rate * m_t / (epsilon_t + sqrt(v_t))
    val mTSlice = Basic.gather(mT, gradient.indices)
    val vTSlice = Basic.gather(vT, gradient.indices)
    val denominatorSlice = Math.sqrt(vTSlice) + epsilon
    val update = variable.assignScatterSub(gradient.indices, learningRate * mTSlice / denominatorSlice)

    ControlFlow.group(Set(update.op, mT.op, vT.op))
  }
}

object LazyAdam {
  def apply(
      learningRate: Float = 0.001f,
      decay: Schedule = FixedSchedule,
      beta1: Float = 0.9f,
      beta2: Float = 0.999f,
      useNesterov: Boolean = false,
      epsilon: Float = 1e-8f,
      useLocking: Boolean = false,
      learningRateSummaryTag: String = null,
      name: String = "LazyAdam"
  ): LazyAdam = {
    new LazyAdam(learningRate, decay, beta1, beta2, useNesterov, epsilon, useLocking, learningRateSummaryTag, name)
  }
}
