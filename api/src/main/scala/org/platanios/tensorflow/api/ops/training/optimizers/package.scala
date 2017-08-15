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

package org.platanios.tensorflow.api.ops.training

/**
  * @author Emmanouil Antonios Platanios
  */
package object optimizers {
  private[training] trait API
      extends Decay.API {
    type Optimizer = optimizers.Optimizer
    type AdaDelta = optimizers.AdaDelta
    type AdaGrad = optimizers.AdaGrad
    type GradientDescent = optimizers.GradientDescent

    def adaGrad(
        learningRate: Double = 0.01, decay: Decay = NoDecay, initialAccumulatorValue: Double = 1e-8,
        useLocking: Boolean = false, name: String = "AdaGradOptimizer"): AdaGrad = {
      AdaGrad(
        learningRate = learningRate, decay = decay, epsilon = initialAccumulatorValue, useLocking = useLocking,
        name = name)
    }

    def AdaDelta(
        learningRate: Double = 0.01, decay: Decay = NoDecay, rho: Double = 0.95, epsilon: Double = 1e-8,
        useLocking: Boolean = false, name: String = "AdaDeltaOptimizer"): AdaDelta = {
      AdaDelta(
        learningRate = learningRate, decay = decay, rho = rho, epsilon = epsilon, useLocking = useLocking, name = name)
    }

    def gradientDescent(
        learningRate: Double, decay: Decay = NoDecay, momentum: Double = 0.0, useNesterov: Boolean = false,
        useLocking: Boolean = false, name: String = "GradientDescentOptimizer"): GradientDescent = {
      GradientDescent(
        learningRate = learningRate, decay = decay, momentum = momentum, useNesterov = useNesterov,
        useLocking = useLocking, name = name)
    }
  }
}
