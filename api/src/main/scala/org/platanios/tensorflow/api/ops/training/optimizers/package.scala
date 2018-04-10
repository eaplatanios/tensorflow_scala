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

package org.platanios.tensorflow.api.ops.training

/**
  * @author Emmanouil Antonios Platanios
  */
package object optimizers {
  private[api] trait API
      extends schedules.API {
    type Optimizer = optimizers.Optimizer
    type AdaDelta = optimizers.AdaDelta
    type AdaGrad = optimizers.AdaGrad
    type Adam = optimizers.Adam
    type AMSGrad = optimizers.AMSGrad
    type GradientDescent = optimizers.GradientDescent
    type LazyAdam = optimizers.LazyAdam
    type YellowFin = optimizers.YellowFin

    val AdaDelta       : optimizers.AdaDelta.type        = optimizers.AdaDelta
    val AdaGrad        : optimizers.AdaGrad.type         = optimizers.AdaGrad
    val Adam           : optimizers.Adam.type            = optimizers.Adam
    val AMSGrad        : optimizers.AMSGrad.type         = optimizers.AMSGrad
    val GradientDescent: optimizers.GradientDescent.type = optimizers.GradientDescent
    val LazyAdam       : optimizers.LazyAdam.type        = optimizers.LazyAdam
    val YellowFin      : optimizers.YellowFin.type       = optimizers.YellowFin
  }
}
