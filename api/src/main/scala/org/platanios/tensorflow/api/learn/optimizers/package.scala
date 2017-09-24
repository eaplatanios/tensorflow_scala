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

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.ops

/**
  * @author Emmanouil Antonios Platanios
  */
package object optimizers {
  private[api] trait API {
    val NoDecay         : ops.training.optimizers.NoDecay.type          = ops.training.optimizers.NoDecay
    val ExponentialDecay: ops.training.optimizers.ExponentialDecay.type = ops.training.optimizers.ExponentialDecay

    val GradientDescent: ops.training.optimizers.GradientDescent.type = ops.training.optimizers.GradientDescent
    val AdaGrad        : ops.training.optimizers.AdaGrad.type         = ops.training.optimizers.AdaGrad
    val AdaDelta       : ops.training.optimizers.AdaDelta.type        = ops.training.optimizers.AdaDelta
  }

  private[api] object API extends API
}
