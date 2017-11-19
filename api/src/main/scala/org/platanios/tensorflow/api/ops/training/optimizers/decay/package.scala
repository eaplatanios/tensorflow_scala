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

package org.platanios.tensorflow.api.ops.training.optimizers

/**
  * @author Emmanouil Antonios Platanios
  */
package object decay {
  private[optimizers] trait API {
    type Decay = decay.Decay
    type ExponentialDecay = decay.ExponentialDecay
    type LuongExponentialDecay = decay.LuongExponentialDecay
    type WarmUpDecay = decay.WarmUpDecay

    val NoDecay              : decay.NoDecay.type               = decay.NoDecay
    val ExponentialDecay     : decay.ExponentialDecay.type      = decay.ExponentialDecay
    val LuongExponentialDecay: decay.LuongExponentialDecay.type = decay.LuongExponentialDecay
    val WarmUpDecay          : decay.WarmUpDecay.type           = decay.WarmUpDecay
  }
}
