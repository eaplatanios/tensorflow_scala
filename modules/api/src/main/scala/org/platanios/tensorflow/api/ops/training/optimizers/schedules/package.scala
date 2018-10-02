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

/**
  * @author Emmanouil Antonios Platanios
  */
package object schedules {
  private[optimizers] trait API {
    type Schedule[-T] = schedules.Schedule[T]
    type CosineDecay = schedules.CosineDecay
    type CycleLinear10xDecay = schedules.CycleLinear10xDecay
    type ExponentialDecay = schedules.ExponentialDecay
    type LuongExponentialDecay = schedules.LuongExponentialDecay
    type SqrtDecay = schedules.SqrtDecay
    type WarmUpExponentialSchedule = schedules.WarmUpExponentialSchedule
    type WarmUpLinearSchedule = schedules.WarmUpLinearSchedule

    val FixedSchedule            : schedules.FixedSchedule.type             = schedules.FixedSchedule
    val CosineDecay              : schedules.CosineDecay.type               = schedules.CosineDecay
    val CycleLinear10xDecay      : schedules.CycleLinear10xDecay.type       = schedules.CycleLinear10xDecay
    val ExponentialDecay         : schedules.ExponentialDecay.type          = schedules.ExponentialDecay
    val LuongExponentialDecay    : schedules.LuongExponentialDecay.type     = schedules.LuongExponentialDecay
    val SqrtDecay                : schedules.SqrtDecay.type                 = schedules.SqrtDecay
    val WarmUpExponentialSchedule: schedules.WarmUpExponentialSchedule.type = schedules.WarmUpExponentialSchedule
    val WarmUpLinearSchedule     : schedules.WarmUpLinearSchedule.type      = schedules.WarmUpLinearSchedule

    // TODO: Piecewise constant.
    // TODO: Polynomial.
    // TODO: Natural exp.
    // TODO: Inverse time.
  }
}
