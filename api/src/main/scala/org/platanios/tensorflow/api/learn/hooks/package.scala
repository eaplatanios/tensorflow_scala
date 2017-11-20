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

/**
  * @author Emmanouil Antonios Platanios
  */
package object hooks {
  private[api] trait API
      extends HookTrigger.API {
    type Hook = hooks.Hook
    type ModelDependentHook[I, TT, TO, TD, TS] = hooks.ModelDependentHook[I, TT, TO, TD, TS]
    type LossLoggingHook = hooks.LossLoggingHook
    type CheckpointSaverHook = hooks.CheckpointSaverHook
    type StepRateHook = hooks.StepRateHook
    type StopEvaluationHook = hooks.StopEvaluationHook
    type StopHook = hooks.StopHook
    type SummarySaverHook = hooks.SummarySaverHook
    type TensorBoardHook = hooks.TensorBoardHook
    type TensorLoggingHook = hooks.TensorLoggingHook
    type TensorNaNHook = hooks.TensorNaNHook

    val LossLoggingHook    : hooks.LossLoggingHook.type     = hooks.LossLoggingHook
    val CheckpointSaverHook: hooks.CheckpointSaverHook.type = hooks.CheckpointSaverHook
    val StepRateHook       : hooks.StepRateHook.type        = hooks.StepRateHook
    val StopEvaluationHook : hooks.StopEvaluationHook.type  = hooks.StopEvaluationHook
    val StopHook           : hooks.StopHook.type            = hooks.StopHook
    val SummarySaverHook   : hooks.SummarySaverHook.type    = hooks.SummarySaverHook
    val TensorBoardHook    : hooks.TensorBoardHook.type     = hooks.TensorBoardHook
    val TensorLoggingHook  : hooks.TensorLoggingHook.type   = hooks.TensorLoggingHook
    val TensorNaNHook      : hooks.TensorNaNHook.type       = hooks.TensorNaNHook
  }

  private[api] object API extends API
}
