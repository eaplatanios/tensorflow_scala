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
    type CheckpointSaver = hooks.CheckpointSaver
    type Evaluator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = hooks.Evaluator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type LossLogger = hooks.LossLogger
    type ModelDependentHook[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = hooks.ModelDependentHook[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type NaNChecker = hooks.NaNChecker
    type StepRateLogger = hooks.StepRateLogger
    type Stopper = hooks.Stopper
    type SummarySaver = hooks.SummarySaver
    type SummaryWriterHookAddOn = hooks.SummaryWriterHookAddOn
    type TensorBoardHook = hooks.TensorBoardHook
    type TensorLogger = hooks.TensorLogger
    type TriggeredHook = hooks.TriggeredHook

    val CheckpointSaver: hooks.CheckpointSaver.type = hooks.CheckpointSaver
    val Evaluator      : hooks.Evaluator.type       = hooks.Evaluator
    val LossLogger     : hooks.LossLogger.type      = hooks.LossLogger
    val NaNChecker     : hooks.NaNChecker.type      = hooks.NaNChecker
    val StepRateLogger : hooks.StepRateLogger.type  = hooks.StepRateLogger
    val Stopper        : hooks.Stopper.type         = hooks.Stopper
    val SummarySaver   : hooks.SummarySaver.type    = hooks.SummarySaver
    val TensorBoardHook: hooks.TensorBoardHook.type = hooks.TensorBoardHook
    val TensorLogger   : hooks.TensorLogger.type    = hooks.TensorLogger
  }

  private[api] object API extends API
}
