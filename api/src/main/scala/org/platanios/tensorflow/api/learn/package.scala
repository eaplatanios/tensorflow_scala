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

package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.core.exception.{AbortedException, UnavailableException}

/**
  * @author Emmanouil Antonios Platanios
  */
package object learn {
  private[learn] val RECOVERABLE_EXCEPTIONS: Set[Class[_]] = {
    Set(classOf[AbortedException], classOf[UnavailableException])
  }

  private[api] trait API
      extends estimators.API
          with hooks.API
          with layers.API {
    type Configuration = learn.Configuration
    type StopCriteria = learn.StopCriteria

    val Configuration: learn.Configuration.type = learn.Configuration
    val StopCriteria : learn.StopCriteria.type  = learn.StopCriteria

    type Mode = learn.Mode

    val TRAINING  : learn.TRAINING.type   = learn.TRAINING
    val EVALUATION: learn.EVALUATION.type = learn.EVALUATION
    val INFERENCE : learn.INFERENCE.type  = learn.INFERENCE

    val TensorBoardConfig: config.TensorBoardConfig.type = config.TensorBoardConfig

    type ClipGradients = learn.ClipGradients
    type ClipGradientsByValue = learn.ClipGradientsByValue
    type ClipGradientsByNorm = learn.ClipGradientsByNorm
    type ClipGradientsByAverageNorm = learn.ClipGradientsByAverageNorm
    type ClipGradientsByGlobalNorm = learn.ClipGradientsByGlobalNorm

    val NoClipGradients           : learn.NoClipGradients.type            = learn.NoClipGradients
    val ClipGradientsByValue      : learn.ClipGradientsByValue.type       = learn.ClipGradientsByValue
    val ClipGradientsByNorm       : learn.ClipGradientsByNorm.type        = learn.ClipGradientsByNorm
    val ClipGradientsByAverageNorm: learn.ClipGradientsByAverageNorm.type = learn.ClipGradientsByAverageNorm
    val ClipGradientsByGlobalNorm : learn.ClipGradientsByGlobalNorm.type  = learn.ClipGradientsByGlobalNorm

    type Model = learn.Model
    type InferenceModel[IT, IO, ID, IS, I] = learn.InferenceModel[IT, IO, ID, IS, I]
    type TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = learn.TrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T] = learn.SupervisedTrainableModel[IT, IO, ID, IS, I, TT, TO, TD, TS, T]
    type UnsupervisedTrainableModel[IT, IO, ID, IS, I] = learn.UnsupervisedTrainableModel[IT, IO, ID, IS, I]

    val Model: learn.Model.type = learn.Model
  }

  private[api] object API extends API
}
