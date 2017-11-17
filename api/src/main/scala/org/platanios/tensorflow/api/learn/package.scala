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
      extends Model.API
          with estimators.API
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
  }

  private[api] object API extends API
}
