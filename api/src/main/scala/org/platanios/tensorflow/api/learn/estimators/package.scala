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

package org.platanios.tensorflow.api.learn

/**
  * @author Emmanouil Antonios Platanios
  */
package object estimators {
  private[api] trait API {
    type Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = estimators.Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type InMemoryEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = estimators.InMemoryEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type FileBasedEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = estimators.FileBasedEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]

    val Estimator         : estimators.Estimator.type          = estimators.Estimator
    val InMemoryEstimator : estimators.InMemoryEstimator.type  = estimators.InMemoryEstimator
    val FileBasedEstimator: estimators.FileBasedEstimator.type = estimators.FileBasedEstimator
  }

  private[api] object API extends API
}
