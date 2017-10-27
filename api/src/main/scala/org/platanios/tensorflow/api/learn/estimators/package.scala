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

import org.platanios.tensorflow.api.config.TensorBoardConfig
import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.hooks.Hook
import org.platanios.tensorflow.api.learn.layers.Input
import org.platanios.tensorflow.api.learn.models.RBM
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.metrics.Metric
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
package object estimators {
  private[api] trait API {
    type Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = estimators.Estimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type InMemoryEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = estimators.InMemoryEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]
    type FileBasedEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI] = estimators.FileBasedEstimator[IT, IO, ID, IS, I, TT, TO, TD, TS, EI]

    val Estimator : estimators.Estimator.type  = estimators.Estimator
    val InMemoryEstimator : estimators.InMemoryEstimator.type  = estimators.InMemoryEstimator
    val FileBasedEstimator: estimators.FileBasedEstimator.type = estimators.FileBasedEstimator

    def RBMEstimator(
        input: Input[Tensor, Output, DataType, Shape],
        numHidden: Int,
        meanField: Boolean = true,
        numSamples: Int = 100,
        meanFieldCD: Boolean = false,
        cdSteps: Int = 1,
        optimizer: Optimizer,
        name: String = "RBM",
        configurationBase: Configuration = null,
        stopCriteria: StopCriteria = StopCriteria(),
        hooks: Seq[Hook] = Seq.empty,
        chiefOnlyHooks: Seq[Hook] = Seq.empty,
        tensorBoardConfig: TensorBoardConfig = null,
        evaluationMetrics: Seq[Metric[Output, Output]] = Seq.empty,
        inMemory: Boolean = false
    ): Estimator[Tensor, Output, DataType, Shape, Output, Tensor, Output, DataType, Shape, Output] = {
      if (inMemory)
        InMemoryEstimator[Tensor, Output, DataType, Shape, Output, Tensor, Output, DataType, Shape, Output](
          RBM(input, numHidden, meanField, numSamples, meanFieldCD, cdSteps, optimizer, name),
          configurationBase, stopCriteria, hooks, chiefOnlyHooks, tensorBoardConfig, evaluationMetrics)
      else
        FileBasedEstimator[Tensor, Output, DataType, Shape, Output, Tensor, Output, DataType, Shape, Output](
          RBM(input, numHidden, meanField, numSamples, meanFieldCD, cdSteps, optimizer, name),
          configurationBase, stopCriteria, hooks, chiefOnlyHooks, tensorBoardConfig, evaluationMetrics)
    }
  }

  private[api] object API extends API
}
