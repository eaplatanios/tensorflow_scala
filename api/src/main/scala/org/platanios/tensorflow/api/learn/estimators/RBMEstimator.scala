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

package org.platanios.tensorflow.api.learn.estimators

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.{Configuration, UnsupervisedEstimator}
import org.platanios.tensorflow.api.learn.layers.Input
import org.platanios.tensorflow.api.learn.models.RBM
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

case class RBMEstimator(
    input: Input[Tensor, Output, DataType, Shape],
    numHidden: Int,
    meanField: Boolean = true,
    numSamples: Int = 100,
    meanFieldCD: Boolean = false,
    cdSteps: Int = 1,
    optimizer: Optimizer,
    name: String = "RBM",
    private val configurationBase: Configuration = null
) extends UnsupervisedEstimator[Tensor, Output, DataType, Shape, Output](
  RBM(input, numHidden, meanField, numSamples, meanFieldCD, cdSteps, optimizer, name),
  configurationBase
)
