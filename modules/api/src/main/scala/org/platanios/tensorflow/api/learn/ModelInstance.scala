/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.types.{IsFloatOrDouble, TF}
import org.platanios.tensorflow.api.ops.{Output, OutputLike, UntypedOp}
import org.platanios.tensorflow.api.ops.data.DatasetIterator
import org.platanios.tensorflow.api.ops.variables.Variable

// TODO: [LEARN] What about "trainOutput"?

/** Represents an instance of a constructed model. Such instances are constructed by estimators and passed on to
  * model-dependent hooks.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ModelInstance[In, TrainIn, Out, TrainOut, Loss: TF : IsFloatOrDouble, EvalIn](
    model: TrainableModel[In, TrainIn, Out, TrainOut, Loss, EvalIn],
    configuration: Configuration,
    trainInputIterator: Option[DatasetIterator[TrainIn]] = None,
    trainInput: Option[TrainIn] = None,
    output: Option[Out] = None,
    trainOutput: Option[TrainOut] = None,
    loss: Option[Output[Loss]] = None,
    gradientsAndVariables: Option[Seq[(OutputLike[Loss], Variable[Any])]] = None,
    trainOp: Option[UntypedOp] = None)
