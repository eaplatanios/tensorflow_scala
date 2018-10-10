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

package org.platanios.tensorflow.api.learn.hooks

import org.platanios.tensorflow.api.learn.ModelInstance

/** Represents hooks that may dependent on the constructed model.
  *
  * This class offers the `modelInstance` field that sub-classes can access and that contains information specific to
  * the created model. It is only updated when the model graph is constructed (i.e., it is not updated while recovering
  * failed sessions).
  *
  * For example, a hook that logs the loss function value depends on the created loss op, or an evaluation hook may
  * depends on multiple ops created as part of the model.
  *
  * @author Emmanouil Antonios Platanios
  */
trait ModelDependentHook[In, Out, Loss, InEval] extends Hook {
  protected var modelInstance: ModelInstance[In, Out, Loss, InEval] = _

  /** This method will be called by estimators at graph construction time, before `begin()`. It will **not** be called
    * again if a session fails and is recovered. */
  private[learn] final def setModelInstance(
      modelInstance: ModelInstance[In, Out, Loss, InEval]
  ): Unit = {
    this.modelInstance = modelInstance
  }
}
