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

package org.platanios.tensorflow.api.ops

import scala.util.DynamicVariable

/** Groups together all implicits related to constructing symbolic ops.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends Output.Implicits
        with Basic.Implicits
        with Clip.Implicits
        with Embedding.Implicits
        with Math.Implicits
        with NN.Implicits
        with Statistics.Implicits {
  implicit def dynamicVariableToOpCreationContext(context: DynamicVariable[OpCreationContext]): OpCreationContext = {
    context.value
  }

  /** Convenient implicit conversion function used to convert devices specified as [[String]]s for use with the
    * [[Op.createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
    * return a device specification string.
    *
    * @param  device Device specification string.
    * @return Function that returns `device` for any [[OpSpecification]] used as input.
    */
  implicit def deviceImplicitConversion(device: String): (OpSpecification => String) = _ => device
}
