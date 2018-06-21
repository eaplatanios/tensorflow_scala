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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.ops._

/** Groups together all the implicits of the API and takes care of their priorities.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends LowPriorityImplicits
        with IndexerImplicits {
  /** Convenient implicit conversion function used to convert devices specified as [[String]]s for use with the
    * [[Op.createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
    * return a device specification string.
    *
    * @param  device Device specification string.
    * @return Function that returns `device` for any [[OpSpecification]] used as input.
    */
  implicit def deviceImplicitConversion(device: String): OpSpecification => String = _ => device
}

private[api] trait LowPriorityImplicits
    extends LowestPriorityImplicits
        with TensorImplicits
        with DataImplicits
        with LearnImplicits

private[api] trait LowestPriorityImplicits
    extends OpsImplicits

private[api] object Implicits extends Implicits
