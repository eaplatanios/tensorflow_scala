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

package org.platanios.tensorflow.api.ops.training.distribute

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.ops.{Op, OutputLike}

/**
  * @author Emmanouil Antonios Platanios
  */
trait Distributable[T] {
  def op(value: T): Op
  def device(value: T): DeviceSpecification
}

object Distributable {
  implicit def outputLikeDistributable[O](implicit ev: O => OutputLike): Distributable[O] = new Distributable[O] {
    override def op(value: O): Op = ev(value).op
    override def device(value: O): DeviceSpecification = DeviceSpecification.fromString(ev(value).device)
  }

  implicit val variableDistributable: Distributable[Variable] = new Distributable[Variable] {
    override def op(value: Variable): Op = value.op
    override def device(value: Variable): DeviceSpecification = DeviceSpecification.fromString(value.device)
  }
}
