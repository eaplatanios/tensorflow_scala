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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.io.data
import org.platanios.tensorflow.api.ops.{Op, OpCreationContext, OpSpecification, Output}
import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.types.DataType

import scala.util.DynamicVariable

/** Groups together all the implicits of the API and takes care of their priorities.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends LowPriorityImplicits
        with Indexer {
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

  // implicit def dataTypeHelper[D >: DataType.Aux[_] <: DataType]: DataType
}

private[api] trait LowPriorityImplicits
    extends Tensor
        with Ops
        with Data
        with Learn {
  implicit val tensorDataHelper: data.Data.Aux[tensors.Tensor, Output, DataType, Shape] = data.Data.tensorData[DataType]
}

private[api] object Implicits extends Implicits
