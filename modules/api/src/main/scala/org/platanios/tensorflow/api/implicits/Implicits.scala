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

import org.platanios.tensorflow.api.{core, learn, ops, tensors}
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.variables.Variable
import org.platanios.tensorflow.api.tensors.Tensor

/** Groups together all the implicits of the API and takes care of their priorities.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends LowPriorityImplicits
        with core.Implicits
        with tensors.Implicits {
  /** Convenient implicit conversion function used to convert devices specified as strings for use with the
    * [[Op.createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
    * return a device specification string.
    *
    * @param  device Device specification string.
    * @return Function that returns `device` for any [[OpSpecification]] used as input.
    */
  implicit def deviceImplicitConversion(device: String): OpSpecification => String = {
    _ => device
  }
}

private[api] trait LowPriorityImplicits
    extends ops.Implicits
        with learn.Implicits {
  implicit def tensorAsUntyped[T](tensor: Tensor[T]): Tensor[Any] = {
    tensor.asInstanceOf[Tensor[Any]]
  }

  implicit def opAsUntyped[I, O](op: Op[I, O]): UntypedOp = {
    op.asInstanceOf[UntypedOp]
  }

  implicit def opUntypedOutputAsOutputLike[I, O](
      op: Op[Seq[Output[Any]], Seq[Output[Any]]]
  ): Op[Seq[OutputLike[Any]], Seq[OutputLike[Any]]] = {
    op.asInstanceOf[Op[Seq[OutputLike[Any]], Seq[OutputLike[Any]]]]
  }

  implicit def outputAsUntyped[T](output: Output[T]): Output[Any] = {
    output.asInstanceOf[Output[Any]]
  }

  implicit def outputLikeAsUntyped[T](outputLike: OutputLike[T]): OutputLike[Any] = {
    outputLike.asInstanceOf[OutputLike[Any]]
  }

  implicit def tensorArrayAsUntyped[T](tensorArray: TensorArray[T]): TensorArray[Any] = {
    tensorArray.asInstanceOf[TensorArray[Any]]
  }

  implicit def variableAsUntyped[T](variable: Variable[T]): Variable[Any] = {
    variable.asInstanceOf[Variable[Any]]
  }

  implicit def opSetAsUntyped[I, O](ops: Set[Op[I, O]]): Set[UntypedOp] = {
    ops.asInstanceOf[Set[UntypedOp]]
  }

  implicit def outputSeqAsUntyped[T](outputs: Seq[Output[T]]): Seq[Output[Any]] = {
    outputs.asInstanceOf[Seq[Output[Any]]]
  }
}

private[api] object Implicits extends Implicits
