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

package org.platanios.tensorflow.api

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object ops {
  private[ops] val DEFAULT_GRAPH_RANDOM_SEED = 87654321

  private[ops] val COLOCATION_OPS_ATTRIBUTE_NAME  : String = "_class"
  private[ops] val COLOCATION_OPS_ATTRIBUTE_PREFIX: String = "loc:@"
  private[ops] val VALID_OP_NAME_REGEX            : Regex  = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[ops] val VALID_NAME_SCOPE_REGEX         : Regex  = "^[A-Za-z0-9_.\\-/]*$".r

  private[ops] val META_GRAPH_UNBOUND_INPUT_PREFIX: String = "$unbound_inputs_"

  private[api] trait API {
    implicit val opCreationContext: DynamicVariable[OpCreationContext] = {
      new DynamicVariable[OpCreationContext](OpCreationContext(graph = defaultGraph))
    }

    implicit def dynamicVariableToOpCreationContext(context: DynamicVariable[OpCreationContext]): OpCreationContext = {
      context.value
    }

    /** Convenient implicit conversion function used to convert devices specified as [[String]]s for use with the
      * [[tf.createWith]] function, to the expected device function format taking an [[OpSpecification]] as input and
      * return a device specification string.
      *
      * @param  device Device specification string.
      * @return Function that returns `device` for any [[OpSpecification]] used as input.
      */
    implicit def deviceImplicitConversion(device: String): (OpSpecification => String) = _ => device

    implicit def outputToInitialValueFunction(output: Output): () => Output = () => output
  }

  private[api] trait ScopedAPI
      extends Basic
          with ControlFlow
          with DataFlow
          with Image
          with Logging
          with Math
          with NN
          with Random
          with Sparse
          with Statistics
          with Text
          with Gradients.API
          with Op.API
          with Output.API
          with Queue.API
          with io.API
          with variables.API {
    ops.Basic.Gradients
    ops.DataFlow.Gradients
    ops.Logging.Gradients
    ops.Math.Gradients
    ops.NN.Gradients
    ops.Queue.Gradients
    ops.Random.Gradients

    object train extends training.API
  }
}
