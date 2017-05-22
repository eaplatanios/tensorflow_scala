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

package org.platanios.tensorflow

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends Implicits {
  private[api] val defaultGraph: core.Graph = core.Graph()

  private[api] val DEFAULT_TENSOR_MEMORY_STRUCTURE_ORDER = tensors.RowMajorOrder

  //region Op Creation

  private[api] val COLOCATION_OPS_ATTRIBUTE_NAME   = "_class"
  private[api] val COLOCATION_OPS_ATTRIBUTE_PREFIX = "loc:@"
  private[api] val VALID_OP_NAME_REGEX   : Regex   = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[api] val VALID_NAME_SCOPE_REGEX: Regex   = "^[A-Za-z0-9_.\\-/]*$".r

  private[api] val META_GRAPH_UNBOUND_INPUT_PREFIX: String = "$unbound_inputs_"

  import org.platanios.tensorflow.api.ops.OpCreationContext

  implicit val opCreationContext: DynamicVariable[OpCreationContext] = {
    new DynamicVariable[OpCreationContext](OpCreationContext(graph = defaultGraph))
  }

  implicit def dynamicVariableToOpCreationContext(context: DynamicVariable[OpCreationContext]): OpCreationContext = {
    context.value
  }

  //endregion Op Creation

  //region Utilities

  trait Closeable {
    def close(): Unit
  }

  def using[T <: Closeable, R](resource: T)(block: T => R): R = {
    try {
      block(resource)
    } finally {
      if (resource != null)
        resource.close()
    }
  }

  private[api] val Disposer = utilities.Disposer

  type ProtoSerializable = utilities.Proto.Serializable

  //endregion Utilities

  //region Public API

  object tf extends API

  //endregion Public API
}
