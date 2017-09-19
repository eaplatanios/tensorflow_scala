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

// TODO: [LOOKUP] [OPS] Add support for lookup ops.

/**
  * @author Emmanouil Antonios Platanios
  */
object Table {
  /** Returns the set of all table initializers that have been created in the current graph. */
  def initializers: Set[Op] = Op.currentGraph.tableInitializers

  /** Creates an op that groups the provided table initializers.
    *
    * @param  initializers Table initializers to group.
    * @param  name         Name for the created op.
    * @return Created op.
    */
  def initializer(initializers: Set[Op], name: String = "TablesInitializer"): Op = {
    if (initializers.isEmpty)
      ControlFlow.noOp(name)
    else
      ControlFlow.group(initializers, name)
  }
}
