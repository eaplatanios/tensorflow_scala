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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.variables.Variable

class LayerInstance[T, R] private[layers] (
    val input: T,
    val output: R,
    val trainableVariables: Set[Variable] = Set.empty[Variable],
    val nonTrainableVariables: Set[Variable] = Set.empty[Variable],
    val graph: Graph = Op.currentGraph)

object LayerInstance {
  def apply[T, R](
      input: T,
      output: R,
      trainableVariables: Set[Variable] = Set.empty[Variable],
      nonTrainableVariables: Set[Variable] = Set.empty[Variable],
      graph: Graph = Op.currentGraph): LayerInstance[T, R] = {
    new LayerInstance(input, output, trainableVariables, nonTrainableVariables, graph)
  }
}
