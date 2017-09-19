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

package org.platanios.tensorflow.api.learn

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.variables.{Variable, ZerosInitializer}
import org.platanios.tensorflow.api.types.INT64

/** Contains helper methods for creating and obtaining global step variables.
  *
  * @author Emmanouil Antonios Platanios
  */
object GlobalStep {
  /** Creates a global step variable.
    *
    * @param  graph Graph in which to create the global step variable. If missing, the current graph is used.
    * @return Created global step variable.
    * @throws IllegalStateException If a global step variable already exists in the `GLOBAL_STEP` graph collection.
    */
  @throws[IllegalStateException]
  def create(graph: Graph = Op.currentGraph): Variable = {
    get(graph) match {
      case Some(_) => throw new IllegalStateException("A global step variable already exists in this graph.")
      case None => Op.createWith(graph, nameScope = "") {
        Variable.getVariable(
          name = Graph.Keys.GLOBAL_STEP.name,
          dataType = INT64,
          shape = Shape.scalar(),
          initializer = ZerosInitializer,
          trainable = false,
          collections = Set(Graph.Keys.GLOBAL_VARIABLES, Graph.Keys.GLOBAL_STEP))
      }
    }
  }

  /** Gets the global step variable.
    *
    * The global step variable must be an integer variable and it should be included in the graph collection named
    * `GLOBAL_STEP`.
    *
    * @param  graph Graph to find the global step in. If missing, the current graph is used.
    * @return The global step variable, or `None`, if not found.
    * @throws IllegalStateException If more than one variables exist in the `GLOBAL_STEP` graph collection, or if the
    *                               obtained global step is not an integer scalar.
    */
  @throws[IllegalStateException]
  def get(graph: Graph = Op.currentGraph): Option[Variable] = {
    val globalStepVariables = graph.getCollection(Graph.Keys.GLOBAL_STEP)
    if (globalStepVariables.size > 1)
      throw new IllegalStateException("There should only exist one global step variable in a graph.")
    if (globalStepVariables.isEmpty) {
      None
    } else {
      val globalStep = globalStepVariables.head
      if (!globalStep.dataType.isInteger)
        throw new IllegalStateException(
          s"Existing global step variable does not have integer type: ${globalStep.dataType}.")
      if (globalStep.shape.rank != 0 && globalStep.shape.isFullyDefined)
        throw new IllegalStateException(
          s"Existing global step is not a scalar: ${globalStep.shape}.")
      Some(globalStep)
    }
  }

  /** Gets the global step variable, or creates one and returns it, if none exists.
    *
    * @param  graph Graph to find/create the global step in. If missing, the current graph is used.
    * @return Global step variable.
    */
  def getOrCreate(graph: Graph = Op.currentGraph): Variable = get(graph).getOrElse(create(graph))
}
