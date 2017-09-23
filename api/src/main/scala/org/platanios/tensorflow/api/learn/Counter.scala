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

/** Contains helper methods for creating and obtaining counter variables (e.g., epoch or global iteration).
  *
  * @author Emmanouil Antonios Platanios
  */
object Counter {
  /** Creates a counter variable specified by `key`.
    *
    * @param  key   Graph collection key that will contain the created counter.
    * @param  graph Graph in which to create the counter variable. If missing, the current graph is used.
    * @return Created counter variable.
    * @throws IllegalStateException If a counter variable already exists in the graph collection that corresponds to the
    *                               provided key.
    */
  @throws[IllegalStateException]
  def create(key: Graph.Key[Variable], graph: Graph = Op.currentGraph): Variable = {
    get(key, graph) match {
      case Some(_) => throw new IllegalStateException(s"A ${key.name} variable already exists in this graph.")
      case None => Op.createWith(graph, nameScope = "") {
        Variable.getVariable(
          name = key.name,
          dataType = INT64,
          shape = Shape.scalar(),
          initializer = ZerosInitializer,
          trainable = false,
          collections = Set(Graph.Keys.GLOBAL_VARIABLES, key))
      }
    }
  }

  /** Gets the counter variable specified by `key`.
    *
    * The counter variable must be an integer variable and it should be included in the graph collection specified by
    * the provided key.
    *
    * @param  key   Graph collection key that contains the counter.
    * @param  graph Graph to find the counter in. If missing, the current graph is used.
    * @return The counter variable, or `None`, if not found.
    * @throws IllegalStateException If more than one variables exist in the graph collection that corresponds to the
    *                               provided key, or if the obtained global step is not an integer scalar.
    */
  @throws[IllegalStateException]
  def get(key: Graph.Key[Variable], graph: Graph = Op.currentGraph): Option[Variable] = {
    val counterVariables = graph.getCollection(key)
    if (counterVariables.size > 1)
      throw new IllegalStateException(s"There should only exist one ${key.name} variable in the graph.")
    if (counterVariables.isEmpty) {
      None
    } else {
      val counter = counterVariables.head
      if (!counter.dataType.isInteger)
        throw new IllegalStateException(
          s"Existing ${key.name} variable does not have integer type: ${counter.dataType}.")
      if (counter.shape.rank != 0 && counter.shape.isFullyDefined)
        throw new IllegalStateException(
          s"Existing ${key.name} is not a scalar: ${counter.shape}.")
      Some(counter)
    }
  }

  /** Gets the counter variable specified by `key`, or creates one and returns it, if none exists.
    *
    * @param  key   Graph collection key that should contain the counter.
    * @param  graph Graph to find/create the counter in. If missing, the current graph is used.
    * @return Counter variable.
    */
  def getOrCreate(key: Graph.Key[Variable], graph: Graph = Op.currentGraph): Variable = {
    get(key, graph).getOrElse(create(key, graph))
  }
}
