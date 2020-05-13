/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.types.Resource
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.ops.math.Math
import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.tensors.Tensor

/** Represents a TensorFlow resource.
  *
  * @param  handle        Handle to the resource.
  * @param  initializeOp  Op that initializes the resource.
  * @param  isInitialized Scalar tensor denoting whether this resource has been initialized.
  *
  * @author Emmanouil Antonios Platanios
  */
case class ResourceWrapper(
    handle: Output[Resource],
    initializeOp: UntypedOp,
    isInitialized: Output[Boolean])

trait Resources {
  /** Returns the set of all shared resources used by the current graph which need to be initialized once per cluster. */
  def sharedResources: Set[ResourceWrapper] = {
    Op.currentGraph.sharedResources
  }

  /** Returns the set of all local resources used by the current graph which need to be initialized once per cluster. */
  def localResources: Set[ResourceWrapper] = {
    Op.currentGraph.localResources
  }
}

object Resources extends Resources {
  /** Registers the provided resource in the appropriate collections.
    *
    * This makes the resource "findable" in either the shared or local resources collection.
    *
    * @param  resource Resource to register.
    * @param  isShared If `true`, the resource gets added to the shared resource collection. Otherwise, it gets added to
    *                  the local resource collection.
    * @param  graph    Graph to register the resource in.
    */
  def register(
      resource: ResourceWrapper,
      isShared: Boolean = true,
      graph: Graph = Op.currentGraph
  ): Unit = {
    if (isShared)
      graph.addToCollection(Graph.Keys.SHARED_RESOURCES)(resource)
    else
      graph.addToCollection(Graph.Keys.LOCAL_RESOURCES)(resource)
  }

  /** Creates an op that returns a tensor containing the names of all uninitialized resources in `resources`.
    *
    * If all resources have been initialized, then an empty tensor is returned.
    *
    * @param  resources Resources to check. If not provided, the set of all shared and local resources in the current
    *                   graph will be used.
    * @param  name      Name for the created op.
    * @return Created op output, which contains the names of the handles of all resources which have not yet been
    *         initialized.
    */
  def uninitializedResources(
      resources: Set[ResourceWrapper] = sharedResources ++ localResources,
      name: String = "UninitializedResources"
  ): Output[String] = {
    // Run all operations on the CPU.
    Op.createWith(nameScope = name, device = "/CPU:0") {
      if (resources.isEmpty) {
        // Return an empty tensor so we only need to check for the returned tensor size being 0 as an indication of
        // model readiness.
        Basic.constant(Tensor.empty[String])
      } else {
        // Get a 1-D boolean tensor listing whether each resource is initialized.
        val resourcesMask = Math.logicalNot(Basic.stack(resources.map(_.isInitialized).toSeq))
        // Get a 1-D string tensor containing all the resource names.
        val resourcesList = resources.map(_.handle.name).toSeq
        val resourceNames = Basic.constant(tensors.ops.Basic.stack(resourcesList.map(Tensor.fill[String](Shape()))))
        // Return a 1-D tensor containing the names of all uninitialized resources.
        Basic.booleanMask(resourceNames, resourcesMask)
      }
    }
  }
}
