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

package org.platanios.tensorflow.api

import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.ops.control_flow.Context

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.DynamicVariable
import scala.util.matching.Regex

/**
  * @author Emmanouil Antonios Platanios
  */
package object ops {
  private[ops] val logger = Logger(LoggerFactory.getLogger("Graph Construction"))

  type UntypedOp = ops.Op[Seq[ops.Output[Any]], Seq[ops.Output[Any]]]

  private[ops] val LARGE_SPARSE_TENSOR_SIZE  = 100000000
  private[ops] val DEFAULT_GRAPH_RANDOM_SEED = 87654321

  private[ops] val COLOCATION_OPS_ATTRIBUTE_NAME  : String = "_class"
  private[ops] val COLOCATION_OPS_ATTRIBUTE_PREFIX: String = "loc:@"
  private[ops] val VALID_OP_NAME_REGEX            : Regex  = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[ops] val VALID_NAME_SCOPE_REGEX         : Regex  = "^[A-Za-z0-9_.\\-/]*$".r

  private[ops] val graphConstructionScope: DynamicVariable[GraphConstructionScope] = {
    new DynamicVariable[GraphConstructionScope](GraphConstructionScope(graph = api.core.defaultGraph))
  }

  final case class GraphConstructionScope(
      graph: Graph = Graph(),
      nameScope: String = "",
      device: String = "",
      deviceFunction: OpSpecification => String = _.device,
      colocationOps: Set[UntypedOp] = Set.empty,
      controlDependencies: Set[UntypedOp] = Set.empty,
      attributes: Map[String, Any] = Map.empty,
      container: String = "", // TODO: !!! Use containers.
      controlFlowContext: Option[Context] = None,
      outerContext: Option[GraphConstructionScope] = None)

  private[api] trait API
      extends Basic
          with Callback
          with Cast
          with Checks
          with Clip
          with DataFlow
          with Embedding
          with Logging
          with Math
          with NN
          with Linalg
          with Parsing
          with Random
          with Resources
          with Sets
          with Statistics
          with Text
          with Gradients.API
          // with ops.Queue.API
          with ops.control_flow.API
          with ops.lookup.API
          with ops.rnn.API
          with ops.variables.API {
    object data extends ops.data.API
    object image extends Image
    object io extends Files
    object metrics extends ops.metrics.API
    object sparse extends Sparse
    object train extends training.API

    object summary extends Summary {
      type FileWriter = api.io.events.SummaryFileWriter
      val FileWriter: api.io.events.SummaryFileWriter.type = api.io.events.SummaryFileWriter
    }

    type OpCreationContext = ops.GraphConstructionScope
    type OpSpecification = ops.OpSpecification

    def currentGraph: Graph = ops.Op.currentGraph
    def currentNameScope: String = ops.Op.currentNameScope
    def currentDevice: String = ops.Op.currentDevice
    def currentDeviceFunction: OpSpecification => String = ops.Op.currentDeviceFunction
    def currentColocationOps: Set[UntypedOp] = ops.Op.currentColocationOps
    def currentControlDependencies: Set[UntypedOp] = ops.Op.currentControlDependencies
    def currentAttributes: Map[String, Any] = ops.Op.currentAttributes
    def currentContainer: String = ops.Op.currentContainer

    def currentGraphRandomSeed(opSeed: Option[Int] = None): (Option[Int], Option[Int]) = {
      ops.Op.currentGraphRandomSeed(opSeed)
    }

    def setCurrentGraphRandomSeed(value: Int): Unit = {
      ops.Op.setCurrentGraphRandomSeed(value)
    }

    def createWith[R](
        graph: Graph = null, nameScope: String = null, device: String = "",
        deviceFunction: Option[OpSpecification => String] = None, colocationOps: Set[UntypedOp] = null,
        controlDependencies: Set[UntypedOp] = null, attributes: Map[String, Any] = null, container: String = null
    )(block: => R): R = {
      ops.Op.createWith(
        graph, nameScope, device, deviceFunction, colocationOps, controlDependencies, attributes, container)(block)
    }

    def nameScope[R](nameScope: String)(block: => R): R = {
      ops.Op.nameScope(nameScope)(block)
    }

    def device[R](
        device: String = "",
        deviceFunction: Option[OpSpecification => String] = None
    )(block: => R): R = {
      ops.Op.device(device, deviceFunction)(block)
    }

    def colocateWith[R](
        colocationOps: Set[UntypedOp],
        ignoreExisting: Boolean = false
    )(block: => R): R = {
      ops.Op.colocateWith(colocationOps, ignoreExisting)(block)
    }

    def initializationScope[R](block: => R): R = {
      ops.Op.initializationScope(block)
    }

    def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): UntypedOp = {
      ops.Op.currentGraph.globalVariablesInitializer(name)
    }

    def localVariablesInitializer(name: String = "LocalVariablesInitializer"): UntypedOp = {
      ops.Op.currentGraph.localVariablesInitializer(name)
    }

    def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): UntypedOp = {
      ops.Op.currentGraph.modelVariablesInitializer(name)
    }

    def metricVariablesInitializer(name: String = "MetricVariablesInitializer"): UntypedOp = {
      ops.Op.currentGraph.metricVariablesInitializer(name)
    }

    def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): UntypedOp = {
      ops.Op.currentGraph.trainableVariablesInitializer(name)
    }
  }
}
