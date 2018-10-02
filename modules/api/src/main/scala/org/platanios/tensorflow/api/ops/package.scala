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
          with ops.seq2seq.API
          with ops.variables.API {
    object image extends Image
    object train extends training.API

    object summary extends Summary {
      type FileWriter = api.io.events.SummaryFileWriter
      val FileWriter: api.io.events.SummaryFileWriter.type = api.io.events.SummaryFileWriter
    }

    type OpCreationContext = ops.GraphConstructionScope
    type OpSpecification = ops.OpSpecification

    type Op[+I, +O] = ops.Op[I, O]

    type OutputLike[+T] = ops.OutputLike[T]
    type Output[+T] = ops.Output[T]
    type OutputIndexedSlices[+T] = ops.OutputIndexedSlices[T]
    type SparseOutput[+T] = ops.SparseOutput[T]

    type TensorArray[+T] = ops.TensorArray[T]

    val Op         : ops.Op.type          = ops.Op
    val TensorArray: ops.TensorArray.type = ops.TensorArray

    def currentGraph: Graph = Op.currentGraph
    def currentNameScope: String = Op.currentNameScope
    def currentDevice: String = Op.currentDevice
    def currentDeviceFunction: OpSpecification => String = Op.currentDeviceFunction
    def currentColocationOps: Set[UntypedOp] = Op.currentColocationOps
    def currentControlDependencies: Set[UntypedOp] = Op.currentControlDependencies
    def currentAttributes: Map[String, Any] = Op.currentAttributes
    def currentContainer: String = Op.currentContainer

    def currentGraphRandomSeed(opSeed: Option[Int] = None): (Option[Int], Option[Int]) = {
      Op.currentGraphRandomSeed(opSeed)
    }

    def setCurrentGraphRandomSeed(value: Int): Unit = {
      Op.setCurrentGraphRandomSeed(value)
    }

    def createWith[R](
        graph: Graph = null, nameScope: String = null, device: String = "",
        deviceFunction: OpSpecification => String = _.device, colocationOps: Set[UntypedOp] = null,
        controlDependencies: Set[UntypedOp] = null, attributes: Map[String, Any] = null, container: String = null
    )(block: => R): R = {
      Op.createWith(
        graph, nameScope, device, deviceFunction, colocationOps, controlDependencies, attributes, container)(block)
    }

    def nameScope[R](nameScope: String)(block: => R): R = {
      Op.nameScope(nameScope)(block)
    }

    def device[R](
        device: String = "",
        deviceFunction: OpSpecification => String = _.device
    )(block: => R): R = {
      Op.device(device, deviceFunction)(block)
    }

    def colocateWith[R](
        colocationOps: Set[UntypedOp],
        ignoreExisting: Boolean = false
    )(block: => R): R = {
      Op.colocateWith(colocationOps, ignoreExisting)(block)
    }

    def initializationScope[R](block: => R): R = {
      Op.initializationScope(block)
    }

    def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): UntypedOp = {
      Op.currentGraph.globalVariablesInitializer(name)
    }

    def localVariablesInitializer(name: String = "LocalVariablesInitializer"): UntypedOp = {
      Op.currentGraph.localVariablesInitializer(name)
    }

    def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): UntypedOp = {
      Op.currentGraph.modelVariablesInitializer(name)
    }

    def metricVariablesInitializer(name: String = "MetricVariablesInitializer"): UntypedOp = {
      Op.currentGraph.metricVariablesInitializer(name)
    }

    def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): UntypedOp = {
      Op.currentGraph.trainableVariablesInitializer(name)
    }
  }
}
