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

  private[ops] val LARGE_SPARSE_TENSOR_SIZE  = 100000000
  private[ops] val DEFAULT_GRAPH_RANDOM_SEED = 87654321

  private[ops] val COLOCATION_OPS_ATTRIBUTE_NAME  : String = "_class"
  private[ops] val COLOCATION_OPS_ATTRIBUTE_PREFIX: String = "loc:@"
  private[ops] val VALID_OP_NAME_REGEX            : Regex  = "^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$".r
  private[ops] val VALID_NAME_SCOPE_REGEX         : Regex  = "^[A-Za-z0-9_.\\-/]*$".r

  private[ops] val graphConstructionScope: DynamicVariable[api.ops.GraphConstructionScope] = {
    new DynamicVariable[api.ops.GraphConstructionScope](api.ops.GraphConstructionScope(graph = api.core.defaultGraph))
  }

  final case class GraphConstructionScope(
      graph: Graph = Graph(),
      nameScope: String = "",
      device: String = "",
      deviceFunction: OpSpecification => String = _.device,
      colocationOps: Set[Op] = Set.empty,
      controlDependencies: Set[Op] = Set.empty,
      attributes: Map[String, Any] = Map.empty,
      container: String = "", // TODO: !!! Use containers.
      controlFlowContext: Option[Context] = None,
      outerContext: Option[GraphConstructionScope] = None)

  @inline private[ops] def castArgs(output1: Output, output2: Output): (Output, Output) = {
    val dataType = types.DataType.mostPrecise(output1.dataType, output2.dataType)
    (output1.cast(dataType), output2.cast(dataType))
  }

  @inline private[ops] def castArgs(output1: Output, output2: Output, output3: Output): (Output, Output, Output) = {
    val dataType = types.DataType.mostPrecise(output1.dataType, output2.dataType, output3.dataType)
    (output1.cast(dataType), output2.cast(dataType), output3.cast(dataType))
  }

  @inline private[ops] def castArgs(outputs: Seq[Output]): Seq[Output] = {
    val dataType = types.DataType.mostPrecise(outputs.map(_.dataType): _*)
    outputs.map(_.cast(dataType))
  }

  ops.Basic.Gradients
  ops.Cast.Gradients
  ops.DataFlow.Gradients
  ops.Logging.Gradients
  ops.Math.Gradients
  ops.NN.Gradients
  ops.Queue.Gradients
  ops.Parsing.Gradients
  ops.Random.Gradients
  ops.Sets.Gradients
  ops.TensorArray.Gradients
  ops.Text.Gradients
  ops.control_flow.ControlFlow.Gradients
  ops.io.Files.Gradients
  ops.io.Reader.Gradients
  ops.io.data.Dataset.Gradients
  ops.io.data.Iterator.Gradients
  ops.lookup.Lookup.Gradients
  ops.variables.Variable.Gradients

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
          with ops.Op.API
          with ops.Output.API
          with ops.Queue.API
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

    type TensorArray = ops.TensorArray
    val TensorArray: ops.TensorArray.type = ops.TensorArray
  }
}
