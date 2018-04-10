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

package org.platanios.tensorflow

import org.platanios.tensorflow.api.tensors.Context
import org.platanios.tensorflow.api.types.DataType

import spire.math.{UByte, UShort}

import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
package object api extends implicits.Implicits with Documentation {
  //region Graph

  type Graph = core.Graph
  val Graph: core.Graph.type = core.Graph

  Graph.Keys.register(Graph.Keys.RANDOM_SEEDS)
  Graph.Keys.register(Graph.Keys.GLOBAL_VARIABLES)
  Graph.Keys.register(Graph.Keys.LOCAL_VARIABLES)
  Graph.Keys.register(Graph.Keys.MODEL_VARIABLES)
  Graph.Keys.register(Graph.Keys.TRAINABLE_VARIABLES)
  Graph.Keys.register(Graph.Keys.SUMMARIES)
  Graph.Keys.register(Graph.Keys.ASSET_FILEPATHS)
  Graph.Keys.register(Graph.Keys.MOVING_AVERAGE_VARIABLES)
  Graph.Keys.register(Graph.Keys.REGULARIZATION_LOSSES)
  Graph.Keys.register(Graph.Keys.SAVERS)
  Graph.Keys.register(Graph.Keys.WEIGHTS)
  Graph.Keys.register(Graph.Keys.BIASES)
  Graph.Keys.register(Graph.Keys.ACTIVATIONS)
  Graph.Keys.register(Graph.Keys.UPDATE_OPS)
  Graph.Keys.register(Graph.Keys.LOSSES)
  Graph.Keys.register(Graph.Keys.SHARED_RESOURCES)
  Graph.Keys.register(Graph.Keys.LOCAL_RESOURCES)
  Graph.Keys.register(Graph.Keys.TRAINABLE_RESOURCE_VARIABLES)
  Graph.Keys.register(Graph.Keys.INIT_OP)
  Graph.Keys.register(Graph.Keys.LOCAL_INIT_OP)
  Graph.Keys.register(Graph.Keys.READY_OP)
  Graph.Keys.register(Graph.Keys.READY_FOR_LOCAL_INIT_OP)
  Graph.Keys.register(Graph.Keys.SUMMARY_OP)
  Graph.Keys.register(Graph.Keys.GLOBAL_EPOCH)
  Graph.Keys.register(Graph.Keys.GLOBAL_STEP)
  Graph.Keys.register(Graph.Keys.EVAL_STEP)
  Graph.Keys.register(Graph.Keys.TRAIN_OP)
  Graph.Keys.register(Graph.Keys.STREAMING_MODEL_PORTS)
  Graph.Keys.register(Graph.Keys.UNBOUND_INPUTS)

  Graph.Keys.register(ops.control_flow.CondContext.COND_CONTEXTS)
  Graph.Keys.register(ops.control_flow.WhileLoopContext.WHILE_LOOP_CONTEXTS)

  Graph.Keys.register(ops.metrics.Metric.METRIC_VARIABLES)
  Graph.Keys.register(ops.metrics.Metric.METRIC_VALUES)
  Graph.Keys.register(ops.metrics.Metric.METRIC_UPDATES)
  Graph.Keys.register(ops.metrics.Metric.METRIC_RESETS)

  //endregion Graph

  type Session = core.client.Session
  val Session: core.client.Session.type = core.client.Session

  type Shape = core.Shape
  val Shape: core.Shape.type = core.Shape

  type Indexer = core.Indexer
  type Index = core.Index
  type Slice = core.Slice

  val ---    : Indexer = core.Ellipsis
  val NewAxis: Indexer = core.NewAxis
  val ::     : Slice   = core.Slice.::

  implicit val tensorEagerExecutionContext: DynamicVariable[Context] = {
    new DynamicVariable[Context](Context())
  }

  type Tensor = tensors.Tensor
  val Tensor: tensors.Tensor.type = tensors.Tensor

  implicit val opCreationContext: DynamicVariable[api.ops.OpCreationContext] = {
    new DynamicVariable[api.ops.OpCreationContext](api.ops.OpCreationContext(graph = api.core.defaultGraph))
  }

  type Op = ops.Op
  val Op: ops.Op.type = ops.Op

  type OutputLike = ops.OutputLike
  type Output = ops.Output
  type OutputIndexedSlices = ops.OutputIndexedSlices
  type SparseOutput = ops.SparseOutput
  type Variable = ops.variables.Variable
  type PartitionedVariable = ops.variables.PartitionedVariable

  val Output             : ops.Output.type              = ops.Output
  val OutputIndexedSlices: ops.OutputIndexedSlices.type = ops.OutputIndexedSlices
  val SparseOutput       : ops.SparseOutput.type        = ops.SparseOutput

  implicit val layerCreationContext: DynamicVariable[api.learn.layers.LayerCreationContext] = {
    new DynamicVariable[api.learn.layers.LayerCreationContext](api.learn.layers.LayerCreationContext())
  }

  //region Data Types API

  type DataType = types.DataType

  val STRING    : DataType.Aux[String]  = types.STRING
  val BOOLEAN   : DataType.Aux[Boolean] = types.BOOLEAN
  val FLOAT16   : DataType.Aux[Float]   = types.FLOAT16
  val FLOAT32   : DataType.Aux[Float]   = types.FLOAT32
  val FLOAT64   : DataType.Aux[Double]  = types.FLOAT64
  val BFLOAT16  : DataType.Aux[Float]   = types.BFLOAT16
  val COMPLEX64 : DataType.Aux[Double]  = types.COMPLEX64
  val COMPLEX128: DataType.Aux[Double]  = types.COMPLEX128
  val INT8      : DataType.Aux[Byte]    = types.INT8
  val INT16     : DataType.Aux[Short]   = types.INT16
  val INT32     : DataType.Aux[Int]     = types.INT32
  val INT64     : DataType.Aux[Long]    = types.INT64
  val UINT8     : DataType.Aux[UByte]   = types.UINT8
  val UINT16    : DataType.Aux[UShort]  = types.UINT16
  val UINT32    : DataType.Aux[Long]    = types.UINT32
  val QINT8     : DataType.Aux[Byte]    = types.QINT8
  val QINT16    : DataType.Aux[Short]   = types.QINT16
  val QINT32    : DataType.Aux[Int]     = types.QINT32
  val QUINT8    : DataType.Aux[UByte]   = types.QUINT8
  val QUINT16   : DataType.Aux[UShort]  = types.QUINT16
  val RESOURCE  : DataType.Aux[Long]    = types.RESOURCE
  val VARIANT   : DataType.Aux[Long]    = types.VARIANT

  //endregion Data Types API

  type Closeable = utilities.Closeable

  def using[T <: Closeable, R](resource: T)(block: T => R): R = utilities.using(resource)(block)

  private[api] val Disposer = utilities.Disposer

  type ProtoSerializable = utilities.Proto.Serializable

  /** @groupname BasicOps       Ops / Basic
    * @groupprio BasicOps       100
    * @groupname MathOps        Ops / Math
    * @groupprio MathOps        110
    * @groupname SparseOps      Ops / Clip
    * @groupprio SparseOps      120
    * @groupname ClipOps        Ops / Clip
    * @groupprio ClipOps        130
    * @groupname NNOps          Ops / NN
    * @groupprio NNOps          140
    * @groupname StatisticsOps  Ops / Statistics
    * @groupprio StatisticsOps  150
    * @groupname RandomOps      Ops / Random
    * @groupprio RandomOps      160
    * @groupname ParsingOps     Ops / Parsing
    * @groupprio ParsingOps     170
    * @groupname TextOps        Ops / Text
    * @groupprio TextOps        180
    * @groupname ImageOps       Ops / Image
    * @groupprio ImageOps       190
    * @groupname EmbeddingOps   Ops / Embedding
    * @groupprio EmbeddingOps   200
    * @groupname RNNOps         Ops / RNN
    * @groupprio RNNOps         210
    * @groupname RNNCellOps     Ops / RNN Cells
    * @groupprio RNNCellOps     220
    * @groupname ControlFlowOps Ops / Control Flow
    * @groupprio ControlFlowOps 230
    * @groupname LoggingOps     Ops / Logging
    * @groupprio LoggingOps     240
    * @groupname CheckOps       Ops / Checks
    * @groupprio CheckOps       250
    * @groupname SummaryOps     Ops / Summary
    * @groupprio SummaryOps     260
    * @groupname CallbackOps    Ops / Callback
    * @groupprio CallbackOps    270
    */
  object tf
      extends core.API
          with ops.API
          with types.API {
    object data extends api.ops.io.API
    object learn extends api.learn.API
    object metrics extends api.ops.metrics.API
  }

  /** @groupname BasicOps       Ops / Basic
    * @groupprio BasicOps       100
    * @groupname MathOps        Ops / Math
    * @groupprio MathOps        110
    * @groupname SparseOps      Ops / Clip
    * @groupprio SparseOps      120
    * @groupname ClipOps        Ops / Clip
    * @groupprio ClipOps        130
    * @groupname NNOps          Ops / NN
    * @groupprio NNOps          140
    * @groupname StatisticsOps  Ops / Statistics
    * @groupprio StatisticsOps  150
    * @groupname RandomOps      Ops / Random
    * @groupprio RandomOps      160
    * @groupname ParsingOps     Ops / Parsing
    * @groupprio ParsingOps     170
    * @groupname TextOps        Ops / Text
    * @groupprio TextOps        180
    * @groupname ImageOps       Ops / Image
    * @groupprio ImageOps       190
    * @groupname EmbeddingOps   Ops / Embedding
    * @groupprio EmbeddingOps   200
    * @groupname RNNOps         Ops / RNN
    * @groupprio RNNOps         210
    * @groupname RNNCellOps     Ops / RNN Cells
    * @groupprio RNNCellOps     220
    * @groupname ControlFlowOps Ops / Control Flow
    * @groupprio ControlFlowOps 230
    * @groupname LoggingOps     Ops / Logging
    * @groupprio LoggingOps     240
    * @groupname CheckOps       Ops / Checks
    * @groupprio CheckOps       250
    * @groupname SummaryOps     Ops / Summary
    * @groupprio SummaryOps     260
    * @groupname CallbackOps    Ops / Callback
    * @groupprio CallbackOps    270
    */
  object tfi
      extends core.API
          with tensors.API
          with types.API
}
