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

  val Devices: core.Devices.type = core.Devices

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

  type TensorLike[T] = tensors.TensorLike[T]
  type Tensor[T] = tensors.Tensor[T]
  type TensorIndexedSlices[T] = tensors.TensorIndexedSlices[T]
  type SparseTensor[T] = tensors.SparseTensor[T]

  val Tensor             : tensors.Tensor.type              = tensors.Tensor
  val TensorIndexedSlices: tensors.TensorIndexedSlices.type = tensors.TensorIndexedSlices
  val SparseTensor       : tensors.SparseTensor.type        = tensors.SparseTensor

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

  //region Data Types API

  // TODO: [TYPES] !!! Move the value classes here.

  type DataType[T] = types.DataType[T]

  type STRING = types.DataType[String]
  type BOOLEAN = types.DataType[Boolean]
  type FLOAT16 = types.DataType[types.Half]
  type FLOAT32 = types.DataType[Float]
  type FLOAT64 = types.DataType[Double]
  type BFLOAT16 = types.DataType[types.TruncatedHalf]
  type COMPLEX64 = types.DataType[types.ComplexFloat]
  type COMPLEX128 = types.DataType[types.ComplexDouble]
  type INT8 = types.DataType[Byte]
  type INT16 = types.DataType[Short]
  type INT32 = types.DataType[Int]
  type INT64 = types.DataType[Long]
  type UINT8 = types.DataType[types.UByte]
  type UINT16 = types.DataType[types.UShort]
  type UINT32 = types.DataType[types.UInt]
  type UINT64 = types.DataType[types.ULong]
  type QINT8 = types.DataType[types.QByte]
  type QINT16 = types.DataType[types.QShort]
  type QINT32 = types.DataType[types.QInt]
  type QUINT8 = types.DataType[types.QUByte]
  type QUINT16 = types.DataType[types.QUShort]
  type RESOURCE = types.DataType[Long]
  type VARIANT = types.DataType[Long]

  val STRING    : STRING     = types.DataType.STRING
  val BOOLEAN   : BOOLEAN    = types.DataType.BOOLEAN
  val FLOAT16   : FLOAT16    = types.DataType.FLOAT16
  val FLOAT32   : FLOAT32    = types.DataType.FLOAT32
  val FLOAT64   : FLOAT64    = types.DataType.FLOAT64
  val BFLOAT16  : BFLOAT16   = types.DataType.BFLOAT16
  val COMPLEX64 : COMPLEX64  = types.DataType.COMPLEX64
  val COMPLEX128: COMPLEX128 = types.DataType.COMPLEX128
  val INT8      : INT8       = types.DataType.INT8
  val INT16     : INT16      = types.DataType.INT16
  val INT32     : INT32      = types.DataType.INT32
  val INT64     : INT64      = types.DataType.INT64
  val UINT8     : UINT8      = types.DataType.UINT8
  val UINT16    : UINT16     = types.DataType.UINT16
  val UINT32    : UINT32     = types.DataType.UINT32
  val UINT64    : UINT64     = types.DataType.UINT64
  val QINT8     : QINT8      = types.DataType.QINT8
  val QINT16    : QINT16     = types.DataType.QINT16
  val QINT32    : QINT32     = types.DataType.QINT32
  val QUINT8    : QUINT8     = types.DataType.QUINT8
  val QUINT16   : QUINT16    = types.DataType.QUINT16
  val RESOURCE  : RESOURCE   = types.DataType.RESOURCE
  val VARIANT   : VARIANT    = types.DataType.VARIANT

  //endregion Data Types API

  type Closeable = utilities.Closeable

  def using[T <: Closeable, R](resource: T)(block: T => R): R = utilities.using(resource)(block)

  private[api] val Disposer = utilities.Disposer

  type ProtoSerializable = utilities.Proto.Serializable

  /** @groupname BasicOps       Ops / Basic
    * @groupprio BasicOps       100
    * @groupname CastOps        Ops / Cast
    * @groupprio CastOps        110
    * @groupname MathOps        Ops / Math
    * @groupprio MathOps        120
    * @groupname SparseOps      Ops / Sparse
    * @groupprio SparseOps      130
    * @groupname ClipOps        Ops / Clip
    * @groupprio ClipOps        140
    * @groupname NNOps          Ops / NN
    * @groupprio NNOps          150
    * @groupname StatisticsOps  Ops / Statistics
    * @groupprio StatisticsOps  160
    * @groupname RandomOps      Ops / Random
    * @groupprio RandomOps      170
    * @groupname ParsingOps     Ops / Parsing
    * @groupprio ParsingOps     180
    * @groupname TextOps        Ops / Text
    * @groupprio TextOps        190
    * @groupname ImageOps       Ops / Image
    * @groupprio ImageOps       200
    * @groupname EmbeddingOps   Ops / Embedding
    * @groupprio EmbeddingOps   210
    * @groupname RNNOps         Ops / RNN
    * @groupprio RNNOps         220
    * @groupname RNNCellOps     Ops / RNN Cells
    * @groupprio RNNCellOps     230
    * @groupname ControlFlowOps Ops / Control Flow
    * @groupprio ControlFlowOps 240
    * @groupname LoggingOps     Ops / Logging
    * @groupprio LoggingOps     250
    * @groupname CheckOps       Ops / Checks
    * @groupprio CheckOps       260
    * @groupname SummaryOps     Ops / Summary
    * @groupprio SummaryOps     270
    * @groupname CallbackOps    Ops / Callback
    * @groupprio CallbackOps    280
    */
  object tf
      extends core.API
          with ops.API {
    object data extends api.ops.io.API
    object distribute extends api.ops.training.distribute.API
    object learn extends api.learn.API
    object metrics extends api.ops.metrics.API
  }

  /** @groupname BasicOps       Ops / Basic
    * @groupprio BasicOps       100
    * @groupname CastOps        Ops / Cast
    * @groupprio CastOps        110
    * @groupname MathOps        Ops / Math
    * @groupprio MathOps        120
    * @groupname SparseOps      Ops / Sparse
    * @groupprio SparseOps      130
    * @groupname ClipOps        Ops / Clip
    * @groupprio ClipOps        140
    * @groupname NNOps          Ops / NN
    * @groupprio NNOps          150
    * @groupname StatisticsOps  Ops / Statistics
    * @groupprio StatisticsOps  160
    * @groupname RandomOps      Ops / Random
    * @groupprio RandomOps      170
    * @groupname ParsingOps     Ops / Parsing
    * @groupprio ParsingOps     180
    * @groupname TextOps        Ops / Text
    * @groupprio TextOps        190
    * @groupname ImageOps       Ops / Image
    * @groupprio ImageOps       200
    * @groupname EmbeddingOps   Ops / Embedding
    * @groupprio EmbeddingOps   210
    * @groupname RNNOps         Ops / RNN
    * @groupprio RNNOps         220
    * @groupname RNNCellOps     Ops / RNN Cells
    * @groupprio RNNCellOps     230
    * @groupname ControlFlowOps Ops / Control Flow
    * @groupprio ControlFlowOps 240
    * @groupname LoggingOps     Ops / Logging
    * @groupprio LoggingOps     250
    * @groupname CheckOps       Ops / Checks
    * @groupprio CheckOps       260
    * @groupname SummaryOps     Ops / Summary
    * @groupprio SummaryOps     270
    * @groupname CallbackOps    Ops / Callback
    * @groupprio CallbackOps    280
    */
  object tfi
      extends core.API
          with tensors.API
}
