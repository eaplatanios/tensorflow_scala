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

  type Op[I, O] = ops.Op[I, O]
  type UntypedOp = ops.UntypedOp

  type OutputLike[T] = ops.OutputLike[T]
  type Output[T] = ops.Output[T]
  type OutputIndexedSlices[T] = ops.OutputIndexedSlices[T]
  type SparseOutput[T] = ops.SparseOutput[T]

  type TensorArray[T] = ops.TensorArray[T]

  val Op                 : ops.Op.type                  = ops.Op
  val Output             : ops.Output.type              = ops.Output
  val OutputIndexedSlices: ops.OutputIndexedSlices.type = ops.OutputIndexedSlices
  val SparseOutput       : ops.SparseOutput.type        = ops.SparseOutput
  val TensorArray        : ops.TensorArray.type         = ops.TensorArray

  type VariableLike[T] = ops.variables.VariableLike[T]
  type Variable[T] = ops.variables.Variable[T]

  //region Types

  //region Value Classes

  type Half = core.types.Half
  type TruncatedHalf = core.types.TruncatedHalf
  type ComplexFloat = core.types.ComplexFloat
  type ComplexDouble = core.types.ComplexDouble
  type UByte = core.types.UByte
  type UShort = core.types.UShort
  type UInt = core.types.UInt
  type ULong = core.types.ULong
  type QByte = core.types.QByte
  type QShort = core.types.QShort
  type QInt = core.types.QInt
  type QUByte = core.types.QUByte
  type QUShort = core.types.QUShort
  type Resource = core.types.Resource
  type Variant = core.types.Variant

  val Half         : core.types.Half.type          = core.types.Half
  val TruncatedHalf: core.types.TruncatedHalf.type = core.types.TruncatedHalf
  val ComplexFloat : core.types.ComplexFloat.type  = core.types.ComplexFloat
  val ComplexDouble: core.types.ComplexDouble.type = core.types.ComplexDouble
  val UByte        : core.types.UByte.type         = core.types.UByte
  val UShort       : core.types.UShort.type        = core.types.UShort
  val UInt         : core.types.UInt.type          = core.types.UInt
  val ULong        : core.types.ULong.type         = core.types.ULong
  val QByte        : core.types.QByte.type         = core.types.QByte
  val QShort       : core.types.QShort.type        = core.types.QShort
  val QInt         : core.types.QInt.type          = core.types.QInt
  val QUByte       : core.types.QUByte.type        = core.types.QUByte
  val QUShort      : core.types.QUShort.type       = core.types.QUShort
  val Resource     : core.types.Resource.type      = core.types.Resource
  val Variant      : core.types.Variant.type       = core.types.Variant

  //endregion Value Classes

  //region Data Type Instances

  type DataType[T] = core.types.DataType[T]

  type STRING = core.types.DataType[String]
  type BOOLEAN = core.types.DataType[Boolean]
  type FLOAT16 = core.types.DataType[core.types.Half]
  type FLOAT32 = core.types.DataType[Float]
  type FLOAT64 = core.types.DataType[Double]
  type BFLOAT16 = core.types.DataType[core.types.TruncatedHalf]
  type COMPLEX64 = core.types.DataType[core.types.ComplexFloat]
  type COMPLEX128 = core.types.DataType[core.types.ComplexDouble]
  type INT8 = core.types.DataType[Byte]
  type INT16 = core.types.DataType[Short]
  type INT32 = core.types.DataType[Int]
  type INT64 = core.types.DataType[Long]
  type UINT8 = core.types.DataType[core.types.UByte]
  type UINT16 = core.types.DataType[core.types.UShort]
  type UINT32 = core.types.DataType[core.types.UInt]
  type UINT64 = core.types.DataType[core.types.ULong]
  type QINT8 = core.types.DataType[core.types.QByte]
  type QINT16 = core.types.DataType[core.types.QShort]
  type QINT32 = core.types.DataType[core.types.QInt]
  type QUINT8 = core.types.DataType[core.types.QUByte]
  type QUINT16 = core.types.DataType[core.types.QUShort]
  type RESOURCE = core.types.DataType[core.types.Resource]
  type VARIANT = core.types.DataType[core.types.Variant]

  val STRING    : STRING     = core.types.STRING
  val BOOLEAN   : BOOLEAN    = core.types.BOOLEAN
  val FLOAT16   : FLOAT16    = core.types.FLOAT16
  val FLOAT32   : FLOAT32    = core.types.FLOAT32
  val FLOAT64   : FLOAT64    = core.types.FLOAT64
  val BFLOAT16  : BFLOAT16   = core.types.BFLOAT16
  val COMPLEX64 : COMPLEX64  = core.types.COMPLEX64
  val COMPLEX128: COMPLEX128 = core.types.COMPLEX128
  val INT8      : INT8       = core.types.INT8
  val INT16     : INT16      = core.types.INT16
  val INT32     : INT32      = core.types.INT32
  val INT64     : INT64      = core.types.INT64
  val UINT8     : UINT8      = core.types.UINT8
  val UINT16    : UINT16     = core.types.UINT16
  val UINT32    : UINT32     = core.types.UINT32
  val UINT64    : UINT64     = core.types.UINT64
  val QINT8     : QINT8      = core.types.QINT8
  val QINT16    : QINT16     = core.types.QINT16
  val QINT32    : QINT32     = core.types.QINT32
  val QUINT8    : QUINT8     = core.types.QUINT8
  val QUINT16   : QUINT16    = core.types.QUINT16
  val RESOURCE  : RESOURCE   = core.types.RESOURCE
  val VARIANT   : VARIANT    = core.types.VARIANT

  //endregion Data Type Instances

  //region Type Traits

  type TF[T] = core.types.TF[T]
  type IsFloatOrDouble[T] = core.types.IsFloatOrDouble[T]
  type IsHalfOrFloatOrDouble[T] = core.types.IsHalfOrFloatOrDouble[T]
  type IsTruncatedHalfOrFloatOrDouble[T] = core.types.IsTruncatedHalfOrFloatOrDouble[T]
  type IsTruncatedHalfOrHalfOrFloat[T] = core.types.IsTruncatedHalfOrHalfOrFloat[T]
  type IsDecimal[T] = core.types.IsDecimal[T]
  type IsIntOrLong[T] = core.types.IsIntOrLong[T]
  type IsIntOrLongOrFloatOrDouble[T] = core.types.IsIntOrLongOrFloatOrDouble[T]
  type IsIntOrLongOrHalfOrFloatOrDouble[T] = core.types.IsIntOrLongOrHalfOrFloatOrDouble[T]
  type IsIntOrLongOrUByte[T] = core.types.IsIntOrLongOrUByte[T]
  type IsIntOrUInt[T] = core.types.IsIntOrUInt[T]
  type IsStringOrInteger[T] = core.types.IsStringOrInteger[T]
  type IsStringOrFloatOrLong[T] = core.types.IsStringOrFloatOrLong[T]
  type IsReal[T] = core.types.IsReal[T]
  type IsComplex[T] = core.types.IsComplex[T]
  type IsNotQuantized[T] = core.types.IsNotQuantized[T]
  type IsQuantized[T] = core.types.IsQuantized[T]
  type IsNumeric[T] = core.types.IsNumeric[T]
  type IsBooleanOrNumeric[T] = core.types.IsBooleanOrNumeric[T]

  val TF                              : core.types.TF.type                               = core.types.TF
  val IsFloatOrDouble                 : core.types.IsFloatOrDouble.type                  = core.types.IsFloatOrDouble
  val IsHalfOrFloatOrDouble           : core.types.IsHalfOrFloatOrDouble.type            = core.types.IsHalfOrFloatOrDouble
  val IsTruncatedHalfOrFloatOrDouble  : core.types.IsTruncatedHalfOrFloatOrDouble.type   = core.types.IsTruncatedHalfOrFloatOrDouble
  val IsTruncatedHalfOrHalfOrFloat    : core.types.IsTruncatedHalfOrHalfOrFloat.type     = core.types.IsTruncatedHalfOrHalfOrFloat
  val IsDecimal                       : core.types.IsDecimal.type                        = core.types.IsDecimal
  val IsIntOrLong                     : core.types.IsIntOrLong.type                      = core.types.IsIntOrLong
  val IsIntOrLongOrFloatOrDouble      : core.types.IsIntOrLongOrFloatOrDouble.type       = core.types.IsIntOrLongOrFloatOrDouble
  val IsIntOrLongOrHalfOrFloatOrDouble: core.types.IsIntOrLongOrHalfOrFloatOrDouble.type = core.types.IsIntOrLongOrHalfOrFloatOrDouble
  val IsIntOrLongOrUByte              : core.types.IsIntOrLongOrUByte.type               = core.types.IsIntOrLongOrUByte
  val IsIntOrUInt                     : core.types.IsIntOrUInt.type                      = core.types.IsIntOrUInt
  val IsStringOrFloatOrLong           : core.types.IsStringOrFloatOrLong.type            = core.types.IsStringOrFloatOrLong
  val IsReal                          : core.types.IsReal.type                           = core.types.IsReal
  val IsComplex                       : core.types.IsComplex.type                        = core.types.IsComplex
  val IsNotQuantized                  : core.types.IsNotQuantized.type                   = core.types.IsNotQuantized
  val IsQuantized                     : core.types.IsQuantized.type                      = core.types.IsQuantized
  val IsNumeric                       : core.types.IsNumeric.type                        = core.types.IsNumeric
  val IsBooleanOrNumeric              : core.types.IsBooleanOrNumeric.type               = core.types.IsBooleanOrNumeric

  //endregion Type Traits

  //endregion Types

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
    object learn extends api.learn.API
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
