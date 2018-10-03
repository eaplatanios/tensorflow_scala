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

package org.platanios.tensorflow.api.ops.rnn

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops._

/**
  * @author Emmanouil Antonios Platanios
  */
package object attention {
  /** State of the attention wrapper RNN cell.
    *
    * @param  cellState         Wrapped cell state.
    * @param  time              Scalar containing the current time step.
    * @param  attention         Attention emitted at the previous time step.
    * @param  alignments        Alignments emitted at the previous time step for each attention mechanism.
    * @param  alignmentsHistory Alignments emitted at all time steps for each attention mechanism. Call `stack()` on
    *                           each of the tensor arrays to convert them to tensors.
    * @param  attentionState    Attention cell state.
    */
  case class AttentionWrapperState[T, S, SS, AS, ASS](
      cellState: S,
      time: Output[Int],
      attention: Output[T],
      alignments: Seq[Output[T]],
      alignmentsHistory: Seq[TensorArray[T]],
      attentionState: AS
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evAS: WhileLoopVariable.Aux[AS, ASS])

  object AttentionWrapperState {
    implicit def attentionWrapperStateWhileLoopVariable[T, S, SS, AS, ASS](implicit
        evS: WhileLoopVariable.Aux[S, SS],
        evAS: WhileLoopVariable.Aux[AS, ASS],
        evInt: WhileLoopVariable.Aux[Output[Int], Shape],
        evT: WhileLoopVariable.Aux[Output[T], Shape],
        evSeqOutput: WhileLoopVariable.Aux[Seq[Output[T]], Seq[Shape]],
        evSeqTensorArray: WhileLoopVariable.Aux[Seq[TensorArray[T]], Seq[Shape]]
    ): WhileLoopVariable.Aux[AttentionWrapperState[T, S, SS, AS, ASS], (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS)] = {
      new WhileLoopVariable[AttentionWrapperState[T, S, SS, AS, ASS]] {
        override type ShapeType = (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS)

        override def zero(
            batchSize: Output[Int],
            shape: (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS),
            name: String = "Zero"
        ): AttentionWrapperState[T, S, SS, AS, ASS] = {
          Op.nameScope(name) {
            AttentionWrapperState[T, S, SS, AS, ASS](
              evS.zero(batchSize, shape._1, "CellState"),
              evInt.zero(batchSize, shape._2, "Time"),
              evT.zero(batchSize, shape._3, "Attention"),
              evSeqOutput.zero(batchSize, shape._4, "Alignments"),
              evSeqTensorArray.zero(batchSize, shape._5, "AlignmentsHistory"),
              evAS.zero(batchSize, shape._6, "AttentionState"))
          }
        }

        override def size(
            value: AttentionWrapperState[T, S, SS, AS, ASS]
        ): Int = {
          evS.size(value.cellState) +
              evInt.size(value.time) +
              evT.size(value.attention) +
              evSeqOutput.size(value.alignments) +
              evSeqTensorArray.size(value.alignmentsHistory) +
              evAS.size(value.attentionState)
        }

        override def outputs(
            value: AttentionWrapperState[T, S, SS, AS, ASS]
        ): Seq[Output[Any]] = {
          evS.outputs(value.cellState) ++
              evInt.outputs(value.time) ++
              evT.outputs(value.attention) ++
              evSeqOutput.outputs(value.alignments) ++
              evSeqTensorArray.outputs(value.alignmentsHistory) ++
              evAS.outputs(value.attentionState)
        }

        override def shapes(
            shape: (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS)
        ): Seq[Shape] = {
          evS.shapes(shape._1) ++
              evInt.shapes(shape._2) ++
              evT.shapes(shape._3) ++
              evSeqOutput.shapes(shape._4) ++
              evSeqTensorArray.shapes(shape._5) ++
              evAS.shapes(shape._6)
        }

        override def segmentOutputs(
            value: AttentionWrapperState[T, S, SS, AS, ASS],
            values: Seq[Output[Any]]
        ): (AttentionWrapperState[T, S, SS, AS, ASS], Seq[Output[Any]]) = {
          val (cellState, tail1) = evS.segmentOutputs(value.cellState, values)
          val (time, tail2) = evInt.segmentOutputs(value.time, tail1)
          val (attention, tail3) = evT.segmentOutputs(value.attention, tail2)
          val (alignments, tail4) = evSeqOutput.segmentOutputs(value.alignments, tail3)
          val (alignmentsHistory, tail5) = evSeqTensorArray.segmentOutputs(value.alignmentsHistory, tail4)
          val (attentionState, tail6) = evAS.segmentOutputs(value.attentionState, tail5)
          (AttentionWrapperState[T, S, SS, AS, ASS](
            cellState, time, attention, alignments, alignmentsHistory, attentionState), tail6)
        }

        override def segmentShapes(
            value: AttentionWrapperState[T, S, SS, AS, ASS],
            shapes: Seq[Shape]
        ): ((SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS), Seq[Shape]) = {
          val (shape1, tail1) = evS.segmentShapes(value.cellState, shapes)
          val (shape2, tail2) = evInt.segmentShapes(value.time, tail1)
          val (shape3, tail3) = evT.segmentShapes(value.attention, tail2)
          val (shape4, tail4) = evSeqOutput.segmentShapes(value.alignments, tail3)
          val (shape5, tail5) = evSeqTensorArray.segmentShapes(value.alignmentsHistory, tail4)
          val (shape6, tail6) = evAS.segmentShapes(value.attentionState, tail5)
          ((shape1, shape2, shape3, shape4, shape5, shape6), tail6)
        }

        override def map(
            value: AttentionWrapperState[T, S, SS, AS, ASS],
            mapFn: OutputLikeOrTensorArray[Any] => OutputLikeOrTensorArray[Any]
        ): AttentionWrapperState[T, S, SS, AS, ASS] = {
          val cellState = evS.map(value.cellState, mapFn)
          val time = evInt.map(value.time, mapFn)
          val attention = evT.map(value.attention, mapFn)
          val alignments = evSeqOutput.map(value.alignments, mapFn)
          val alignmentsHistory = evSeqTensorArray.map(value.alignmentsHistory, mapFn)
          val attentionState = evAS.map(value.attentionState, mapFn)
          AttentionWrapperState[T, S, SS, AS, ASS](
            cellState, time, attention, alignments, alignmentsHistory, attentionState)
        }

        override def mapWithShape(
            value: AttentionWrapperState[T, S, SS, AS, ASS],
            shape: (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS),
            mapFn: (OutputLikeOrTensorArray[Any], Shape) => OutputLikeOrTensorArray[Any]
        ): AttentionWrapperState[T, S, SS, AS, ASS] = {
          val cellState = evS.mapWithShape(value.cellState, shape._1, mapFn)
          val time = evInt.mapWithShape(value.time, shape._2, mapFn)
          val attention = evT.mapWithShape(value.attention, shape._3, mapFn)
          val alignments = evSeqOutput.mapWithShape(value.alignments, shape._4, mapFn)
          val alignmentsHistory = evSeqTensorArray.mapWithShape(value.alignmentsHistory, shape._5, mapFn)
          val attentionState = evAS.mapWithShape(value.attentionState, shape._6, mapFn)
          AttentionWrapperState[T, S, SS, AS, ASS](
            cellState, time, attention, alignments, alignmentsHistory, attentionState)
        }
      }
    }
  }

  private[rnn] trait API {
    type Attention[T, AS, ASS] = attention.Attention[T, AS, ASS]
    type BahdanauAttention[T] = attention.BahdanauAttention[T]
    type LuongAttention[T] = attention.LuongAttention[T]
    type AttentionWrapperCell[T, S, SS, AS, ASS] = attention.AttentionWrapperCell[T, S, SS, AS, ASS]

    val LuongAttention      : attention.LuongAttention.type       = attention.LuongAttention
    val BahdanauAttention   : attention.BahdanauAttention.type    = attention.BahdanauAttention
    val AttentionWrapperCell: attention.AttentionWrapperCell.type = attention.AttentionWrapperCell
  }
}
