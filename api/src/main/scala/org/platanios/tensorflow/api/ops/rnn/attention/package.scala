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

package org.platanios.tensorflow.api.ops.rnn

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.{Op, Output, TensorArray}
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
package object attention {
  /** State of the attention wrapper RNN cell.
    *
    * @param  cellState         Wrapped cell state.
    * @param  time              `INT32` scalar containing the current time step.
    * @param  attention         Attention emitted at the previous time step.
    * @param  alignments        Alignments emitted at the previous time step for each attention mechanism.
    * @param  alignmentsHistory Alignments emitted at all time steps for each attention mechanism. Call `stack()` on
    *                           each of the tensor arrays to convert them to tensors.
    * @param  attentionState    Attention cell state.
    */
  case class AttentionWrapperState[S, SS, AS, ASS](
      cellState: S, time: Output, attention: Output, alignments: Seq[Output], alignmentsHistory: Seq[TensorArray],
      attentionState: AS
  )(implicit
      evS: WhileLoopVariable.Aux[S, SS],
      evAS: WhileLoopVariable.Aux[AS, ASS]
  )

  object AttentionWrapperState {
    implicit def attentionWrapperStateWhileLoopVariable[S, SS, AS, ASS](implicit
        evS: WhileLoopVariable.Aux[S, SS],
        evAS: WhileLoopVariable.Aux[AS, ASS],
        evOutput: WhileLoopVariable.Aux[Output, Shape],
        evSeqOutput: WhileLoopVariable.Aux[Seq[Output], Seq[Shape]],
        evSeqTensorArray: WhileLoopVariable.Aux[Seq[TensorArray], Seq[Shape]]
    ): WhileLoopVariable.Aux[AttentionWrapperState[S, SS, AS, ASS], (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS)] = {
      new WhileLoopVariable[AttentionWrapperState[S, SS, AS, ASS]] {
        override type ShapeType = (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS)

        override def zero(
            batchSize: Output, dataType: DataType, shape: (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS),
            name: String = "Zero"
        ): AttentionWrapperState[S, SS, AS, ASS] = Op.createWithNameScope(name) {
          AttentionWrapperState[S, SS, AS, ASS](
            evS.zero(batchSize, dataType, shape._1, "CellState"),
            evOutput.zero(batchSize, dataType, shape._2, "Time"),
            evOutput.zero(batchSize, dataType, shape._3, "Attention"),
            evSeqOutput.zero(batchSize, dataType, shape._4, "Alignments"),
            evSeqTensorArray.zero(batchSize, dataType, shape._5, "AlignmentsHistory"),
            evAS.zero(batchSize, dataType, shape._6, "AttentionState"))
        }

        override def size(value: AttentionWrapperState[S, SS, AS, ASS]): Int = {
          evS.size(value.cellState) + evOutput.size(value.time) + evOutput.size(value.attention) +
              evSeqOutput.size(value.alignments) + evSeqTensorArray.size(value.alignmentsHistory) +
              evAS.size(value.attentionState)
        }

        override def outputs(value: AttentionWrapperState[S, SS, AS, ASS]): Seq[Output] = {
          evS.outputs(value.cellState) ++ evOutput.outputs(value.time) ++ evOutput.outputs(value.attention) ++
              evSeqOutput.outputs(value.alignments) ++ evSeqTensorArray.outputs(value.alignmentsHistory) ++
              evAS.outputs(value.attentionState)
        }

        override def shapes(shape: (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS)): Seq[Shape] = {
          evS.shapes(shape._1) ++ evOutput.shapes(shape._2) ++ evOutput.shapes(shape._3) ++
              evSeqOutput.shapes(shape._4) ++ evSeqTensorArray.shapes(shape._5) ++ evAS.shapes(shape._6)
        }

        override def segmentOutputs(
            value: AttentionWrapperState[S, SS, AS, ASS], values: Seq[Output]
        ): (AttentionWrapperState[S, SS, AS, ASS], Seq[Output]) = {
          val (cellState, tail1) = evS.segmentOutputs(value.cellState, values)
          val (time, tail2) = evOutput.segmentOutputs(value.time, tail1)
          val (attention, tail3) = evOutput.segmentOutputs(value.attention, tail2)
          val (alignments, tail4) = evSeqOutput.segmentOutputs(value.alignments, tail3)
          val (alignmentsHistory, tail5) = evSeqTensorArray.segmentOutputs(value.alignmentsHistory, tail4)
          val (attentionState, tail6) = evAS.segmentOutputs(value.attentionState, tail5)
          (AttentionWrapperState[S, SS, AS, ASS](
            cellState, time, attention, alignments, alignmentsHistory, attentionState), tail6)
        }

        override def segmentShapes(
            value: AttentionWrapperState[S, SS, AS, ASS], shapes: Seq[Shape]
        ): ((SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS), Seq[Shape]) = {
          val (shape1, tail1) = evS.segmentShapes(value.cellState, shapes)
          val (shape2, tail2) = evOutput.segmentShapes(value.time, tail1)
          val (shape3, tail3) = evOutput.segmentShapes(value.attention, tail2)
          val (shape4, tail4) = evSeqOutput.segmentShapes(value.alignments, tail3)
          val (shape5, tail5) = evSeqTensorArray.segmentShapes(value.alignmentsHistory, tail4)
          val (shape6, tail6) = evAS.segmentShapes(value.attentionState, tail5)
          ((shape1, shape2, shape3, shape4, shape5, shape6), tail6)
        }

        override def map(
            value: AttentionWrapperState[S, SS, AS, ASS], mapFn: (ops.Symbol) => ops.Symbol
        ): AttentionWrapperState[S, SS, AS, ASS] = {
          val cellState = evS.map(value.cellState, mapFn)
          val time = evOutput.map(value.time, mapFn)
          val attention = evOutput.map(value.attention, mapFn)
          val alignments = evSeqOutput.map(value.alignments, mapFn)
          val alignmentsHistory = evSeqTensorArray.map(value.alignmentsHistory, mapFn)
          val attentionState = evAS.map(value.attentionState, mapFn)
          AttentionWrapperState[S, SS, AS, ASS](
            cellState, time, attention, alignments, alignmentsHistory, attentionState)
        }

        override def mapWithShape(
            value: AttentionWrapperState[S, SS, AS, ASS], shape: (SS, Shape, Shape, Seq[Shape], Seq[Shape], ASS),
            mapFn: (ops.Symbol, Shape) => ops.Symbol
        ): AttentionWrapperState[S, SS, AS, ASS] = {
          val cellState = evS.mapWithShape(value.cellState, shape._1, mapFn)
          val time = evOutput.mapWithShape(value.time, shape._2, mapFn)
          val attention = evOutput.mapWithShape(value.attention, shape._3, mapFn)
          val alignments = evSeqOutput.mapWithShape(value.alignments, shape._4, mapFn)
          val alignmentsHistory = evSeqTensorArray.mapWithShape(value.alignmentsHistory, shape._5, mapFn)
          val attentionState = evAS.mapWithShape(value.attentionState, shape._6, mapFn)
          AttentionWrapperState[S, SS, AS, ASS](
            cellState, time, attention, alignments, alignmentsHistory, attentionState)
        }
      }
    }
  }

  private[rnn] trait API {
    type Attention[AS, ASS] = attention.Attention[AS, ASS]
    type BahdanauAttention = attention.BahdanauAttention
    type LuongAttention = attention.LuongAttention
    type AttentionWrapperCell[S, SS, AS, ASS] = attention.AttentionWrapperCell[S, SS, AS, ASS]

    val LuongAttention      : attention.LuongAttention.type       = attention.LuongAttention
    val BahdanauAttention   : attention.BahdanauAttention.type    = attention.BahdanauAttention
    val AttentionWrapperCell: attention.AttentionWrapperCell.type = attention.AttentionWrapperCell
  }
}
