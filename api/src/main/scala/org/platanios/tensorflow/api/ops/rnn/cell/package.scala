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
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
package object cell {
  class Tuple[O, S](val output: O, val state: S)

  object Tuple {
    def apply[O, S](output: O, state: S): Tuple[O, S] = new Tuple(output, state)
  }

  type BasicTuple = Tuple[Output, Output]

  /** LSTM state tuple.
    *
    * @param  c Memory state tensor (i.e., previous output).
    * @param  m State tensor.
    */
  case class LSTMState(c: Output, m: Output)

  object LSTMState {
    implicit def lstmStateWhileLoopVariable(implicit
        evOutput: WhileLoopVariable.Aux[Output, Shape]
    ): WhileLoopVariable.Aux[LSTMState, (Shape, Shape)] = {
      new WhileLoopVariable[LSTMState] {
        override type ShapeType = (Shape, Shape)

        override def zero(
            batchSize: Output, dataType: DataType, shape: (Shape, Shape), name: String = "Zero"
        ): LSTMState = Op.createWithNameScope(name) {
          LSTMState(
            evOutput.zero(batchSize, dataType, shape._1, "Output"),
            evOutput.zero(batchSize, dataType, shape._2, "State"))
        }

        override def size(value: LSTMState): Int = evOutput.size(value.c) + evOutput.size(value.m)
        override def outputs(value: LSTMState): Seq[Output] = evOutput.outputs(value.c) ++ evOutput.outputs(value.m)
        override def shapes(shape: (Shape, Shape)): Seq[Shape] = evOutput.shapes(shape._1) ++ evOutput.shapes(shape._2)

        override def segmentOutputs(value: LSTMState, values: Seq[Output]): (LSTMState, Seq[Output]) = {
          (LSTMState(values(0), values(1)), values.drop(2))
        }

        override def segmentShapes(value: LSTMState, shapes: Seq[Shape]): ((Shape, Shape), Seq[Shape]) = {
          ((shapes(0), shapes(1)), shapes.drop(2))
        }

        override def map(value: LSTMState, mapFn: (ops.Symbol) => ops.Symbol): LSTMState = {
          LSTMState(
            evOutput.map(value.c, mapFn),
            evOutput.map(value.m, mapFn))
        }

        override def mapWithShape(
            value: LSTMState, shape: (Shape, Shape), mapFn: (ops.Symbol, Shape) => ops.Symbol
        ): LSTMState = {
          LSTMState(
            evOutput.mapWithShape(value.c, shape._1, mapFn),
            evOutput.mapWithShape(value.m, shape._2, mapFn))
        }
      }
    }
  }

  type LSTMTuple = Tuple[Output, LSTMState]

  def LSTMTuple(output: Output, state: LSTMState): LSTMTuple = Tuple(output, state)

  private[rnn] trait API {
    type RNNCell[O, OS, S, SS] = cell.RNNCell[O, OS, S, SS]
    type BasicRNNCell = cell.BasicRNNCell
    type GRUCell = cell.GRUCell
    type BasicLSTMCell = cell.BasicLSTMCell
    type LSTMCell = cell.LSTMCell
    type DropoutRNNCell[O, OS, S, SS] = cell.DropoutRNNCell[O, OS, S, SS]
    type MultiRNNCell[O, OS, S, SS] = cell.MultiRNNCell[O, OS, S, SS]

    val BasicRNNCell  : cell.BasicRNNCell.type   = cell.BasicRNNCell
    val GRUCell       : cell.GRUCell.type        = cell.GRUCell
    val BasicLSTMCell : cell.BasicLSTMCell.type  = cell.BasicLSTMCell
    val LSTMCell      : cell.LSTMCell.type       = cell.LSTMCell
    val DropoutRNNCell: cell.DropoutRNNCell.type = cell.DropoutRNNCell
    val MultiRNNCell  : cell.MultiRNNCell.type   = cell.MultiRNNCell

    type RNNTuple[O, S] = cell.Tuple[O, S]
    type BasicTuple = cell.Tuple[Output, Output]
    type LSTMTuple = cell.Tuple[Output, LSTMState]

    val RNNTuple: cell.Tuple.type = cell.Tuple

    def LSTMTuple(output: Output, state: LSTMState): LSTMTuple = cell.Tuple(output, state)

    type LSTMState = cell.LSTMState

    val LSTMState: cell.LSTMState.type = cell.LSTMState
  }
}
