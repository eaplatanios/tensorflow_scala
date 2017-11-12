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

package org.platanios.tensorflow.api.learn.layers.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{Layer, LayerInstance}
import org.platanios.tensorflow.api.learn.layers.rnn.cell.RNNCell.Tuple
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.Variable

abstract class RNNCell[O, OS, S, SS](override protected val name: String)
    extends Layer[RNNCell.Tuple[O, S], RNNCell.Tuple[O, S]](name) {
  def createCell(input: O, mode: Mode): RNNCell.CellInstance[O, OS, S, SS]

  override final def forward(input: Tuple[O, S], mode: Mode): LayerInstance[Tuple[O, S], Tuple[O, S]] = {
    val cellInstance = createCell(input.output, mode)
    val output = cellInstance.cell.forward(input)
    LayerInstance(input, output, cellInstance.trainableVariables, cellInstance.nonTrainableVariables)
  }
}

object RNNCell {
  case class CellInstance[O, OS, S, SS](
      cell: ops.rnn.cell.RNNCell[O, OS, S, SS],
      trainableVariables: Set[Variable] = Set.empty,
      nonTrainableVariables: Set[Variable] = Set.empty)

  type BasicCellInstance = CellInstance[Output, Shape, Output, Shape]

  def BasicCellInstance(
      cell: ops.rnn.cell.RNNCell.BasicCell,
      trainableVariables: Set[Variable] = Set.empty,
      nonTrainableVariables: Set[Variable] = Set.empty
  ): BasicCellInstance = {
    CellInstance(cell, trainableVariables, nonTrainableVariables)
  }

  type BasicCell = RNNCell[Output, Shape, Output, Shape]

  type Tuple[O, S] = ops.rnn.cell.RNNCell.Tuple[O, S]
  type BasicTuple = Tuple[Output, Output]

  val Tuple: ops.rnn.cell.RNNCell.Tuple.type = ops.rnn.cell.RNNCell.Tuple

  type LSTMCell = RNNCell[Output, Shape, (Output, Output), (Shape, Shape)]
  type LSTMCellInstance = CellInstance[Output, Shape, (Output, Output), (Shape, Shape)]
  type LSTMTuple = ops.rnn.cell.RNNCell.LSTMTuple

  def LSTMTuple(output: Output, state: (Output, Output)): LSTMTuple = ops.rnn.cell.RNNCell.LSTMTuple(output, state)

  def LSTMCellInstance(
      cell: ops.rnn.cell.RNNCell.LSTMCell,
      trainableVariables: Set[Variable] = Set.empty,
      nonTrainableVariables: Set[Variable] = Set.empty
  ): LSTMCellInstance = {
    CellInstance(cell, trainableVariables, nonTrainableVariables)
  }
}

///**
//  * @param  name              Desired name for this layer (note that this name will be made unique by potentially
//  *                           appending a number to it, if it has been used before for another layer).
//  */
//class RNNCellDropoutWrapper(
//    cell: RNNCell,
//    inputKeepProbability: Float = 1.0f,
//    stateKeepProbability: Float = 1.0f,
//    outputKeepProbability: Float = 1.0f,
//    seed: Long = 0L,
//    override protected val name: String = "RNNCellDropoutWrapper"
//) extends RNNCell(name) {
//  override val layerType: String = "BasicRNNCell"
//
//  override def stateSize: Seq[Shape] = Seq(Shape(numUnits))
//  override def outputSize: Seq[Shape] = Seq(Shape(numUnits))
//
//  override def forward(input: RNNCell.Tuple, mode: Mode): LayerInstance[RNNCell.Tuple, RNNCell.Tuple] = {
//    if (input.output.rank != 2)
//      throw InvalidArgumentException(s"Input must be rank-2 (provided rank-${input.output.rank}).")
//    if (input.output.shape(1) == -1)
//      throw InvalidArgumentException(s"Last axis of input shape (${input.output.shape}) must be known.")
//    if (input.state.length != 1)
//      throw InvalidArgumentException(s"The state must consist of one tensor.")
//    val kernel = variable(
//      RNNCell.KERNEL_NAME, input.output.dataType, Shape(input.output.shape(1) + numUnits, numUnits), kernelInitializer)
//    val bias = variable(RNNCell.BIAS_NAME, input.output.dataType, Shape(numUnits), biasInitializer)
//    val linear = addBias(matmul(concatenate(input.output, input.state.head, axis = 1), kernel), bias)
//    val output = activation(linear)
//    LayerInstance(input, RNNCell.Tuple(output, Seq(output)), Set(kernel, bias))
//  }
//}
