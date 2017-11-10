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
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}
import org.platanios.tensorflow.api.types.DataType

abstract class RNNCell(override protected val name: String)
    extends Layer[RNNCell.Tuple, RNNCell.Tuple](name) {
  def stateSize: Seq[Int]
  def outputSize: Seq[Int]

  /** Returns a sequence of zero-filled tensors representing a zero-valued state for this RNN cell.
    *
    * @param  batchSize Batch size.
    * @param  dataType  Data type for the state tensors.
    * @return Sequence of zero-filled state tensors.
    */
  def zeroState(batchSize: Output, dataType: DataType): Seq[Output] = {
    // TODO: Add support for caching the zero state per graph and batch size.
    Op.createWithNameScope(s"$uniquifiedName/ZeroState", Set(batchSize.op)) {
      stateSize.map(size => {
        val zeroOutput = Basic.fill(dataType, Basic.stack(Seq(batchSize, size), axis = 0))(0)
        zeroOutput.setShape(Shape(Output.constantValue(batchSize).map(_.scalar.asInstanceOf[Int]).getOrElse(-1), size))
        zeroOutput
      })
    }
  }
}

object RNNCell {
  type Tuple = ops.RNNCell.Tuple
  val Tuple: ops.RNNCell.Tuple.type = ops.RNNCell.Tuple
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
