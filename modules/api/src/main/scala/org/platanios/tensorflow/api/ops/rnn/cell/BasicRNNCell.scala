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

package org.platanios.tensorflow.api.ops.rnn.cell

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output}
import org.platanios.tensorflow.api.types.{IsNotQuantized, TF}

/** The most basic RNN cell.
  *
  * It is defined as: `output = newState = activation(W * input + U * state + b)`.
  *
  * Input tensors must be two-dimensional.
  *
  * @group RNNCellOps
  * @param  kernel     Kernel matrix to use.
  * @param  bias       Bias vector to use.
  * @param  activation Activation function to use.
  * @param  name       Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicRNNCell[T: IsNotQuantized : TF] protected (
    val kernel: Output[T],
    val bias: Output[T],
    val activation: Output[T] => Output[T],
    val name: String = "BasicRNNCell"
) extends RNNCell[Output[T], Shape, Output[T], Shape] {
  override def outputShape: Shape = bias.shape
  override def stateShape: Shape = bias.shape

  @throws[IllegalArgumentException]
  override def forward(input: BasicTuple[T]): BasicTuple[T] = {
    Op.nameScope(name) {
      val output = input.output
      val state = input.state
      if (output.rank != 2)
        throw new IllegalArgumentException(s"Input must be rank-2 (provided rank-${output.rank}).")
      if (output.shape(1) == -1)
        throw new IllegalArgumentException(s"Last axis of input shape (${output.shape}) must be known.")
      val linear = NN.addBias(Math.matmul(Basic.concatenate(Seq(output, state), axis = 1), kernel), bias)
      val newOutput = activation(linear)
      Tuple(newOutput, newOutput)
    }
  }
}

object BasicRNNCell {
  def apply[T: IsNotQuantized : TF](
      kernel: Output[T],
      bias: Output[T],
      activation: Output[T] => Output[T],
      name: String = "BasicRNNCell"
  ): BasicRNNCell[T] = {
    new BasicRNNCell(kernel, bias, activation, name)
  }
}
