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
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output}

/** A basic Long-Short Term Memory (LSTM) cell.
  *
  * The implementation is based on: ["Recurrent Neural Network Regularization", Zaremba et al](http://arxiv.org/abs/1409.2329).
  *
  * We add `forgetBias` (which defaults to 1) to the biases of the forget gate in order to reduce the scale of
  * forgetting in the beginning of training.
  *
  * This cell does not allow for cell clipping, a projection layer, or for peep-hole connections. For advanced
  * models, please use the full `lstmCell` op.
  *
  * Input tensors must be two-dimensional.
  *
  * @group RNNCellOps
  * @param  kernel     Kernel matrix to use.
  * @param  bias       Bias vector to use.
  * @param  activation Activation function to use.
  * @param  forgetBias Forget bias added to the forget gate.
  * @param  name       Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class BasicLSTMCell[T: TF : IsNotQuantized] protected (
    val kernel: Output[T],
    val bias: Output[T],
    val activation: Output[T] => Output[T],
    val forgetBias: Float = 1.0f,
    val name: String = "BasicLSTMCell"
) extends RNNCell[Output[T], LSTMState[T]] {
  private val numUnits = bias.shape(0) / 4

  override def outputShape[OV, OD, OS](implicit evStructureO: NestedStructure.Aux[Output[T], OV, OD, OS]): OS = {
    Shape(numUnits).asInstanceOf[OS]
  }

  override def stateShape[SV, SD, SS](implicit evStructureS: NestedStructure.Aux[LSTMState[T], SV, SD, SS]): SS = {
    (Shape(numUnits), Shape(numUnits)).asInstanceOf[SS]
  }

  @throws[IllegalArgumentException]
  override def forward[OV, OD, OS, SV, SD, SS](
      input: LSTMTuple[T]
  )(implicit
      evStructureO: NestedStructure.Aux[Output[T], OV, OD, OS],
      evStructureS: NestedStructure.Aux[LSTMState[T], SV, SD, SS]
  ): LSTMTuple[T] = {
    Op.nameScope(name) {
      val output = input.output
      if (output.rank != 2)
        throw new IllegalArgumentException(s"Input must be rank-2 (provided rank-${output.rank}).")
      if (output.shape(1) == -1)
        throw new IllegalArgumentException(s"Last axis of input shape (${output.shape}) must be known.")
      val one = Basic.constant(1)
      // Parameters of gates are concatenated into one multiply for efficiency.
      val lstmMatrix = NN.addBias(Math.matmul(Basic.concatenate(Seq(output, input.state.m), axis = 1), kernel), bias)
      // i = input gate, j = new input, f = forget gate, o = output gate
      val lstmMatrixBlocks = Basic.splitEvenly(lstmMatrix, 4, axis = one)
      val (i, j, f, o) = (lstmMatrixBlocks(0), lstmMatrixBlocks(1), lstmMatrixBlocks(2), lstmMatrixBlocks(3))
      val forgetBiasTensor = Basic.constant(forgetBias).castTo[T]
      val c = Math.add(
        Math.multiply(input.state.c, Math.sigmoid(f + forgetBiasTensor)),
        Math.multiply(Math.sigmoid(i), activation(j)))
      val m = Math.multiply(activation(c), Math.sigmoid(o))
      LSTMTuple(m, LSTMState(c, m))
    }
  }
}

object BasicLSTMCell {
  def apply[T: TF : IsNotQuantized](
      kernel: Output[T],
      bias: Output[T],
      activation: Output[T] => Output[T],
      forgetBias: Float = 1.0f,
      name: String = "BasicLSTMCell"
  ): BasicLSTMCell[T] = {
    new BasicLSTMCell(kernel, bias, activation, forgetBias, name)
  }
}
