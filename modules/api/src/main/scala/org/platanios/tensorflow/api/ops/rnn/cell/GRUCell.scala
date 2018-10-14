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
import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.ops.{Basic, Math, NN, Op, Output}

/** The Gated Recurrent Unit (GRU) cell.
  *
  * For details refer to
  * [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](http://arxiv.org/abs/1406.1078).
  *
  * Input tensors must be two-dimensional.
  *
  * @group RNNCellOps
  * @param  gateKernel      Gate kernel matrix to use.
  * @param  gateBias        Gate bias vector to use.
  * @param  candidateKernel Candidate kernel matrix to use.
  * @param  candidateBias   Candidate bias vector to use.
  * @param  activation      Activation function to use.
  * @param  name            Name scope for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class GRUCell[T: TF : IsNotQuantized] protected (
    val gateKernel: Output[T],
    val gateBias: Output[T],
    val candidateKernel: Output[T],
    val candidateBias: Output[T],
    val activation: Output[T] => Output[T],
    val name: String = "GRUCell"
) extends RNNCell[Output[T], Output[T]] {
  override def outputShape[OV, OD, OS](implicit evStructureO: NestedStructure.Aux[Output[T], OV, OD, OS]): OS = {
    candidateBias.shape.asInstanceOf[OS]
  }

  override def stateShape[SV, SD, SS](implicit evStructureS: NestedStructure.Aux[Output[T], SV, SD, SS]): SS = {
    candidateBias.shape.asInstanceOf[SS]
  }

  @throws[IllegalArgumentException]
  override def forward[OV, OD, OS, SV, SD, SS](
      input: BasicTuple[T]
  )(implicit
      evStructureO: NestedStructure.Aux[Output[T], OV, OD, OS],
      evStructureS: NestedStructure.Aux[Output[T], SV, SD, SS]
  ): BasicTuple[T] = {
    Op.nameScope(name) {
      val output = input.output
      val state = input.state
      if (output.rank != 2)
        throw new IllegalArgumentException(s"Input must be rank-2 (provided rank-${output.rank}).")
      if (output.shape(1) == -1)
        throw new IllegalArgumentException(s"Last axis of input shape (${output.shape}) must be known.")
      val gateIn = NN.addBias(Math.matmul(Basic.concatenate(Seq(output, state), axis = 1), gateKernel), gateBias)
      val value = Basic.splitEvenly(Math.sigmoid(gateIn), 2, axis = 1)
      val (r, u) = (value(0), value(1))
      val rState = Math.multiply(r, state)
      val c = NN.addBias(Math.matmul(Basic.concatenate(Seq(output, rState), axis = 1), candidateKernel), candidateBias)
      val newH = Math.add(Math.multiply(u, state), Math.multiply(Basic.ones[T](Shape()) - u, c))
      Tuple(newH, newH)
    }
  }
}

object GRUCell {
  def apply[T: TF : IsNotQuantized](
      gateKernel: Output[T],
      gateBias: Output[T],
      candidateKernel: Output[T],
      candidateBias: Output[T],
      activation: Output[T] => Output[T],
      name: String = "GRUCell"
  ): GRUCell[T] = {
    new GRUCell(
      gateKernel, gateBias,
      candidateKernel, candidateBias,
      activation, name)
  }
}
