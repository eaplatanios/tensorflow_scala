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

package org.platanios.tensorflow.api.learn.layers.rnn.cell

import org.platanios.tensorflow.api.implicits.helpers.NestedStructure
import org.platanios.tensorflow.api.learn.{Mode, TRAINING}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Basic

/** RNN cell that applies dropout to the provided RNN cell.
  *
  * Note that currently, a different dropout mask is used for each time step in an RNN (i.e., not using the variational
  * recurrent dropout method described in
  * ["A Theoretically Grounded Application of Dropout in Recurrent Neural Networks"](https://arxiv.org/abs/1512.05287).
  *
  * Note also that for LSTM cells, no dropout is applied to the memory tensor of the state. It is only applied to the
  * state tensor.
  *
  * @param  name                  Name scope (also acting as variable scope) for this layer.
  * @param  cell                  RNN cell on which to perform dropout.
  * @param  inputKeepProbability  Keep probability for the input of the RNN cell.
  * @param  outputKeepProbability Keep probability for the output of the RNN cell.
  * @param  stateKeepProbability  Keep probability for the output state of the RNN cell.
  * @param  seed                  Optional random seed, used to generate a random seed pair for the random number
  *                               generator, when combined with the graph-level seed.
  *
  * @author Emmanouil Antonios Platanios
  */
class DropoutWrapper[O: NestedStructure, S: NestedStructure](
    override val name: String,
    val cell: RNNCell[O, S],
    val inputKeepProbability: Float = 1.0f,
    val outputKeepProbability: Float = 1.0f,
    val stateKeepProbability: Float = 1.0f,
    val seed: Option[Int] = None
) extends RNNCell[O, S](name) {
  require(inputKeepProbability > 0.0 && inputKeepProbability <= 1.0,
    s"'inputKeepProbability' ($inputKeepProbability) must be in (0, 1].")
  require(outputKeepProbability > 0.0 && outputKeepProbability <= 1.0,
    s"'outputKeepProbability' ($outputKeepProbability) must be in (0, 1].")
  require(stateKeepProbability > 0.0 && stateKeepProbability <= 1.0,
    s"'stateKeepProbability' ($stateKeepProbability) must be in (0, 1].")

  override val layerType: String = "DropoutWrapper"

  override def createCellWithoutContext[OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureO: NestedStructure.Aux[O, _, _, OS]): ops.rnn.cell.RNNCell[O, S] = {
    val createdCell = cell.createCellWithoutContext(mode, inputShape)
    mode match {
      case TRAINING =>
        ops.rnn.cell.DropoutWrapper(
          createdCell,
          Basic.constant(inputKeepProbability, name = "InputKeepProbability"),
          Basic.constant(outputKeepProbability, name = "OutputKeepProbability"),
          Basic.constant(stateKeepProbability, name = "StateKeepProbability"), seed,
          name)
      case _ => createdCell
    }
  }
}

object DropoutWrapper {
  def apply[O: NestedStructure, S: NestedStructure](
      variableScope: String,
      cell: RNNCell[O, S],
      inputKeepProbability: Float = 1.0f,
      outputKeepProbability: Float = 1.0f,
      stateKeepProbability: Float = 1.0f,
      seed: Option[Int] = None
  ): DropoutWrapper[O, S] = {
    new DropoutWrapper(
      variableScope, cell, inputKeepProbability, outputKeepProbability, stateKeepProbability, seed)
  }
}
