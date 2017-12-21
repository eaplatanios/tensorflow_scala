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

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNCell[O, OS, S, SS](override val variableScope: String)(implicit
  evO: WhileLoopVariable.Aux[O, OS],
  evS: WhileLoopVariable.Aux[S, SS]
) extends Layer[Tuple[O, S], Tuple[O, S]](variableScope) {
  def createCell(mode: Mode, inputShape: OS): ops.rnn.cell.RNNCell[O, OS, S, SS]

  override final protected def forward(input: Tuple[O, S], mode: Mode): Tuple[O, S] = {
    createCell(mode, evO.fromShapes(input.output, evO.outputs(input.output).map(_.shape))).forward(input)
  }
}
