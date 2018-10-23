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

import org.platanios.tensorflow.api.implicits.helpers.OutputToShape
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.VariableScope

/**
  * @param  name Name scope (also acting as variable scope) for this layer.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNCell[Out, State, OutShape, StateShape](
    override val name: String
)(implicit
    val evOutputToShapeOut: OutputToShape.Aux[Out, OutShape],
    val evOutputToShapeState: OutputToShape.Aux[State, StateShape]
) extends Layer[Tuple[Out, State], Tuple[Out, State]](name) {
  def createCellWithoutContext(
      mode: Mode,
      inputShape: OutShape
  ): ops.rnn.cell.RNNCell[Out, State, OutShape, StateShape]

  final def createCell(
      mode: Mode,
      inputShape: OutShape
  ): ops.rnn.cell.RNNCell[Out, State, OutShape, StateShape] = {
    if (name != null) {
      VariableScope.scope(name, isPure = true) {
        createCellWithoutContext(mode, inputShape)
      }
    } else {
      createCellWithoutContext(mode, inputShape)
    }
  }

  override final def forwardWithoutContext(
      input: Tuple[Out, State]
  )(implicit mode: Mode): Tuple[Out, State] = {
    createCellWithoutContext(mode, evOutputToShapeOut.shape(input.output)).forward(input)
  }
}
