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

import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.{layerContext, Layer}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.ops.control_flow.WhileLoopVariable
import org.platanios.tensorflow.api.ops.variables.VariableScope

/**
  * @param  name Name scope (also acting as variable scope) for this layer.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNCell[O, OS, S, SS](override val name: String)(implicit
  evO: WhileLoopVariable.Aux[O, OS],
  evS: WhileLoopVariable.Aux[S, SS]
) extends Layer[Tuple[O, S], Tuple[O, S]](name) {
  def createCellWithoutContext(mode: Mode, inputShape: OS): ops.rnn.cell.RNNCell[O, OS, S, SS]

  final def createCell(mode: Mode, inputShape: OS): ops.rnn.cell.RNNCell[O, OS, S, SS ] = Op.createWith(
    nameScope = layerContext.value.nameScope,
    device = layerContext.value.device,
    deviceFunction = layerContext.value.deviceFunction
  ) {
    VariableScope.updatedScope(layerContext.value.variableScope, isPure = true) {
      if (name != null) {
        VariableScope.scope(name, isPure = true) {
          createCellWithoutContext(mode, inputShape)
        }
      } else {
        createCellWithoutContext(mode, inputShape)
      }
    }
  }

  override final protected def _forward(input: Tuple[O, S])(implicit mode: Mode): Tuple[O, S] = {
    createCellWithoutContext(mode, evO.fromShapes(input.output, evO.outputs(input.output).map(_.shape))).forward(input)
  }
}
