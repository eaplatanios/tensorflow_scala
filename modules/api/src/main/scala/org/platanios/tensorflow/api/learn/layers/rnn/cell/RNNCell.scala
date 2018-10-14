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
import org.platanios.tensorflow.api.learn.Mode
import org.platanios.tensorflow.api.learn.layers.Layer
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.variables.VariableScope

/**
  * @param  name Name scope (also acting as variable scope) for this layer.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class RNNCell[O, S](
    override val name: String
)(implicit
    protected val evStructureO: NestedStructure[O],
    protected val evStructureS: NestedStructure[S]
) extends Layer[Tuple[O, S], Tuple[O, S]](name) {
  protected implicit val evStructureOAux: NestedStructure.Aux[O, evStructureO.V, evStructureO.D, evStructureO.S] = {
    evStructureO.asAux()
  }

  protected implicit val evStructureSAux: NestedStructure.Aux[S, evStructureS.V, evStructureS.D, evStructureS.S] = {
    evStructureS.asAux()
  }

  def createCellWithoutContext[OV, OD, OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureO: NestedStructure.Aux[O, OV, OD, OS]): ops.rnn.cell.RNNCell[O, S]

  final def createCell[OV, OD, OS](
      mode: Mode,
      inputShape: OS
  )(implicit evStructureO: NestedStructure.Aux[O, OV, OD, OS]): ops.rnn.cell.RNNCell[O, S] = {
    if (name != null) {
      VariableScope.scope(name, isPure = true) {
        createCellWithoutContext(mode, inputShape)
      }
    } else {
      createCellWithoutContext(mode, inputShape)
    }
  }

  override final def forwardWithoutContext(
      input: Tuple[O, S]
  )(implicit mode: Mode): Tuple[O, S] = {
    createCellWithoutContext(mode, evStructureOAux.shapeFromOutput(input.output)).forward(input)
  }
}
