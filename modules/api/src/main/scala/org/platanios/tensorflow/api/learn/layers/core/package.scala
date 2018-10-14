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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.core.types.{IsFloat16OrFloat32OrFloat64, TF}
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
package object core {
  private[layers] trait API {
    def MLP[T: TF : IsFloat16OrFloat32OrFloat64](
        name: String,
        hiddenLayers: Seq[Int],
        outputSize: Int,
        activation: String => Layer[Output[T], Output[T]] = null,
        dropout: Float = 0.0f
    ): Layer[Output[T], Output[T]] = {
      if (hiddenLayers.isEmpty) {
        Linear(s"$name/Linear", outputSize)
      } else {
        val activationWithDefault = {
          if (activation == null)
            (name: String) => ReLU[T](name, 0.1f)
          else
            activation
        }
        val size = hiddenLayers.head
        var layer = Linear(s"$name/Layer0/Linear", size) >> activationWithDefault(s"$name/Layer0/Activation")
        hiddenLayers.zipWithIndex.tail.foreach(s => {
          layer = layer >>
              Linear(s"$name/Layer${s._2}/Linear", s._1) >>
              Dropout(s"$name/Layer${s._2}/Dropout", 1 - dropout) >>
              activationWithDefault(s"$name/Layer${s._2}/Activation")
        })
        layer >> Linear("OutputLayer/Linear", outputSize)
      }
    }
  }
}
