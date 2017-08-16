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

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.layers

/**
  * @author Emmanouil Antonios Platanios
  */
object Basic {
  trait API {
    type Flatten = layers.Flatten
    type OneHot = layers.OneHot

    def flatten(name: String = "Flatten"): Flatten = Flatten(name = name)

    def oneHot(numberOfLabels: Int, name: String = "OneHot"): OneHot = {
      OneHot(numberOfLabels = numberOfLabels, name = name)
    }
  }

  object API extends API
}

case class Flatten private[layers](override val name: String = "Flatten") extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String = s"Flatten"
  override val forward: tf.Output => tf.Output = input => {
    if (input.rank == 1)
      input
    else if (input.rank > -1 && input.shape(0) > -1)
      tf.reshape(input, Shape(input.shape(0), -1), name = name)
    else
      tf.reshape(input, Shape(-1) + input.shape.asArray.tail.product, name = name)
  }
}

case class OneHot private[layers](numberOfLabels: Int, override val name: String = "OneHot")
    extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String                 = s"OneHot[$numberOfLabels]"
  override val forward  : tf.Output => tf.Output = tf.oneHot(_, numberOfLabels, name = name)
}
