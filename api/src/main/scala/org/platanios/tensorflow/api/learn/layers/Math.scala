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

import org.platanios.tensorflow.api.tf
import org.platanios.tensorflow.api.learn.layers

/**
  * @author Emmanouil Antonios Platanios
  */
object Math {
  trait API {
    type Cast = layers.Cast
    type Mean = layers.Mean
    type Linear = layers.Linear

    def cast(dataType: tf.DataType, name: String = "Cast"): Cast = Cast(dataType = dataType, name = name)

    def mean(name: String = "Mean"): Mean = Mean(name = name)

    def linear(units: Int, useBias: Boolean = true, name: String = "Linear"): Linear = {
      Linear(units = units, useBias = useBias, name = name)
    }
  }

  object API extends API
}

case class Cast private[layers](dataType: tf.DataType, override val name: String = "Cast")
    extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String                 = s"Cast[$dataType]"
  override val forward  : tf.Output => tf.Output = tf.cast(_, dataType, name = name)
}

case class Mean private[layers](override val name: String = "Mean") extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String                 = s"Mean"
  override val forward  : tf.Output => tf.Output = tf.mean(_, name = name)
}

case class Linear private[layers](units: Int, useBias: Boolean = true, override val name: String = "Linear")
    extends NetworkLayer[tf.Output, tf.Output] {
  override val layerType: String                 = s"Linear[$units]"
  override val forward  : tf.Output => tf.Output = input => {
    val weights = tf.variable(
      s"$name.weights", input.dataType, tf.shape(input.shape(-1), units), tf.randomNormalInitializer())
    val product = tf.matmul(input, weights)
    if (useBias)
      tf.addBias(product, tf.variable(s"$name.bias", input.dataType, tf.shape(units), tf.randomNormalInitializer()))
    else
      product
  }
}
