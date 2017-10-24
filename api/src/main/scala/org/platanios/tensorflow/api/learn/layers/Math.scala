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

import org.platanios.tensorflow.api.Shape
import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{RandomNormalInitializer, Variable}
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
object Math {
  trait API {
    val Cast  : layers.Cast.type   = layers.Cast
    val Sum   : layers.Sum.type    = layers.Sum
    val Mean  : layers.Mean.type   = layers.Mean
    val Linear: layers.Linear.type = layers.Linear
  }

  object API extends API
}

case class Cast private[layers](dataType: DataType, override val name: String = "Cast")
    extends NetworkLayer[Output, Output] {
  override val layerType: String           = s"Cast[$dataType]"
  override val forward  : Output => Output = ops.Math.cast(_, dataType, name = name)
}

case class Sum private[layers](override val name: String = "Sum") extends NetworkLayer[Output, Output] {
  override val layerType: String           = s"Sum"
  override val forward  : Output => Output = ops.Math.sum(_, name = name)
}

case class Mean private[layers](override val name: String = "Mean") extends NetworkLayer[Output, Output] {
  override val layerType: String           = s"Mean"
  override val forward  : Output => Output = ops.Math.mean(_, name = name)
}

case class Linear private[layers](units: Int, useBias: Boolean = true, override val name: String = "Linear")
    extends NetworkLayer[Output, Output] {
  override val layerType: String           = s"Linear[$units]"
  override val forward  : Output => Output = input => {
    val weights = Variable.getVariable(
      s"$name/Weights", input.dataType, Shape(input.shape(-1), units), RandomNormalInitializer())
    val product = ops.Math.matmul(input, weights.value)
    if (useBias) {
      val bias = Variable.getVariable(s"$name/Bias", input.dataType, Shape(units), RandomNormalInitializer())
      ops.NN.addBias(product, bias.value)
    } else {
      product
    }
  }
}
