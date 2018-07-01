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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer}
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
object Math {
  private[layers] trait API {
    type Cast = layers.Cast
    type AddN = layers.AddN
    type Sum = layers.Sum
    type Mean = layers.Mean
    type AddBias = layers.AddBias
    type Linear = layers.Linear

    val Cast   : layers.Cast.type    = layers.Cast
    val AddN   : layers.AddN.type    = layers.AddN
    val Sum    : layers.Sum.type     = layers.Sum
    val Mean   : layers.Mean.type    = layers.Mean
    val AddBias: layers.AddBias.type = layers.AddBias
    val Linear : layers.Linear.type  = layers.Linear
  }

  object API extends API
}

case class Cast(override val name: String, dataType: DataType)
    extends Layer[Output, Output](name) {
  override val layerType: String = s"Cast[$dataType]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.Cast.cast(input, dataType, name = name)
  }
}

case class AddN(override val name: String)
    extends Layer[Seq[Output], Output](name) {
  override val layerType: String = "AddN"

  override def forwardWithoutContext(input: Seq[Output])(implicit mode: Mode): Output = {
    ops.Math.addN(input, name = name)
  }
}

case class Sum(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = "Sum"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.Math.sum(input, name = name)
  }
}

case class Mean(override val name: String)
    extends Layer[Output, Output](name) {
  override val layerType: String = "Mean"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.Math.mean(input, name = name)
  }
}

case class AddBias(
    override val name: String,
    initializer: Initializer = RandomNormalInitializer()
) extends Layer[Output, Output](name) {
  override val layerType: String = "AddBias"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    val bias = getParameter(s"$name/Bias", input.dataType, Shape(input.shape(-1)), initializer)
    ops.NN.addBias(input, bias)
  }
}

case class Linear(
    override val name: String,
    units: Int,
    useBias: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer(),
    biasInitializer: Initializer = RandomNormalInitializer()
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Linear[$units]"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    val weights = getParameter("Weights", input.dataType, Shape(input.shape(-1), units), weightsInitializer)
    if (useBias)
      ops.NN.linear(input, weights, getParameter("Bias", input.dataType, Shape(units), biasInitializer))
    else
      ops.NN.linear(input, weights)
  }
}
