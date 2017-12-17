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
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer, Variable}
import org.platanios.tensorflow.api.types.DataType

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Math {
  private[layers] trait API {
    type Cast = layers.Cast
    type Sum = layers.Sum
    type Mean = layers.Mean
    type AddBias = layers.AddBias
    type Linear = layers.Linear

    val Cast   : layers.Cast.type    = layers.Cast
    val Sum    : layers.Sum.type     = layers.Sum
    val Mean   : layers.Mean.type    = layers.Mean
    val AddBias: layers.AddBias.type = layers.AddBias
    val Linear : layers.Linear.type  = layers.Linear
  }

  object API extends API
}

case class Cast(override val variableScope: String, dataType: DataType)
    extends Layer[Output, Output](variableScope) {
  override val layerType: String = s"Cast[$dataType]"

  override protected def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.cast(input, dataType, name = variableScope))
  }
}

case class Sum(override val variableScope: String)
    extends Layer[Output, Output](variableScope) {
  override val layerType: String = "Sum"

  override protected def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.sum(input, name = variableScope))
  }
}

case class Mean(override val variableScope: String)
    extends Layer[Output, Output](variableScope) {
  override val layerType: String = "Mean"

  override protected def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.mean(input, name = variableScope))
  }
}

case class AddBias(
    override val variableScope: String,
    initializer: Initializer = RandomNormalInitializer()
) extends Layer[Output, Output](variableScope) {
  override val layerType: String = "AddBias"

  override protected def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val bias = tf.variable(s"$variableScope/Bias", input.dataType, Shape(input.shape(-1)), initializer)
    LayerInstance(input, ops.NN.addBias(input, bias.value), Set(bias))
  }
}

case class Linear(
    override val variableScope: String,
    units: Int,
    useBias: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer(),
    biasInitializer: Initializer = RandomNormalInitializer()
) extends Layer[Output, Output](variableScope) {
  override val layerType: String = s"Linear[$units]"

  override protected def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val weights = tf.variable("Weights", input.dataType, Shape(input.shape(-1), units), weightsInitializer)
    val trainableVariables = mutable.Set[Variable](weights)
    val bias = {
      if (useBias) {
        val bias = tf.variable("Bias", input.dataType, Shape(units), biasInitializer)
        trainableVariables += bias
        bias.value
      } else {
        null
      }
    }
    val output = ops.NN.linear(input, weights.value, bias)
    LayerInstance(input, output, trainableVariables.toSet)
  }
}
