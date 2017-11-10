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
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
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

case class Cast(dataType: DataType, override protected val name: String = "Cast")
    extends Layer[Output, Output](name) {
  override val layerType: String = s"Cast[$dataType]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.cast(input, dataType, name = uniquifiedName))
  }
}

case class Sum(override protected val name: String = "Sum")
    extends Layer[Output, Output](name) {
  override val layerType: String = "Sum"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.sum(input, name = uniquifiedName))
  }
}

case class Mean(override protected val name: String = "Mean")
    extends Layer[Output, Output](name) {
  override val layerType: String = "Mean"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.mean(input, name = uniquifiedName))
  }
}

case class AddBias(
    initializer: Initializer = RandomNormalInitializer(),
    override protected val name: String = "AddBias"
) extends Layer[Output, Output](name) {
  override val layerType: String = "AddBias"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val bias = variable(s"$uniquifiedName/Bias", input.dataType, Shape(input.shape(-1)), initializer)
    LayerInstance(input, ops.NN.addBias(input, bias.value), Set(bias))
  }
}

case class Linear(
    units: Int,
    useBias: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer(),
    biasInitializer: Initializer = RandomNormalInitializer(),
    override protected val name: String = "Linear"
) extends Layer[Output, Output](name) {
  override val layerType: String = s"Linear[$units]"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    val weights = variable(s"$uniquifiedName/Weights", input.dataType, Shape(input.shape(-1), units), weightsInitializer)
    val trainableVariables = mutable.Set[Variable](weights)
    val product = ops.Math.matmul(input, weights.value)
    val output = {
      if (useBias) {
        val bias = variable(s"$uniquifiedName/Bias", input.dataType, Shape(units), biasInitializer)
        trainableVariables += bias
        ops.NN.addBias(product, bias.value)
      } else {
        product
      }
    }
    LayerInstance(input, output, trainableVariables.toSet)
  }
}
