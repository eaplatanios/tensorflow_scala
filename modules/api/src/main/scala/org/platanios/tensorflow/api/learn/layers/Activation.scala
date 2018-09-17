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

import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Activation(override val name: String) extends Layer[Output, Output](name)

object Activation {
  private[layers] trait API {
    type Activation = layers.Activation

    val Sigmoid   : layers.Sigmoid.type    = layers.Sigmoid
    val LogSigmoid: layers.LogSigmoid.type = layers.LogSigmoid
    val ReLU      : layers.ReLU.type       = layers.ReLU
    val ReLU6     : layers.ReLU6.type      = layers.ReLU6
    val CReLU     : layers.CReLU.type      = layers.CReLU
    val ELU       : layers.ELU.type        = layers.ELU
    val SELU      : layers.SELU.type       = layers.SELU
    val Softplus  : layers.Softplus.type   = layers.Softplus
    val Softsign  : layers.Softsign.type   = layers.Softsign
  }

  object API extends API
}

case class Sigmoid(override val name: String)
    extends Activation(name) {
  override val layerType: String = "Sigmoid"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.Math.sigmoid(input)
  }
}

case class LogSigmoid(override val name: String)
    extends Activation(name) {
  override val layerType: String = "LogSigmoid"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.Math.logSigmoid(input)
  }
}

case class ReLU(override val name: String, alpha: Float = 0.0f)
    extends Activation(name) {
  override val layerType: String = if (alpha > 0.0f) f"LeakyReLU($alpha%.2f)" else "ReLU"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.relu(input, alpha = alpha)
  }
}

case class ReLU6(override val name: String)
    extends Activation(name) {
  override val layerType: String = "ReLU6"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.relu6(input)
  }
}

case class CReLU(override val name: String)
    extends Activation(name) {
  override val layerType: String = "CReLU"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.crelu(input)
  }
}

case class ELU(override val name: String)
    extends Activation(name) {
  override val layerType: String = "ELU"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.elu(input)
  }
}

case class SELU(override val name: String)
    extends Activation(name) {
  override val layerType: String = "SELU"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.selu(input)
  }
}

case class Softplus(override val name: String)
    extends Activation(name) {
  override val layerType: String = "Softplus"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.softplus(input)
  }
}

case class Softsign(override val name: String)
    extends Activation(name) {
  override val layerType: String = "Softsign"

  override def forwardWithoutContext(input: Output)(implicit mode: Mode): Output = {
    ops.NN.softsign(input)
  }
}
