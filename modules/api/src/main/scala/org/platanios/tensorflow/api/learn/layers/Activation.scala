/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.types.{IsDecimal, IsReal, IsNotQuantized, TF}
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Activation[T: TF](
    override val name: String
) extends Layer[Output[T], Output[T]](name)

object Activation {
  private[layers] trait API {
    type Activation[T] = layers.Activation[T]
    type Sigmoid[T] = layers.Sigmoid[T]
    type LogSigmoid[T] = layers.LogSigmoid[T]
    type ReLU[T] = layers.ReLU[T]
    type ReLU6[T] = layers.ReLU6[T]
    type CReLU[T] = layers.CReLU[T]
    type ELU[T] = layers.ELU[T]
    type SELU[T] = layers.SELU[T]
    type Softplus[T] = layers.Softplus[T]
    type Softsign[T] = layers.Softsign[T]

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

case class Sigmoid[T: TF : IsNotQuantized](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "Sigmoid"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Math.sigmoid(input)
  }
}

case class LogSigmoid[T: TF : IsDecimal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "LogSigmoid"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.Math.logSigmoid(input)
  }
}

case class ReLU[T: TF : IsReal](
    override val name: String,
    alpha: Float = 0.0f
) extends Activation(name) {
  override val layerType: String = if (alpha > 0.0f) f"LeakyReLU($alpha%.2f)" else "ReLU"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.relu(input, alpha = alpha)
  }
}

case class ReLU6[T: TF : IsReal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "ReLU6"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.relu6(input)
  }
}

case class CReLU[T: TF : IsReal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "CReLU"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.crelu(input)
  }
}

case class ELU[T: TF : IsReal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "ELU"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.elu(input)
  }
}

case class SELU[T: TF : IsReal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "SELU"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.selu(input)
  }
}

case class Softplus[T: TF : IsDecimal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "Softplus"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.softplus(input)
  }
}

case class Softsign[T: TF : IsDecimal](
    override val name: String
) extends Activation(name) {
  override val layerType: String = "Softsign"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.NN.softsign(input)
  }
}
