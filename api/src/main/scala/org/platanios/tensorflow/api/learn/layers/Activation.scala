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

import org.platanios.tensorflow.api.learn.layers
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
trait Activation extends NetworkLayer[Output, Output]

object Activation {
  trait API {
    type Activation = layers.Activation

    val Sigmoid : layers.Sigmoid.type  = layers.Sigmoid
    val ReLU    : layers.ReLU.type     = layers.ReLU
    val ReLU6   : layers.ReLU6.type    = layers.ReLU6
    val CReLU   : layers.CReLU.type    = layers.CReLU
    val ELU     : layers.ELU.type      = layers.ELU
    val SELU    : layers.SELU.type     = layers.SELU
    val Softplus: layers.Softplus.type = layers.Softplus
    val Softsign: layers.Softsign.type = layers.Softsign
  }

  object API extends API
}

case class Sigmoid private[layers](override val name: String = "Sigmoid") extends Activation {
  override val layerType: String           = s"Sigmoid"
  override val forward  : Output => Output = ops.Math.sigmoid(_, name = name)
}

case class ReLU private[layers](alpha: Float = 0.0f, override val name: String = "ReLU") extends Activation {
  override val layerType: String           = if (alpha > 0.0f) f"LeakyReLU($alpha%.2f)" else "ReLU"
  override val forward  : Output => Output = ops.NN.relu(_, alpha = alpha, name = name)
}

case class ReLU6 private[layers](override val name: String = "ReLU6") extends Activation {
  override val layerType: String           = s"ReLU6"
  override val forward  : Output => Output = ops.NN.relu6(_, name = name)
}

case class CReLU private[layers](override val name: String = "CReLU") extends Activation {
  override val layerType: String           = s"CReLU"
  override val forward  : Output => Output = ops.NN.crelu(_, name = name)
}

case class ELU private[layers](override val name: String = "ELU") extends Activation {
  override val layerType: String           = s"ELU"
  override val forward  : Output => Output = ops.NN.elu(_, name = name)
}

case class SELU private[layers](override val name: String = "SELU") extends Activation {
  override val layerType: String           = s"SELU"
  override val forward  : Output => Output = ops.NN.selu(_, name = name)
}

case class Softplus private[layers](override val name: String = "Softplus") extends Activation {
  override val layerType: String           = s"Softplus"
  override val forward  : Output => Output = ops.NN.softplus(_, name = name)
}

case class Softsign private[layers](override val name: String = "Softsign") extends Activation {
  override val layerType: String           = s"Softsign"
  override val forward  : Output => Output = ops.NN.softsign(_, name = name)
}
