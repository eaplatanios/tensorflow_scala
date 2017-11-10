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

import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.Output

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Activation(override protected val name: String) extends Layer[Output, Output](name)

object Activation {
  private[layers] trait API {
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

case class Sigmoid(override protected val name: String = "Sigmoid")
    extends Activation(name) {
  override val layerType: String = "Sigmoid"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.Math.sigmoid(input, name = uniquifiedName))
  }
}

case class ReLU(alpha: Float = 0.0f, override protected val name: String = "ReLU")
    extends Activation(name) {
  override val layerType: String = if (alpha > 0.0f) f"LeakyReLU($alpha%.2f)" else "ReLU"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.relu(input, alpha = alpha, name = uniquifiedName))
  }
}

case class ReLU6(override protected val name: String = "ReLU6")
    extends Activation(name) {
  override val layerType: String = "ReLU6"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.relu6(input, name = uniquifiedName))
  }
}

case class CReLU(override protected val name: String = "CReLU")
    extends Activation(name) {
  override val layerType: String = "CReLU"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.crelu(input, name = uniquifiedName))
  }
}

case class ELU(override protected val name: String = "ELU")
    extends Activation(name) {
  override val layerType: String = "ELU"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.elu(input, name = uniquifiedName))
  }
}

case class SELU(override protected val name: String = "SELU")
    extends Activation(name) {
  override val layerType: String = "SELU"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.selu(input, name = uniquifiedName))
  }
}

case class Softplus(override protected val name: String = "Softplus")
    extends Activation(name) {
  override val layerType: String = "Softplus"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.softplus(input, name = uniquifiedName))
  }
}

case class Softsign(override protected val name: String = "Softsign")
    extends Activation(name) {
  override val layerType: String = "Softsign"

  override def forward(input: Output, mode: Mode): LayerInstance[Output, Output] = {
    LayerInstance(input, ops.NN.softsign(input, name = uniquifiedName))
  }
}
