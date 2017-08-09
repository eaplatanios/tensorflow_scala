/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WItf.OutputHOUtf.Output
 * WARRANtf.OutputIES OR CONDItf.OutputIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.learn.layers

import org.platanios.tensorflow.api.tf

/**
  * @author Emmanouil Antonios Platanios
  */
trait Activation extends NetworkLayer[tf.Output, tf.Output]

object Activation {
  trait API {
    def sigmoid(name: String = "Sigmoid"): Sigmoid = Sigmoid(name = name)
    def relu(alpha: Float = 0.0f, name: String = "ReLU"): ReLU = ReLU(alpha = alpha, name = name)
    def relu6(name: String = "ReLU6"): ReLU6 = ReLU6(name = name)
    def crelu(name: String = "CReLU"): CReLU = CReLU(name = name)
    def elu(name: String = "ELU"): ELU = ELU(name = name)
    def selu(name: String = "SELU"): SELU = SELU(name = name)
    def softplus(name: String = "Softplus"): Softplus = Softplus(name = name)
    def softsign(name: String = "Softsign"): Softsign = Softsign(name = name)
  }

  object API extends API
}

case class Sigmoid private[layers](override val name: String = "Sigmoid") extends Activation {
  override val layerType: String                 = s"Sigmoid"
  override val forward  : tf.Output => tf.Output = tf.sigmoid(_, name = name)
}

case class ReLU private[layers](alpha: Float = 0.0f, override val name: String = "ReLU") extends Activation {
  override val layerType: String                 = if (alpha > 0.0f) f"LeakyReLU($alpha%.2f)" else "ReLU"
  override val forward  : tf.Output => tf.Output = tf.relu(_, alpha = alpha, name = name)
}

case class ReLU6 private[layers](override val name: String = "ReLU6") extends Activation {
  override val layerType: String                 = s"ReLU6"
  override val forward  : tf.Output => tf.Output = tf.relu6(_, name = name)
}

case class CReLU private[layers](override val name: String = "CReLU") extends Activation {
  override val layerType: String                 = s"CReLU"
  override val forward  : tf.Output => tf.Output = tf.crelu(_, name = name)
}

case class ELU private[layers](override val name: String = "ELU") extends Activation {
  override val layerType: String                 = s"ELU"
  override val forward  : tf.Output => tf.Output = tf.elu(_, name = name)
}

case class SELU private[layers](override val name: String = "SELU") extends Activation {
  override val layerType: String                 = s"SELU"
  override val forward  : tf.Output => tf.Output = tf.selu(_, name = name)
}

case class Softplus private[layers](override val name: String = "Softplus") extends Activation {
  override val layerType: String                 = s"Softplus"
  override val forward  : tf.Output => tf.Output = tf.softplus(_, name = name)
}

case class Softsign private[layers](override val name: String = "Softsign") extends Activation {
  override val layerType: String                 = s"Softsign"
  override val forward  : tf.Output => tf.Output = tf.softsign(_, name = name)
}
