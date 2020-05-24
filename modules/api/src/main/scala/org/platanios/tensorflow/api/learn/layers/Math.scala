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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.{IsNotQuantized, IsNumeric, TF}
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.learn.{Mode, layers}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.variables.{Initializer, RandomNormalInitializer}

/**
  * @author Emmanouil Antonios Platanios
  */
object Math {
  private[layers] trait API {
    type Cast[From, To] = layers.Cast[From, To]
    type AddN[T] = layers.AddN[T]
    type Sum[T] = layers.Sum[T]
    type Mean[T] = layers.Mean[T]
    type AddBias[T] = layers.AddBias[T]
    type Linear[T] = layers.Linear[T]

    val Cast   : layers.Cast.type    = layers.Cast
    val AddN   : layers.AddN.type    = layers.AddN
    val Sum    : layers.Sum.type     = layers.Sum
    val Mean   : layers.Mean.type    = layers.Mean
    val AddBias: layers.AddBias.type = layers.AddBias
    val Linear : layers.Linear.type  = layers.Linear
  }

  object API extends API
}

case class Cast[From, To: TF](
    override val name: String
) extends Layer[Output[From], Output[To]](name) {
  override val layerType: String = s"Cast[${TF[To].dataType}]"

  override def forwardWithoutContext(
      input: Output[From]
  )(implicit mode: Mode): Output[To] = {
    Op.nameScope(name) {
      input.castTo[To]
    }
  }
}

case class AddN[T: TF : IsNumeric](
    override val name: String
) extends Layer[Seq[Output[T]], Output[T]](name) {
  override val layerType: String = "AddN"

  override def forwardWithoutContext(
      input: Seq[Output[T]]
  )(implicit mode: Mode): Output[T] = {
    ops.math.Math.addN(input, name = name)
  }
}

case class Sum[T: TF : IsNumeric](
    override val name: String,
    axes: Seq[Int] = null,
    keepDims: Boolean = false
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "Sum"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.math.Math.sum[T, Int](input, axes, keepDims = keepDims, name = name)
  }
}

case class Mean[T: TF : IsNotQuantized](
    override val name: String,
    axes: Seq[Int] = null,
    keepDims: Boolean = false
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "Mean"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    ops.math.Math.mean[T, Int](input, axes, keepDims = keepDims, name = name)
  }
}

case class AddBias[T: TF : IsNumeric](
    override val name: String,
    initializer: Initializer = RandomNormalInitializer()
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = "AddBias"

  override def forwardWithoutContext(
      input: Output[T]
  )(implicit mode: Mode): Output[T] = {
    val bias = getParameter[T](s"$name/Bias", Shape(input.shape(-1)), initializer)
    ops.NN.addBias(input, bias)
  }
}

case class Linear[T: TF : IsNotQuantized](
    override val name: String,
    units: Int,
    useBias: Boolean = true,
    weightsInitializer: Initializer = RandomNormalInitializer(),
    biasInitializer: Initializer = RandomNormalInitializer()
) extends Layer[Output[T], Output[T]](name) {
  override val layerType: String = s"Linear[$units]"

  override def forwardWithoutContext(input: Output[T])(implicit mode: Mode): Output[T] = {
    val weights = getParameter[T]("Weights", Shape(input.shape(-1), units), weightsInitializer)
    if (useBias)
      ops.NN.linear(input, weights, getParameter[T]("Bias", Shape(units), biasInitializer))
    else
      ops.NN.linear(input, weights)
  }
}
