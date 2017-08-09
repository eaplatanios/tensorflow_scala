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

import org.platanios.tensorflow.api.tf
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.learn.layers.Core._
import org.platanios.tensorflow.api.learn.layers.ModeConditionalNetworkLayer._
import org.platanios.tensorflow.api.learn.layers.ModeConditionedNetworkLayer._

import scala.language.postfixOps
import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class NetworkLayer[T, R] private[layers](implicit context: DynamicVariable[List[LayerCreationContext]])
    extends CachedLayer[T, R] {
  private val creationContext: List[LayerCreationContext] = context.value

  val layerType: String

  def whenIn(mode: Mode): ModeConditionedCapableLayer[T, R] = ModeConditionedCapableLayer(this, Set(mode))
  def whenIn(mode: Mode, modes: Mode*): ModeConditionedCapableLayer[T, R] = {
    ModeConditionedCapableLayer(this, modes.toSet + mode)
  }

  val forward: T => R

  private def forwardWithContext(input: T, context: List[LayerCreationContext] = creationContext): R = {
    context match {
      case LayerCreationGeneralContext(nS, d, c) :: tail =>
        tf.createWith(nameScope = nS, device = d, container = c)(forwardWithContext(input, tail))
      case LayerVariableScope(n, rU, d, i, r, p, c, cG, iDN, iP) :: tail =>
        tf.createWithVariableScope(
          name = n, reuse = rU, dataType = d, initializer = i, regularizer = r, partitioner = p, cachingDevice = c,
          customGetter = cG, isDefaultName = iDN, isPure = iP)(forwardWithContext(input, tail))
      case LayerUpdatedVariableScope(vS, rU, d, i, r, p, c, cG, iP) :: tail =>
        tf.createWithUpdatedVariableScope(
          variableScope = vS, reuse = rU, dataType = d, initializer = i, regularizer = r, partitioner = p,
          cachingDevice = c, customGetter = cG, isPure = iP)(forwardWithContext(input, tail))
      case Nil => forward(input)
    }
  }

  override protected final lazy val callProxy: T => R = forwardWithContext(_)

  override def toString: String = s"$name($layerType)"
}

trait ModeConditionalNetworkLayer {
  // TODO: !!! Handle composition and concatenation.

  implicit def toModeConditionalCapableFunction[C](function: () => C): ModeConditionalCapableFunction[C] = {
    ModeConditionalCapableFunction[C](this, function)
  }

  private[layers] def forModes[C](modes: Set[Mode])(function: () => C, default: () => C): C = {
    // TODO: [CONTROL_FLOW] If (cond) statement combining all modes together.
    ???
  }

  private[layers] def forModes[C](modes: Mode*)(function: () => C, default: () => C): C = {
    forModes(modes.toSet)(function, default)
  }
}

object ModeConditionalNetworkLayer {
  case class ModeConditionalCapableFunction[C] private[layers](layer: ModeConditionalNetworkLayer, function: () => C) {
    def whenIn(mode: Mode): ModeConditionalFunction[C] = ModeConditionalFunction(layer, function, Set(mode))
    def whenIn(mode: Mode, modes: Mode*): ModeConditionalFunction[C] = {
      ModeConditionalFunction(layer, function, modes.toSet + mode)
    }
  }

  case class ModeConditionalFunction[C] private[layers](
      layer: ModeConditionalNetworkLayer, function: () => C, modes: Set[Mode]) {
    def otherwise(default: () => C): C = layer.forModes(modes)(function, default)
  }
}

case class ModeConditionedNetworkLayer[T, R] private[layers](
    layer: NetworkLayer[T, R], modes: Set[Mode], default: NetworkLayer[T, R])
    extends NetworkLayer[T, R] with ModeConditionalNetworkLayer {
  override val name     : String = s"Conditioned[$layer whenIn ${modes.mkString("|")} otherwise $default]"
  override val layerType: String = "ModeConditioned"
  override val forward  : T => R = input => forModes[R](modes)(() => layer.forward(input), () => default.forward(input))
}

object ModeConditionedNetworkLayer {
  case class ModeConditionedCapableLayer[T, R] private[layers](layer: NetworkLayer[T, R], modes: Set[Mode]) {
    def otherwise(default: NetworkLayer[T, R]): ModeConditionedNetworkLayer[T, R] = {
      ModeConditionedNetworkLayer(layer, modes, default)
    }
  }
}

object Core {
  private[layers] sealed trait LayerCreationContext

  private[layers] final case class LayerCreationGeneralContext(
      nameScope: String = "", device: tf.OpSpecification => String = _ => "", container: String = "")
      extends LayerCreationContext

  private[layers] final case class LayerVariableScope(
      name: String, reuse: tf.VariableReuseAllowed = tf.ReuseOrCreateNewVariable, dataType: tf.DataType = null,
      initializer: tf.VariableInitializer = null, regularizer: tf.VariableRegularizer = null,
      partitioner: tf.VariablePartitioner = null, cachingDevice: tf.OpSpecification => String = null,
      customGetter: tf.VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false)
      extends LayerCreationContext

  private[layers] final case class LayerUpdatedVariableScope(
      variableScope: tf.VariableScope, reuse: tf.VariableReuseAllowed = tf.ReuseOrCreateNewVariable,
      dataType: tf.DataType = null, initializer: tf.VariableInitializer = null,
      regularizer: tf.VariableRegularizer = null, partitioner: tf.VariablePartitioner = null,
      cachingDevice: tf.OpSpecification => String = null, customGetter: tf.VariableGetter = null,
      isPure: Boolean = false) extends LayerCreationContext

  trait API {
    type LayerBase[T, R] = layers.NetworkLayer[T, R]

    def withNameScope[R](nameScope: String)(block: => R)
        (implicit context: DynamicVariable[List[LayerCreationContext]]): R = {
      context.withValue(context.value :+ LayerCreationGeneralContext(nameScope = nameScope))(block)
    }

    def withVariableScope[R](
        name: String, reuse: tf.VariableReuseAllowed = tf.ReuseOrCreateNewVariable, dataType: tf.DataType = null,
        initializer: tf.VariableInitializer = null, regularizer: tf.VariableRegularizer = null,
        partitioner: tf.VariablePartitioner = null, cachingDevice: tf.OpSpecification => String = null,
        customGetter: tf.VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false)(block: => R)
        (implicit context: DynamicVariable[List[LayerCreationContext]]): R = {
      context.withValue(context.value :+ LayerVariableScope(
        name = name, reuse = reuse, dataType = dataType, initializer = initializer, regularizer = regularizer,
        partitioner = partitioner, cachingDevice = cachingDevice, customGetter = customGetter,
        isDefaultName = isDefaultName, isPure = isPure))(block)
    }

    def withUpdatedVariableScope[R](
        variableScope: tf.VariableScope, reuse: tf.VariableReuseAllowed = tf.ReuseOrCreateNewVariable,
        dataType: tf.DataType = null, initializer: tf.VariableInitializer = null,
        regularizer: tf.VariableRegularizer = null, partitioner: tf.VariablePartitioner = null,
        cachingDevice: tf.OpSpecification => String = null, customGetter: tf.VariableGetter = null,
        isPure: Boolean = false)(block: => R)
        (implicit context: DynamicVariable[List[LayerCreationContext]]): R = {
      context.withValue(context.value :+ LayerUpdatedVariableScope(
        variableScope = variableScope, reuse = reuse, dataType = dataType, initializer = initializer,
        regularizer = regularizer, partitioner = partitioner, cachingDevice = cachingDevice,
        customGetter = customGetter, isPure = isPure))(block)
    }

    def withDevice[R](device: tf.OpSpecification => String)(block: => R)
        (implicit context: DynamicVariable[List[LayerCreationContext]]): R = {
      context.withValue(context.value :+ LayerCreationGeneralContext(device = device))(block)
    }

    def withContainer[R](container: String)(block: => R)
        (implicit context: DynamicVariable[List[LayerCreationContext]]): R = {
      context.withValue(context.value :+ LayerCreationGeneralContext(container = container))(block)
    }
  }

  object API extends API
}
