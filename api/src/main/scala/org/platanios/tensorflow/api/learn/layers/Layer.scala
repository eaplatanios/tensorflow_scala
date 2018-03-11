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

import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.ops.variables.VariableScope.maybeWrapCustomVariableGetter
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.DataType

import scala.util.DynamicVariable

/**
  *
  * '''NOTE:''' Subclasses must implement the `_forward` method. Callers should always use either the `forward` or the
  * `apply` methods.
  *
  * @param  name Name scope (also acting as variable scope) for this layer.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Layer[T, R](
    val name: String
)(implicit
    val context: DynamicVariable[LayerCreationContext]
) {
  val layerType: String

  protected def _forward(input: T, mode: Mode): R

  def forward(input: T, mode: Mode): R = Op.createWith(
    nameScope = context.value.nameScope,
    device = context.value.device,
    deviceFunction = context.value.deviceFunction
  ) {
    VariableScope.createWithUpdatedVariableScope(context.value.variableScope, isPure = true) {
      if (name != null) {
        VariableScope.createWithVariableScope(name, isPure = true) {
          _forward(input, mode)
        }
      } else {
        _forward(input, mode)
      }
    }
  }

  def apply(input: T, mode: Mode): R = forward(input, mode)

  def >>[S](other: Layer[R, S]): Compose[T, R, S] = compose(other)

  def +(other: Layer[T, R]): Concatenate[T, R] = concatenate(other)

  def ++(others: Seq[Layer[T, R]]): Concatenate[T, R] = concatenate(others: _*)

  def compose[S](other: Layer[R, S]): Compose[T, R, S] = Compose(name, this, other)

  def concatenate(others: Layer[T, R]*): Concatenate[T, R] = Concatenate(name, this +: others)

  override def toString: String = layerType
}

private[api] final case class LayerCreationContext(
    nameScope: String = "", variableScope: VariableScope = VariableScope(reuse = ReuseOrCreateNew),
    device: String = "", deviceFunction: OpSpecification => String = _.device)

object Layer {
  trait API {
    def currentNameScope: String = Layer.currentNameScope
    def currentVariableScope: VariableScope = Layer.currentVariableScope
    def currentDevice: String = Layer.currentDevice
    def currentDeviceFunction: OpSpecification => String = Layer.currentDeviceFunction

    def createWith[R](
        nameScope: String = null,
        device: String = "",
        deviceFunction: OpSpecification => String = op => op.device
    )(block: => R): R = {
      Layer.createWith(nameScope, device, deviceFunction)(block)
    }

    def nameScope[R](nameScope: String)(block: => R): R = Layer.createWith(nameScope = nameScope)(block)

    def device[R](device: String)(block: => R): R = Layer.createWith(device = device)(block)

    def deviceFunction[R](deviceFunction: OpSpecification => String)(block: => R): R = {
      Layer.createWith(deviceFunction = deviceFunction)(block)
    }

    def variableScope[R](
        name: String, reuse: ReuseAllowed = ReuseOrCreateNew, dataType: DataType = null,
        initializer: Initializer = null, regularizer: Regularizer = null, partitioner: Partitioner = null,
        cachingDevice: OpSpecification => String = null, customGetter: VariableGetter = null,
        isDefaultName: Boolean = false, isPure: Boolean = false
    )(block: => R): R = {
      Layer.createWithVariableScope(
        name, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter, isPure)(block)
    }

    def updatedVariableScope[R](
        variableScope: VariableScope, reuse: ReuseAllowed = ReuseOrCreateNew, dataType: DataType = null,
        initializer: Initializer = null, regularizer: Regularizer = null, partitioner: Partitioner = null,
        cachingDevice: OpSpecification => String = null, customGetter: VariableGetter = null, isPure: Boolean = false
    )(block: => R): R = {
      Layer.createWithUpdatedVariableScope(
        variableScope, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter,
        isPure)(block)
    }

    type Layer[T, R] = layers.Layer[T, R]
  }

  object API extends API

  /** Variable store object used when creating layers. This variable store is used to store created variables and keep
    * track of variable scope usages. */
  private[this] val variableStore: VariableStore = VariableStore()

  /** Returns the name scope of the current layer creation context. */
  private[layers] def currentNameScope(implicit context: DynamicVariable[LayerCreationContext]): String = {
    if (context.value.nameScope == "")
      ""
    else
      s"${context.value.nameScope}/"
  }

  /** Returns the variable scope of the current layer creation context. */
  private[layers] def currentVariableScope(implicit context: DynamicVariable[LayerCreationContext]): VariableScope = {
    context.value.variableScope
  }

  /** Returns the device of the current layer creation context. */
  private[layers] def currentDevice(implicit context: DynamicVariable[LayerCreationContext]): String = {
    context.value.device
  }

  /** Returns the device function of the current layer creation context. */
  private[layers] def currentDeviceFunction(
      implicit context: DynamicVariable[LayerCreationContext]): OpSpecification => String = {
    context.value.deviceFunction
  }

  private[api] def createWith[R](
      nameScope: String = null,
      device: String = "",
      deviceFunction: OpSpecification => String = _.device
  )(
      block: => R
  )(implicit
      context: DynamicVariable[LayerCreationContext]
  ): R = {
    var updatedContext = context.value
    val newNameScope: String = Op.mergeNameScope(nameScope, updatedContext.nameScope, identity[String])
    updatedContext = updatedContext.copy(nameScope = newNameScope)
    val newDevice: String = Op.mergeDevice(device, updatedContext.device)
    updatedContext = updatedContext.copy(device = newDevice)
    val newDeviceFunction: OpSpecification => String = Op.mergeDeviceFunction(
      deviceFunction, updatedContext.deviceFunction, updatedContext.device)
    updatedContext = updatedContext.copy(deviceFunction = newDeviceFunction)
    context.withValue(updatedContext)(block)
  }

  // TODO: There is a lot of code duplicated between here and the variables package.

  private[api] def createWithVariableScope[R](
      name: String, reuse: ReuseAllowed = ReuseOrCreateNew, dataType: DataType = null, initializer: Initializer = null,
      regularizer: Regularizer = null, partitioner: Partitioner = null, cachingDevice: OpSpecification => String = null,
      customGetter: VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false
  )(block: => R)(implicit context: DynamicVariable[LayerCreationContext]): R = {
    if (reuse == ReuseExistingOnly && isDefaultName)
      throw new IllegalArgumentException(
        "'reuse' cannot be set to 'ReuseExistingOnly' with 'isDefaultName' set to 'true'.")
    val variableScope = context.value.variableScope
    val newName = {
      val uniqueName = if (isDefaultName) variableStore.uniqueVariableScope(name) else name
      if (variableScope.name != null && variableScope.name != "")
        s"${variableScope.name}/$uniqueName"
      else
        uniqueName
    }
    variableStore.enterVariableScope(variableScope.name)
    val newVariableScope = VariableScope(
      // TODO: !!! [VARIABLES] Have 'name' as first argument in order to be consistent.
      reuse = if (reuse == ReuseOrCreateNew) variableScope.reuse else reuse,
      name = newName,
      dataType = if (dataType == null) variableScope.dataType else dataType,
      initializer = if (initializer == null) variableScope.initializer else initializer,
      regularizer = if (regularizer == null) variableScope.regularizer else regularizer,
      partitioner = if (partitioner == null) variableScope.partitioner else partitioner,
      cachingDevice = if (cachingDevice == null) variableScope.cachingDevice else cachingDevice,
      nameScope = name,
      customGetter = {
        if (customGetter == null)
          variableScope.customGetter
        else
          maybeWrapCustomVariableGetter(customGetter, variableScope.customGetter)
      })
    val result = {
      if (isPure)
        context.withValue(context.value.copy(variableScope = newVariableScope))(block)
      else
        Layer.createWith(nameScope = name) {
          context.withValue(context.value.copy(variableScope = newVariableScope))(block)
        }
    }
    variableStore.closeVariableSubScopes(variableScope.name)
    result
  }

  private[api] def createWithUpdatedVariableScope[R](
      variableScope: VariableScope, reuse: ReuseAllowed = ReuseOrCreateNew, dataType: DataType = null,
      initializer: Initializer = null, regularizer: Regularizer = null, partitioner: Partitioner = null,
      cachingDevice: OpSpecification => String = null, customGetter: VariableGetter = null, isPure: Boolean = false
  )(block: => R)(implicit context: DynamicVariable[LayerCreationContext]): R = {
    val subScopeCounts = variableStore.getVariableSubScopeCounts(variableScope.name)
    variableStore.enterVariableScope(variableScope.name)
    val newVariableScope = VariableScope(
      reuse = if (reuse == ReuseOrCreateNew) variableScope.reuse else reuse,
      name = variableScope.name,
      dataType = if (dataType == null) variableScope.dataType else dataType,
      initializer = if (initializer == null) variableScope.initializer else initializer,
      regularizer = if (regularizer == null) variableScope.regularizer else regularizer,
      partitioner = if (partitioner == null) variableScope.partitioner else partitioner,
      cachingDevice = if (cachingDevice == null) variableScope.cachingDevice else cachingDevice,
      nameScope = variableScope.nameScope,
      customGetter = {
        if (customGetter == null)
          variableScope.customGetter
        else
          maybeWrapCustomVariableGetter(customGetter, variableScope.customGetter)
      })
    val result = {
      if (isPure)
        context.withValue(context.value.copy(variableScope = newVariableScope))(block)
      else
        context.withValue(context.value.copy(
          nameScope = variableScope.name.split("/").last, variableScope = newVariableScope))(block)
    }
    variableStore.closeVariableSubScopes(variableScope.name)
    variableStore.setVariableScopeCounts(subScopeCounts)
    result
  }
}
