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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.learn._
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.ops.variables.VariableScope.maybeWrapCustomVariableGetter
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}
import org.platanios.tensorflow.api.ops.variables._
import org.platanios.tensorflow.api.types.DataType

import scala.collection.generic.CanBuildFrom
import scala.collection.{TraversableLike, mutable}
import scala.util.DynamicVariable

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Layer[T, R](
    protected val name: String
)(implicit
    context: DynamicVariable[LayerCreationContext]
) {
  val uniquifiedName: String = Layer.uniqueName(name)

  val layerType: String

  def forward(input: T, mode: Mode): LayerInstance[T, R]

  def apply(input: T, mode: Mode): LayerInstance[T, R] = Op.createWith(
    nameScope = context.value.nameScope,
    device = context.value.device,
    deviceFunction = context.value.deviceFunction
  ) {
    forward(input, mode)
  }

  def >>[S](other: Layer[R, S]): Compose[T, R, S] = compose(other)

  def +(other: Layer[T, R]): Concatenate[T, R] = concatenate(other)

  def ++(others: Seq[Layer[T, R]]): Concatenate[T, R] = concatenate(others: _*)

  def compose[S](other: Layer[R, S]): Compose[T, R, S] = Compose(this, other)

  def concatenate(others: Layer[T, R]*): Concatenate[T, R] = Concatenate(this +: others)

  protected def variable(
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null): Variable = {
    context.value.variableScope.getVariable(
      Op.currentVariableStore, name, dataType, shape, initializer, regularizer, trainable, reuse, collections,
      cachingDevice)
  }

  protected def partitionedVariable(
      name: String, dataType: DataType = null, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, partitioner: Partitioner = null, trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew, collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): PartitionedVariable = {
    context.value.variableScope.getPartitionedVariable(
      Op.currentVariableStore, name, dataType, shape, initializer, regularizer, partitioner, trainable, reuse,
      collections, cachingDevice)
  }

  override def toString: String = s"$uniquifiedName[$layerType]"
}

private[api] final case class LayerCreationContext(
    nameScope: String = "", variableScope: VariableScope = VariableScope(reuse = ReuseOrCreateNew),
    device: String = "", deviceFunction: OpSpecification => String = _.device)

object Layer {
  trait API {
    def currentLayerNameScope: String = Layer.currentNameScope
    def currentLayerVariableScope: VariableScope = Layer.currentVariableScope
    def currentLayerDevice: String = Layer.currentDevice
    def currentLayerDeviceFunction: OpSpecification => String = Layer.currentDeviceFunction

    def createLayersWith[R](
        nameScope: String = null,
        device: String = "",
        deviceFunction: OpSpecification => String = _.device,
    )(block: => R): R = {
      Layer.createWith(nameScope, device, deviceFunction)(block)
    }

    def createLayersWithVariableScope[R](
        name: String, reuse: ReuseAllowed = ReuseOrCreateNew, dataType: DataType = null,
        initializer: Initializer = null, regularizer: Regularizer = null, partitioner: Partitioner = null,
        cachingDevice: OpSpecification => String = null, customGetter: VariableGetter = null,
        isDefaultName: Boolean = false, isPure: Boolean = false
    )(block: => R): R = {
      Layer.createWithVariableScope(
        name, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter, isPure)(block)
    }

    def createLayersWithUpdatedVariableScope[R](
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

  /** Set that contains the current layer names in use. */
  private[this] val namesInUse: mutable.Set[String] = mutable.Set.empty[String]

  /** Marks `name` as a used layer name (i.e., increments its usage counter). */
  private[layers] def markNameAsUsed(name: String): Unit = namesInUse synchronized {
    namesInUse += name
  }

  /** Returns a unique layer name, based on the provided `name`.
    *
    * @param  name       Name in which to base the generated unique name.
    * @param  markAsUsed If `true`, which is the default, a new unique name is created and marked as in use. If `false`,
    *                    the unique name is returned without actually being marked as used. This is useful when the
    *                    caller simply wants to know what the name to be created will be.
    * @return Unique name.
    */
  private[layers] def uniqueName(name: String, markAsUsed: Boolean = true): String = namesInUse synchronized {
    val nameScope = Op.convertNameScopeToName(Layer.currentNameScope)
    val fullName = {
      if (nameScope == null || nameScope == "")
        name
      else
        s"$nameScope/$name"
    }
    var count = if (namesInUse.contains(fullName)) 1 else 0
    // Increment the counter for the provided name.
    if (markAsUsed)
      namesInUse += fullName
    if (count > 0) {
      var uniqueName = fullName
      // Make sure the composed name is not already being used.
      while (namesInUse.contains(uniqueName)) {
        uniqueName = s"${fullName}_$count"
        count += 1
      }
      // Mark the composed name as used.
      if (markAsUsed)
        namesInUse += uniqueName
      uniqueName
    } else {
      fullName
    }
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
    val newNameScope: String = Op.mergeNameScope(nameScope, updatedContext.nameScope, Layer.uniqueName(_))
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

  private[layers] trait Implicits {
    implicit class MappableLayer[T, R, CC[A] <: TraversableLike[A, CC[A]]] private[learn](
        layer: Layer[CC[T], CC[R]]
    ) extends Layer[CC[T], CC[R]]("Mappable") {
      override val layerType: String = "Mappable"

      override def forward(input: CC[T], mode: Mode): LayerInstance[CC[T], CC[R]] = {
        layer.forward(input, mode)
      }

      def map[S](
          layer: Layer[CC[T], CC[R]],
          mapLayer: Layer[R, S]
      )(implicit
          cbfSS: CanBuildFrom[CC[LayerInstance[R, S]], S, CC[S]],
          cbfLIRS: CanBuildFrom[CC[R], LayerInstance[R, S], CC[LayerInstance[R, S]]]
      ): layers.Map[T, R, S, CC] = {
        layers.Map[T, R, S, CC](layer, mapLayer)(cbfSS, cbfLIRS)
      }
    }
  }
}
