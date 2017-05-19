// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait OpAPI
    extends Basic
        with ControlFlow
        with Logging
        with Math
        with Text
        with variables.VariableAPI {
  type Op = ops.Op
  val Op = ops.Op

  type OpCreationContext = ops.OpCreationContext
  type OpSpecification = ops.OpSpecification

  val Gradients         = ops.Gradients
  val GradientsRegistry = ops.Gradients.Registry

  ops.Basic.Gradients
  ops.Math.Gradients
  ops.variables.Variable.Gradients

  //region Op Construction Aliases

  def currentGraph: Graph = Op.currentGraph
  def currentNameScope: String = Op.currentNameScope
  def currentVariableScope: VariableScope = Op.currentVariableScope
  def currentDevice: OpSpecification => String = Op.currentDevice
  def currentColocationOps: Set[Op] = Op.currentColocationOps
  def currentControlDependencies: Set[Op] = Op.currentControlDependencies
  def currentAttributes: Map[String, Any] = Op.currentAttributes
  def currentContainer: String = Op.currentContainer

  // TODO: Maybe remove "current" from the above names.

  def globalVariablesInitializer(name: String = "GlobalVariablesInitializer"): Op = {
    Op.currentGraph.globalVariablesInitializer(name)
  }

  def localVariablesInitializer(name: String = "LocalVariablesInitializer"): Op = {
    Op.currentGraph.localVariablesInitializer(name)
  }

  def modelVariablesInitializer(name: String = "ModelVariablesInitializer"): Op = {
    Op.currentGraph.modelVariablesInitializer(name)
  }

  def trainableVariablesInitializer(name: String = "TrainableVariablesInitializer"): Op = {
    Op.currentGraph.trainableVariablesInitializer(name)
  }

  def createWith[R](
      graph: Graph = null, nameScope: String = null, device: OpSpecification => String = _ => "",
      colocationOps: Set[Op] = null, controlDependencies: Set[Op] = null, attributes: Map[String, Any] = null,
      container: String = null)(block: => R): R = {
    Op.createWith(graph, nameScope, device, colocationOps, controlDependencies, attributes, container)(block)
  }

  def createWithNameScope[R](nameScope: String, values: Set[Op] = Set.empty[Op])(block: => R): R = {
    Op.createWithNameScope(nameScope, values)(block)
  }

  def createWithVariableScope[R](
      name: String, reuse: java.lang.Boolean = null, dataType: DataType = null,
      initializer: VariableInitializer = null, regularizer: VariableRegularizer = null,
      partitioner: VariablePartitioner = null, cachingDevice: OpSpecification => String = null,
      customGetter: VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false)
      (block: => R): R = {
    variables.VariableScope.createWithVariableScope(
      name, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter, isDefaultName,
      isPure)(block)
  }

  def createWithUpdatedVariableScope[R](
      variableScope: VariableScope, reuse: java.lang.Boolean = null, dataType: DataType = null,
      initializer: VariableInitializer = null, regularizer: VariableRegularizer = null,
      partitioner: VariablePartitioner = null, cachingDevice: OpSpecification => String = null,
      customGetter: VariableGetter = null, isPure: Boolean = false)(block: => R): R = {
    variables.VariableScope.createWithUpdatedVariableScope(
      variableScope, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter,
      isPure)(block)
  }

  def colocateWith[R](colocationOps: Set[Op], ignoreExisting: Boolean = false)(block: => R): R = {
    Op.colocateWith(colocationOps, ignoreExisting)(block)
  }

  object train extends training.TrainingAPI
}
