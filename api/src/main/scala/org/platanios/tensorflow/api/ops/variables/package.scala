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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.variables.Saver.{V2, WriterVersion}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.DataType

/**
  * @author Emmanouil Antonios Platanios
  */
package object variables {
  private[ops] trait API {
    type Variable = variables.Variable
    type PartitionedVariable = variables.PartitionedVariable
    type VariableReuse = variables.Reuse
    type VariableReuseAllowed = variables.ReuseAllowed
    type VariableGetter = variables.Variable.VariableGetter
    type VariableInitializer = variables.Initializer
    type VariableRegularizer = variables.Regularizer
    type VariablePartitioner = variables.Partitioner
    type VariableStore = variables.VariableStore
    type VariableScope = variables.VariableScope

    val ReuseExistingVariableOnly: variables.ReuseExistingOnly.type = variables.ReuseExistingOnly
    val CreateNewVariableOnly    : variables.CreateNewOnly.type     = variables.CreateNewOnly
    val ReuseOrCreateNewVariable : variables.ReuseOrCreateNew.type  = variables.ReuseOrCreateNew
    val VariableStore            : variables.VariableStore.type     = variables.VariableStore
    val VariableScope            : variables.VariableScope.type     = variables.VariableScope

    val ZerosInitializer: variables.ZerosInitializer.type = variables.ZerosInitializer
    val OnesInitializer : variables.OnesInitializer.type  = variables.OnesInitializer
    def ConstantInitializer(value: Tensor): variables.Initializer = variables.ConstantInitializer(value)
    def ConstantInitializer(value: Output): variables.Initializer = variables.DynamicConstantInitializer(value)
    val RandomUniformInitializer        : variables.RandomUniformInitializer.type         = variables.RandomUniformInitializer
    val RandomNormalInitializer         : variables.RandomNormalInitializer.type          = variables.RandomNormalInitializer
    val RandomTruncatedNormalInitializer: variables.RandomTruncatedNormalInitializer.type = variables.RandomTruncatedNormalInitializer
    val VarianceScalingInitializer      : variables.VarianceScalingInitializer.type       = variables.VarianceScalingInitializer
    val GlorotUniformInitializer        : variables.GlorotUniformInitializer.type         = variables.GlorotUniformInitializer
    val GlorotNormalInitializer         : variables.GlorotNormalInitializer.type          = variables.GlorotNormalInitializer

    type Saver = variables.Saver
    val Saver: variables.Saver.type = variables.Saver

    def saver(
        saveables: Set[Saveable] = null, reshape: Boolean = false, sharded: Boolean = false, maxToKeep: Int = 5,
        keepCheckpointEveryNHours: Float = 10000.0f, restoreSequentially: Boolean = false, filename: String = "model",
        builder: SaverDefBuilder = DefaultSaverDefBuilder, allowEmpty: Boolean = false,
        writerVersion: WriterVersion = V2, saveRelativePaths: Boolean = false, padGlobalStep: Boolean = false,
        name: String = "Saver"): Saver = {
      Saver(
        saveables, reshape, sharded, maxToKeep, keepCheckpointEveryNHours, restoreSequentially, filename, builder,
        allowEmpty, writerVersion, saveRelativePaths, padGlobalStep, name)
    }

    def variable(
        name: String, dataType: DataType = null, shape: Shape = null, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): Variable = {
      Variable.getVariable(
        name, dataType, shape, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    }

    def partitionedVariable(
        name: String, dataType: DataType = null, shape: Shape = null, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, trainable: Boolean = true,
        reuse: Reuse = ReuseOrCreateNew, collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): PartitionedVariable = {
      Variable.getPartitionedVariable(
        name, dataType, shape, initializer, regularizer, partitioner, trainable, reuse, collections, cachingDevice)
    }

    def localVariable(
        name: String, dataType: DataType = null, shape: Shape = null, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): Variable = {
      Variable.getLocalVariable(name, dataType, shape, initializer, regularizer, reuse, collections, cachingDevice)
    }

    def localPartitionedVariable(
        name: String, dataType: DataType = null, shape: Shape = null, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): PartitionedVariable = {
      Variable.getLocalPartitionedVariable(
        name, dataType, shape, initializer, regularizer, partitioner, reuse, collections, cachingDevice)
    }

    def createWithVariableScope[R](
        name: String, reuse: VariableReuse = ReuseOrCreateNewVariable, dataType: DataType = null,
        initializer: VariableInitializer = null, regularizer: VariableRegularizer = null,
        partitioner: VariablePartitioner = null, cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false)
        (block: => R): R = {
      variables.VariableScope.createWithVariableScope(
        name, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter, isDefaultName,
        isPure)(block)
    }

    def createWithUpdatedVariableScope[R](
        variableScope: VariableScope, reuse: VariableReuse = ReuseOrCreateNewVariable, dataType: DataType = null,
        initializer: VariableInitializer = null, regularizer: VariableRegularizer = null,
        partitioner: VariablePartitioner = null, cachingDevice: OpSpecification => String = null,
        customGetter: VariableGetter = null, isPure: Boolean = false)(block: => R): R = {
      variables.VariableScope.createWithUpdatedVariableScope(
        variableScope, reuse, dataType, initializer, regularizer, partitioner, cachingDevice, customGetter,
        isPure)(block)
    }
  }
}
