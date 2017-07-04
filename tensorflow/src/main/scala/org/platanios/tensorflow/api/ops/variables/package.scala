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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, FLOAT32}

/**
  * @author Emmanouil Antonios Platanios
  */
package object variables {
  private[api] trait API {
    type Variable = variables.Variable
    type PartitionedVariable = variables.PartitionedVariable
    type VariableGetter = variables.Variable.VariableGetter
    type VariableInitializer = variables.Initializer
    type VariableRegularizer = variables.Regularizer
    type VariablePartitioner = variables.Partitioner
    type VariableStore = variables.VariableStore
    type VariableScope = variables.VariableScope

    val Variable      = variables.Variable
    val VariableStore = variables.VariableStore
    val VariableScope = variables.VariableScope

    val zerosInitializer = variables.ZerosInitializer
    val onesInitializer  = variables.OnesInitializer

    def constantInitializer(value: Tensor) = variables.ConstantInitializer(value)
    def constantInitializer(value: Output[DataType]) = variables.DynamicConstantInitializer(value)

    type Saver = variables.Saver
    val Saver = variables.Saver

    // TODO: !!! [VARIABLES] Change the order of the arguments in the following functions.

    def variable(
        name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, trainable: Boolean = true, reuse: java.lang.Boolean = null,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): Variable = {
      Variable.getVariable(
        name, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    }

    def partitionedVariable(
        name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, trainable: Boolean = true,
        reuse: java.lang.Boolean = null, collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): PartitionedVariable = {
      Variable.getPartitionedVariable(
        name, shape, dataType, initializer, regularizer, partitioner, trainable, reuse, collections, cachingDevice)
    }

    def localVariable(
        name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, reuse: java.lang.Boolean = null,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): Variable = {
      Variable.getLocalVariable(name, shape, dataType, initializer, regularizer, reuse, collections, cachingDevice)
    }

    def localPartitionedVariable(
        name: String, shape: Shape = null, dataType: DataType = FLOAT32, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, partitioner: VariablePartitioner, reuse: java.lang.Boolean = null,
        collections: Set[Graph.Key[Variable]] = Set.empty,
        cachingDevice: OpSpecification => String = null): PartitionedVariable = {
      Variable.getLocalPartitionedVariable(
        name, shape, dataType, initializer, regularizer, partitioner, reuse, collections, cachingDevice)
    }
  }
}
