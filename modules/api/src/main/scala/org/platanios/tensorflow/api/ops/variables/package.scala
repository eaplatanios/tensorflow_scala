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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.types.{DataType, TF}
import org.platanios.tensorflow.api.ops.variables.Saver.{V2, WriterVersion}
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.tensors.Tensor

// TODO: [VARIABLES/DOC] Create a documentation page for the Scala API (https://www.tensorflow.org/programmers_guide/variables).
// TODO: [VARIABLES/DOC/EXAMPLES] Examples.
// TODO: !!! [VARIABLES] Look into the cached pure variable scope.
// TODO: [VARIABLES] Make reads optional in the assignment ops.
// TODO: [VARIABLES] Add support for slice assignment.
// TODO: [VARIABLES] Unify the interfaces of variables and partitioned variables.
// TODO: [VARIABLES] Support partitioned variables in getters.
// TODO: [VARIABLES/INITIALIZERS] UniformUnitScaling and orthogonal.

/**
  * @author Emmanouil Antonios Platanios
  */
package object variables {
  private[ops] trait API {
    type VariableLike[T] = variables.VariableLike[T]
    type Variable[T] = variables.Variable[T]
    type VariableReuse = variables.Reuse
    type VariableReuseAllowed = variables.ReuseAllowed
    type VariableGetter = variables.Variable.VariableGetter
    type VariableInitializer = variables.Initializer
    type VariableRegularizer = variables.Regularizer
    type VariableStore = variables.VariableStore
    type VariableScope = variables.VariableScope

    val ReuseExistingVariableOnly: variables.ReuseExistingOnly.type = variables.ReuseExistingOnly
    val CreateNewVariableOnly    : variables.CreateNewOnly.type     = variables.CreateNewOnly
    val ReuseOrCreateNewVariable : variables.ReuseOrCreateNew.type  = variables.ReuseOrCreateNew
    val VariableStore            : variables.VariableStore.type     = variables.VariableStore
    val VariableScope            : variables.VariableScope.type     = variables.VariableScope

    val ZerosInitializer: variables.ZerosInitializer.type = variables.ZerosInitializer
    val OnesInitializer : variables.OnesInitializer.type  = variables.OnesInitializer
    def ConstantInitializer[T: TF](value: Tensor[T]): variables.Initializer = variables.ConstantInitializer(value)
    def ConstantInitializer[T: TF](value: Output[T]): variables.Initializer = variables.DynamicConstantInitializer(value)
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
        name: String = "Saver"
    ): Saver = {
      Saver(
        saveables, reshape, sharded, maxToKeep, keepCheckpointEveryNHours, restoreSequentially, filename, builder,
        allowEmpty, writerVersion, saveRelativePaths, padGlobalStep, name)
    }

    def variable[T: TF](
        name: String, shape: Shape = null, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable[Any]]] = Set.empty,
        cachingDevice: OpSpecification => String = null
    ): Variable[T] = {
      Variable.getVariable(
        name, shape, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    }

    def localVariable[T: TF](
        name: String, shape: Shape = null, initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null, reuse: Reuse = ReuseOrCreateNew,
        collections: Set[Graph.Key[Variable[Any]]] = Set.empty,
        cachingDevice: OpSpecification => String = null
    ): Variable[T] = {
      Variable.getLocalVariable(name, shape, initializer, regularizer, reuse, collections, cachingDevice)
    }

    def variableScope[R](
        name: String,
        reuse: VariableReuse = ReuseOrCreateNewVariable,
        initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null,
        cachingDevice: OpSpecification => String = null,
        underlyingGetter: VariableGetter = null,
        isDefaultName: Boolean = false,
        isPure: Boolean = false
    )(block: => R): R = {
      variables.VariableScope.scope(
        name, reuse, initializer, regularizer, cachingDevice, underlyingGetter, isDefaultName,
        isPure)(block)
    }

    def updatedVariableScope[R](
        variableScope: VariableScope = VariableScope.current,
        reuse: VariableReuse = ReuseOrCreateNewVariable,
        initializer: VariableInitializer = null,
        regularizer: VariableRegularizer = null,
        cachingDevice: OpSpecification => String = null,
        underlyingGetter: VariableGetter = null,
        isPure: Boolean = false
    )(block: => R): R = {
      variables.VariableScope.updatedScope(
        variableScope, reuse, initializer, regularizer, cachingDevice, underlyingGetter,
        isPure)(block)
    }

    /** Adds `getter` to the scope that `block` is executed in. */
    def variableGetter[R](getter: VariableGetter)(block: => R): R = Variable.getter(getter)(block)

    /** Returns the variable getters in the current scope. */
    def currentVariableGetters: Seq[VariableGetter] = Variable.currentGetters

    /** Returns the variable scope in the current scope. */
    def currentVariableScope: VariableScope = VariableScope.current

    /** Returns the variable store in the current scope. */
    def currentVariableStore: VariableStore = VariableStore.current
  }

  /** Returns a default variable initializer.
    *
    * @param  name     Variable name.
    * @param  dataType Variable data type.
    * @return Default initializer.
    * @throws IllegalArgumentException If no default initializer is defined for the specified data type.
    */
  @throws[IllegalArgumentException]
  def defaultInitializer(
      name: String,
      dataType: DataType[Any]
  ): Initializer = {
    if (dataType.isFloatingPoint)
      GlorotUniformInitializer()
    else if (dataType.isInteger || dataType.isUnsigned || dataType.isBoolean)
      ZerosInitializer
    else
      throw new IllegalArgumentException(s"A default initializer for variable '$name' of type '$dataType' is required.")
  }

  /** This function defines the main logic of 'getVariable'. However, 'underlyingGetter' may override this logic.
    * That is why we pass it as an argument to the 'underlyingGetter'. */
  val defaultGetter: VariableGetter = new VariableGetter {
    override def apply[T: TF](
        name: String,
        dataType: DataType[T],
        shape: Shape,
        initializer: Initializer,
        regularizer: Regularizer,
        trainable: Boolean,
        reuse: Reuse,
        collections: Set[Graph.Key[Variable[Any]]],
        cachingDevice: OpSpecification => String,
        underlyingGetter: VariableGetter
    ): Variable[T] = {
      val actualInitializer = Op.initializationScope {
        if (initializer == null)
          defaultInitializer(name, dataType)
        else
          initializer
      }
      Variable(actualInitializer, dataType, shape, trainable, collections, cachingDevice, name)
    }
  }

  private[variables] def makeGetter(): VariableGetter = {
    var currentGetter = defaultGetter
    Op.currentGraph.variableGetters.value.foreach(g => {
      currentGetter = new VariableGetter {
        override def apply[T: TF](
            name: String,
            dataType: DataType[T],
            shape: Shape,
            initializer: Initializer,
            regularizer: Regularizer,
            trainable: Boolean,
            reuse: Reuse,
            collections: Set[Graph.Key[Variable[Any]]],
            cachingDevice: OpSpecification => String,
            underlyingGetter: VariableGetter
        ): Variable[T] = {
          g(
            name, dataType, shape, initializer, regularizer, trainable, reuse, collections,
            cachingDevice, currentGetter)
        }
      }
    })
    currentGetter
  }
}
