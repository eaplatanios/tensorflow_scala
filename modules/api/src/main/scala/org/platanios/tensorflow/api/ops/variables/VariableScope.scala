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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.types.{DataType, TF}

/** Variable scope that carries default settings to provide to `getVariable`.
  *
  * A variable scope allows to create new variables and to share already created ones while providing checks to not
  * create or share by accident.
  *
  * Many of the arguments we need for `getVariable` in a variable store are most easily handled with a context.
  * [[VariableScope]] objects are used for the defaults.
  *
  * @param  reuse            [[Reuse]] value indicating whether to re-use an existing variable with the same name,
  *                          create a new variable, or do either.
  * @param  name             Name of the variable scope, used as a prefix in `getVariable`.
  * @param  initializer      Default initializer passed to `getVariable`.
  * @param  regularizer      Default regularizer passed to `getVariable`.
  * @param  cachingDevice    Default caching device passed to `getVariable`.
  * @param  nameScope        Default name scope passed to `getVariable`.
  * @param  underlyingGetter Default underlying variable getter passed to `getVariable`.
  *
  * @author Emmanouil Antonios Platanios
  */
case class VariableScope private[variables](
    reuse: Reuse,
    name: String = "",
    initializer: Initializer = null,
    regularizer: Regularizer = null,
    cachingDevice: OpSpecification => String = null,
    nameScope: String = "",
    underlyingGetter: VariableGetter = null
) {
  /** Gets an existing variable with the specified name or creates a new one.
    *
    * @param  store         Variable store currently being used to store variables.
    * @param  name          Variable name.
    * @param  shape         Variable shape.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
    *                       to use by the optimizers.
    * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
    *                       a new variable, or do either.
    *                       - If `reuse` is `null` (the default), both new and existing variables are returned.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
    * @tparam T             Variable data type.
    * @return Requested variable.
    * @throws IllegalArgumentException If any of the provided arguments are not compatible with each other, or with the
    *                                  variables stored in this variable store.
    * @throws ShapeMismatchException   If the provided shape does not match the shape of the corresponding variable
    *                                  stored in this variable store (if there exists one).
    * @throws InvalidDataTypeException If the provided data type does not match the data type of the corresponding
    *                                  variable stored in this variable store (if there exists one).
    */
  @throws[IllegalArgumentException]
  @throws[ShapeMismatchException]
  @throws[InvalidDataTypeException]
  def getVariable[T: TF](
      store: VariableStore,
      name: String,
      shape: Shape,
      initializer: Initializer = this.initializer,
      regularizer: Regularizer = this.regularizer,
      trainable: Boolean = true,
      reuse: Reuse = this.reuse,
      collections: Set[Graph.Key[Variable[Any]]] = Set.empty,
      cachingDevice: OpSpecification => String = this.cachingDevice
  ): Variable[T] = {
    val fullName = {
      if (this.name != null && this.name != "")
        s"${this.name}/$name"
      else
        name
    }
    // Variable names only depend on the variable scope and not the name scope, so we reset it below for the time of
    // variable creation.
    Op.nameScope("") {
      store.getVariable(
        fullName, shape, initializer, regularizer,
        trainable, reuse, collections, cachingDevice)
    }
  }
}

private[api] object VariableScope {
  /** Returns the current variable scope. */
  def current: VariableScope = {
    VariableScopeStore.current.scope
  }

  /** Sets the variable scope to use for op creation context, for all code in `block`.
    *
    * @param  name             Variable scope name, that may also change the name scope of the op creation context,
    *                          depending on the value of `isPure`.
    * @param  reuse            [[Reuse]] value indicating whether to re-use an existing variable with the same name, or
    *                          do either. Note that this argument cannot be set to [[CreateNewOnly]] in this function.
    *                          If set to [[ReuseOrCreateNew]], then the parent variable scope `reuse` value is used
    *                          (i.e., propagated).
    * @param  initializer      Default initializer for variables within the scope.
    * @param  regularizer      Default regularizer for variables within the scope.
    * @param  cachingDevice    Default caching device for variables within the scope.
    * @param  underlyingGetter Default variable getter for variables within the scope.
    * @param  isDefaultName    Boolean value indicating whether `name` is a default name or not. If `true`, then `name`
    *                          will be made unique before being used. `isDefaultName` cannot be set to `true` when
    *                          `reuse` is set to [[ReuseExistingOnly]].
    * @param  isPure           Boolean value indicating whether to use a "pure" variable scope. That is, a variable
    *                          scope that does not affect the name scope of the current op creation context.
    * @param  block            Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def scope[R](
      name: String,
      reuse: Reuse = ReuseOrCreateNew,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      cachingDevice: OpSpecification => String = null,
      underlyingGetter: VariableGetter = null,
      isDefaultName: Boolean = false,
      isPure: Boolean = false
  )(block: => R): R = {
    if (reuse == ReuseExistingOnly && isDefaultName)
      throw new IllegalArgumentException(
        "'reuse' cannot be set to 'ReuseExistingOnly' with 'isDefaultName' set to 'true'.")
    val variableScopeStore = VariableScopeStore.current
    val oldVariableScope = variableScopeStore.scope
    val newName = {
      val uniqueName = if (isDefaultName) VariableScope.unique(name) else name
      if (oldVariableScope.name != null && oldVariableScope.name != "")
        s"${oldVariableScope.name}/$uniqueName"
      else
        uniqueName
    }
    variableScopeStore.enterVariableScope(newName)
    val newVariableScope = VariableScope(
      reuse = if (reuse == ReuseOrCreateNew) oldVariableScope.reuse else reuse,
      name = newName,
      initializer = if (initializer == null) oldVariableScope.initializer else initializer,
      regularizer = if (regularizer == null) oldVariableScope.regularizer else regularizer,
      cachingDevice = if (cachingDevice == null) oldVariableScope.cachingDevice else cachingDevice,
      nameScope = name,
      underlyingGetter = {
        if (underlyingGetter == null)
          oldVariableScope.underlyingGetter
        else
          maybeWrapCustomVariableGetter(underlyingGetter, oldVariableScope.underlyingGetter)
      })
    variableScopeStore.scope = newVariableScope
    val result = if (isPure) block else Op.nameScope(name)(block)
    variableScopeStore.closeVariableSubScopes(newName)
    variableScopeStore.scope = oldVariableScope
    result
  }

  /** Sets the variable scope to use for op creation context, for all code in `block`.
    *
    * @param  variableScope    Default variable scope to use. Other arguments of this function can override the
    *                          corresponding parameters of `variableScope`.
    * @param  reuse            [[Reuse]] value indicating whether to re-use an existing variable with the same name, or
    *                          do either. Note that this argument cannot be set to [[CreateNewOnly]] in this function.
    *                          If set to [[ReuseOrCreateNew]], then the parent variable scope `reuse` value is used
    *                          (i.e., propagated).
    * @param  initializer      Default initializer for variables within the scope.
    * @param  regularizer      Default regularizer for variables within the scope.
    * @param  cachingDevice    Default caching device for variables within the scope.
    * @param  underlyingGetter Default variable getter for variables within the scope.
    * @param  isPure           Boolean value indicating whether to use a "pure" variable scope. That is, a variable
    *                          scope that does not affect the name scope of the current op creation context.
    * @param  block            Code block to run using the provided options.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def updatedScope[R](
      variableScope: VariableScope = VariableScope.current,
      reuse: Reuse = ReuseOrCreateNew,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      cachingDevice: OpSpecification => String = null,
      underlyingGetter: VariableGetter = null,
      isPure: Boolean = false
  )(block: => R): R = {
    val variableScopeStore = VariableScopeStore.current
    val oldVariableScope = variableScopeStore.scope
    val oldVariableScopeCounts = variableScopeStore.variableScopeCounts
    variableScopeStore.enterVariableScope(variableScope.name)
    val newVariableScope = VariableScope(
      reuse = if (reuse == ReuseOrCreateNew) variableScope.reuse else reuse,
      name = variableScope.name,
      initializer = if (initializer == null) variableScope.initializer else initializer,
      regularizer = if (regularizer == null) variableScope.regularizer else regularizer,
      cachingDevice = if (cachingDevice == null) variableScope.cachingDevice else cachingDevice,
      nameScope = variableScope.nameScope,
      underlyingGetter = {
        if (underlyingGetter == null)
          variableScope.underlyingGetter
        else
          maybeWrapCustomVariableGetter(underlyingGetter, variableScope.underlyingGetter)
      })
    variableScopeStore.scope = newVariableScope
    val result = if (isPure) block else Op.nameScope(variableScope.name.split("/").last)(block)
    variableScopeStore.closeVariableSubScopes(variableScope.name)
    variableScopeStore.variableScopeCounts = oldVariableScopeCounts
    variableScopeStore.scope = oldVariableScope
    result
  }

  /** If a new getter is provided, it wraps around the old one and the new wrapped getter is returned. Otherwise, the
    * old getter is returned.
    *
    * @param  getter    New variable getter.
    * @param  oldGetter Old variable getter.
    * @return Variable getter to use.
    */
  private[api] def maybeWrapCustomVariableGetter(
      getter: VariableGetter,
      oldGetter: VariableGetter
  ): VariableGetter = {
    if (getter == null) {
      oldGetter
    } else {
      new VariableGetter {
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
          val baseGetter: VariableGetter = new VariableGetter {
            override def apply[R: TF](
                name: String,
                dataType: DataType[R],
                shape: Shape,
                initializer: Initializer,
                regularizer: Regularizer,
                trainable: Boolean,
                reuse: Reuse,
                collections: Set[Graph.Key[Variable[Any]]],
                cachingDevice: OpSpecification => String,
                underlyingGetter: VariableGetter
            ): Variable[R] = {
              oldGetter(
                name, dataType, shape, initializer, regularizer, trainable,
                reuse, collections, cachingDevice, underlyingGetter)
            }
          }
          getter(
            name, dataType, shape, initializer, regularizer, trainable,
            reuse, collections, cachingDevice, baseGetter)
        }
      }
    }
  }

  /** Gets a name with the provided prefix that is unique in the current variable scope.
    *
    * @param  prefix Prefix.
    * @return Unique name with the provided prefix.
    */
  private[api] def unique(prefix: String): String = {
    val currentScopeStore = VariableScopeStore.current
    val currentScope = Op.convertNameScopeToName(VariableScope.current.name)
    val name = {
      if (currentScope == null || currentScope == "")
        prefix
      else
        s"$currentScope/$prefix"
    }
    if (currentScopeStore.variableScopeCount(name) == 0) {
      prefix
    } else {
      var uniqueName = name
      var count = 1
      while (currentScopeStore.variableScopeCount(uniqueName) > 0) {
        uniqueName = s"${name}_$count"
        count += 1
      }
      uniqueName
    }
  }
}
