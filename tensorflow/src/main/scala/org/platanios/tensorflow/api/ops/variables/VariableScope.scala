package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.ops.{Op, OpCreationContext, OpSpecification}
import org.platanios.tensorflow.api.ops.variables.Variable.VariableGetter
import org.platanios.tensorflow.api.types.DataType

import scala.util.DynamicVariable

// TODO: [VARIABLES] Resolve customGetter weirdness related to partitioned variables.
// TODO: [VARIABLES] Is a custom getter necessary/useful?

/** Variable scope that carries default settings to provide to `getVariable`.
  *
  * A variable scope allows to create new variables and to share already created ones while providing checks to not
  * create or share by accident.
  *
  * TODO: [VARIABLES] [EXAMPLES] Examples.
  *
  * Many of the arguments we need for `getVariable` in a variable store are most easily handled with a context.
  * [[VariableScope]] objects are used for the defaults.
  *
  * @param  reuse         Boolean value indicating whether to reuse existing variables with the same name.
  * @param  name          Name of the variable scope, used as a prefix in `getVariable`.
  * @param  initializer   Default initializer passed to `getVariable`.
  * @param  regularizer   Default regularizer passed to `getVariable`.
  * @param  partitioner   Default partitioner passed to `getVariable`.
  * @param  cachingDevice Default caching device passed to `getVariable`.
  * @param  nameScope     Default name scope passed to `getVariable`.
  * @param  dataType      Default data type passed to `getVariable`.
  * @param  customGetter  Default variable getter passed to `getVariable`.
  *
  * @author Emmanouil Antonios Platanios
  */
case class VariableScope(
    reuse: java.lang.Boolean,
    name: String = "",
    dataType: DataType = null,
    initializer: Initializer = null,
    regularizer: Regularizer = null,
    partitioner: Partitioner = null,
    cachingDevice: OpSpecification => String = null,
    nameScope: String = "",
    customGetter: VariableGetter = null) {
  /** Gets an existing variable with the specified name or creates a new one.
    *
    * @param  store         Variable store currently being used to store variables.
    * @param  name          Variable name.
    * @param  shape         Variable shape.
    * @param  dataType      Variable data type.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
    *                       to use by the optimizers.
    * @param  reuse         Boolean value indicating whether to re-use an existing variable with the same name.
    *                       - Set `reuse` to `true` when you only want to reuse existing variables.
    *                       - Set `reuse` to `false` when you only want to create new variables.
    *                       - If `reuse` is `null` (the default), both new and existing variables are returned.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
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
  private[variables] def getVariable(
      store: VariableStore, name: String, shape: Shape = null, dataType: DataType = this.dataType,
      initializer: Initializer = this.initializer, regularizer: Regularizer = this.regularizer,
      trainable: Boolean = true, reuse: java.lang.Boolean = this.reuse, collections: Set[String] = Set.empty,
      cachingDevice: OpSpecification => String = this.cachingDevice): Variable = {
    val fullName = if (this.name != null && this.name != "") s"${this.name}/$name" else name
    // Variable names only depend on the variable scope and not the name scope, so we reset it below for the time of
    // variable creation.
    Op.createWith(nameScope = "") {
      store.getVariable(
        fullName, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice)
    }
  }

  /** Gets an existing partitioned variable with the specified name or creates a new one.
    *
    * @param  store         Variable store currently being used to store variables.
    * @param  name          Variable name.
    * @param  shape         Variable shape.
    * @param  dataType      Variable data type.
    * @param  initializer   Variable initializer. If `initializer` is `null` (the default), the default initializer
    *                       passed in the constructor is used. If that one is `null` too, then we use a new
    *                       `glorotUniformInitializer`. The initializer will be called for each part of the partitioned
    *                       variable separately.
    * @param  regularizer   Variable regularizer.
    * @param  partitioner   Function that accepts a fully defined `Shape` and returns a sequence of integers (i.e., the
    *                       `partitions`). These integers describe how to partition the given variable, along the each
    *                       dimension. That is, `partitions(1) = 3` means that we split the variable into `3` parts
    *                       along dimension `1`. Currently, partitioning along only a single axis is supported.
    * @param  trainable     If `true`, the default, the variable is added to the graph collection
    *                       `Graph.Keys.TRAINABLE_VARIABLES`. This collection is used as the default set of variables
    *                       to use by the optimizers.
    * @param  reuse         Boolean value indicating whether to re-use an existing variable with the same name.
    *                       - Set `reuse` to `true` when you only want to reuse existing variables.
    *                       - Set `reuse` to `false` when you only want to create new variables.
    *                       - If `reuse` is `null` (the default), both new and existing variables are returned.
    * @param  collections   Set of graph collections keys. The variable is added to these collections. Defaults to
    *                       `Set(Graph.Keys.GLOBAL_VARIABLES)`.
    * @param  cachingDevice Device specification describing where the variable should be cached for reading. Defaults
    *                       to the variable's device. Typical use is to cache on the device where the ops using the
    *                       variable reside, to deduplicate copying through `Switch` and other conditional statements.
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
  private[variables] def getPartitionedVariable(
      store: VariableStore, name: String, shape: Shape = null, dataType: DataType = this.dataType,
      initializer: Initializer = this.initializer, regularizer: Regularizer = this.regularizer,
      partitioner: Partitioner = this.partitioner, trainable: Boolean = true,
      reuse: java.lang.Boolean = this.reuse, collections: Set[String] = Set.empty,
      cachingDevice: OpSpecification => String = this.cachingDevice): PartitionedVariable = {
    if (customGetter != null)
      throw new IllegalArgumentException(
        "Private access to 'getPartitionedVariable' is not allowed when a custom getter is set.")
    if (partitioner == null)
      throw new IllegalArgumentException("No partitioner was specified.")
    val fullName = if (this.name != null && this.name != "") s"${this.name}/$name" else name
    // Variable names only depend on the variable scope and not the name scope, so we reset it below for the time of
    // variable creation.
    Op.createWith(nameScope = "") {
      store.getPartitionedVariable(
        fullName, shape, dataType, initializer, regularizer, partitioner, trainable, reuse, collections, cachingDevice)
    }
  }
}

object VariableScope {
  /** Sets the variable scope to use for op creation context, for all code in `block`.
    *
    * @param  name          Variable scope name, that may also change the name scope of the op creation context,
    *                       depending on the value of `isPure`.
    * @param  reuse         If `true`, we go into variable reuse mode for this scope as well as all sub-scopes. If
    *                       `null`, we just inherit the parent variable scope reuse mode.
    * @param  dataType      Default data type for variables within the scope.
    * @param  initializer   Default initializer for variables within the scope.
    * @param  regularizer   Default regularizer for variables within the scope.
    * @param  partitioner   Default partitioner for variables within the scope.
    * @param  cachingDevice Default caching device for variables within the scope.
    * @param  customGetter  Default variable getter for variables within the scope.
    * @param  isDefaultName Boolean value indicating whether `name` is a default name or not. If `true`, then `name`
    *                       will be made unique before being used. `isDefaultName` and `reuse` cannot both be set to
    *                       `true` at the same time.
    * @param  isPure        Boolean value indicating whether to use a "pure" variable scope. That is, a variable scope
    *                       that does not affect the name scope of the current op creation context.
    * @param  block         Code block to run using the provided options.
    * @param  context       Current op creation context.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def createWithVariableScope[R](
      name: String, reuse: java.lang.Boolean = null, dataType: DataType = null, initializer: Initializer = null,
      regularizer: Regularizer = null, partitioner: Partitioner = null, cachingDevice: OpSpecification => String = null,
      customGetter: VariableGetter = null, isDefaultName: Boolean = false, isPure: Boolean = false)
      (block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    if (reuse != null && reuse && isDefaultName)
      throw new IllegalArgumentException("'reuse' and 'isDefaultName' cannot both be 'false'.")
    val variableStore = context.graph.variableStore
    val variableScope = context.variableScope
    val newName = {
      val uniqueName = if (isDefaultName) variableStore.uniqueVariableScope(name) else name
      if (variableScope.name != null && variableScope.name != "")
        s"${variableScope.name}/$uniqueName"
      else
        uniqueName
    }
    variableStore.enterVariableScope(variableScope.name)
    val newVariableScope = VariableScope(
      reuse = if (reuse == null) variableScope.reuse else reuse,
      name = newName,
      dataType = if (dataType == null) variableScope.dataType else dataType,
      initializer = if (initializer == null) variableScope.initializer else initializer,
      regularizer = if (regularizer == null) variableScope.regularizer else regularizer,
      partitioner = if (partitioner == null) variableScope.partitioner else partitioner,
      cachingDevice = if (cachingDevice == null) variableScope.cachingDevice else cachingDevice,
      nameScope = name,
      customGetter = {
        if (customGetter == null) {
          variableScope.customGetter
        } else {
          maybeWrapCustomVariableGetter(customGetter, variableScope.customGetter)
        }
      })
    val result = {
      if (isPure)
        context.withValue(context.copy(variableScope = newVariableScope))(block)
      else
        context.withValue(context.copy(nameScope = name, variableScope = newVariableScope))(block)
    }
    variableStore.closeVariableSubScopes(variableScope.name)
    result
  }

  /** Sets the variable scope to use for op creation context, for all code in `block`.
    *
    * @param  variableScope Default variable scope to use. Other arguments of this function can override the
    *                       corresponding parameters of `variableScope`.
    * @param  reuse         If `true`, we go into variable reuse mode for this scope as well as all sub-scopes. If
    *                       `null`, we just inherit the parent variable scope reuse mode.
    * @param  dataType      Default data type for variables within the scope.
    * @param  initializer   Default initializer for variables within the scope.
    * @param  regularizer   Default regularizer for variables within the scope.
    * @param  partitioner   Default partitioner for variables within the scope.
    * @param  cachingDevice Default caching device for variables within the scope.
    * @param  customGetter  Default variable getter for variables within the scope.
    * @param  isPure        Boolean value indicating whether to use a "pure" variable scope. That is, a variable scope
    *                       that does not affect the name scope of the current op creation context.
    * @param  block         Code block to run using the provided options.
    * @param  context       Current op creation context.
    * @tparam R Return type of the code block.
    * @return Return value of the code block.
    */
  private[api] def createWithUpdatedVariableScope[R](
      variableScope: VariableScope, reuse: java.lang.Boolean = null, dataType: DataType = null,
      initializer: Initializer = null, regularizer: Regularizer = null, partitioner: Partitioner = null,
      cachingDevice: OpSpecification => String = null, customGetter: VariableGetter = null, isPure: Boolean = false)
      (block: => R)(implicit context: DynamicVariable[OpCreationContext]): R = {
    val variableStore = context.graph.variableStore
    val subScopeCounts = variableStore.getVariableSubScopeCounts(variableScope.name)
    variableStore.enterVariableScope(variableScope.name)
    val newVariableScope = VariableScope(
      reuse = if (reuse == null) variableScope.reuse else reuse,
      name = variableScope.name,
      dataType = if (dataType == null) variableScope.dataType else dataType,
      initializer = if (initializer == null) variableScope.initializer else initializer,
      regularizer = if (regularizer == null) variableScope.regularizer else regularizer,
      partitioner = if (partitioner == null) variableScope.partitioner else partitioner,
      cachingDevice = if (cachingDevice == null) variableScope.cachingDevice else cachingDevice,
      nameScope = variableScope.nameScope,
      customGetter = {
        if (customGetter == null) {
          variableScope.customGetter
        } else {
          maybeWrapCustomVariableGetter(customGetter, variableScope.customGetter)
        }
      })
    val result = {
      if (isPure)
        context.withValue(context.copy(variableScope = newVariableScope))(block)
      else
        context.withValue(context.copy(
          nameScope = variableScope.name.split("/").last, variableScope = newVariableScope))(block)
    }
    variableStore.closeVariableSubScopes(variableScope.name)
    variableStore.setVariableScopeCounts(subScopeCounts)
    result
  }

  /** If a new getter is provided, it wraps around the old one and the new wrapped getter is returned. Otherwise, the
    * old getter is returned.
    *
    * @param  getter    New variable getter.
    * @param  oldGetter Old variable getter.
    * @return Variable getter to use.
    */
  private[this] def maybeWrapCustomVariableGetter(getter: VariableGetter, oldGetter: VariableGetter): VariableGetter = {
    if (getter == null) {
      oldGetter
    } else {
      (name: String, shape: Shape, dataType: DataType, initializer: Initializer, regularizer: Regularizer,
          trainable: Boolean, reuse: java.lang.Boolean, collections: Set[String],
          cachingDevice: (OpSpecification) => String, customGetter: VariableGetter) => {
        val g: VariableGetter = (n: String, s: Shape, dt: DataType, init: Initializer, reg: Regularizer, train: Boolean,
            reuse: java.lang.Boolean, colls: Set[String], cDevice: (OpSpecification) => String, _: VariableGetter) => {
          oldGetter(n, s, dt, init, reg, train, reuse, colls, cDevice, customGetter)
        }
        getter(name, shape, dataType, initializer, regularizer, trainable, reuse, collections, cachingDevice, g)
      }
    }
  }
}
