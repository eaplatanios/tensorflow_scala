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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.core.types.{DataType, TF}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}

/** Variable store that carries a number of named variables.
  *
  * @author Emmanouil Antonios Platanios
  */
case class VariableStore private[variables]() {
  /** Map with variable names as keys and the corresponding variables as values. */
  private var variables: Map[String, Variable[Any]] = {
    Map.empty[String, Variable[Any]]
  }

  /** Gets or creates a variable.
    *
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
      name: String,
      shape: Shape,
      initializer: Initializer = null,
      regularizer: Regularizer = null,
      trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable[Any]]] = Set.empty,
      cachingDevice: OpSpecification => String = null
  ): Variable[T] = {
    val dataType = implicitly[TF[T]].dataType
    // Single variable case.
    if (variables.contains(s"$name/part_0"))
      throw new IllegalArgumentException(
        s"No partitioner was provided, but a partitioned version of the variable ('$name/part_0') was found in " +
            s"the variable store. Perhaps a variable of the same name was already created with partitioning?")
    if (variables.contains(name)) {
      // Here we handle the case of returning an existing variable.
      if (reuse == CreateNewOnly)
        throw new IllegalArgumentException(
          s"Variable '$name' already exists, but variable scope re-use was set to 'CreateNewOnly'.")
      val foundVariable = variables(name)
      if (shape != null && !shape.isCompatibleWith(foundVariable.shape))
        throw ShapeMismatchException(
          s"Trying to share variable '$name', but the specified shape '$shape' is not compatible with the " +
              s"existing variable shape '${foundVariable.shape}'.")
      if (dataType != foundVariable.dataType)
        throw InvalidDataTypeException(
          s"Trying to share variable '$name', but the specified data type '$dataType' is not compatible with the " +
              s"existing variable data type '${foundVariable.dataType}'.")
      foundVariable.asInstanceOf[Variable[T]]
    } else {
      // Here we handle the case of creating a new variable.
      if (reuse == ReuseExistingOnly)
        throw new IllegalArgumentException(
          s"Variable '$name' does not exist, but variable scope re-use was set to 'ReuseExistingOnly'.")
      if (shape != null && !shape.isFullyDefined)
        throw new IllegalArgumentException(
          s"The shape of a new variable ('$name') must be fully defined, but instead it was set to '$shape'.")
      val actualInitializer = Op.initializationScope {
        if (initializer == null)
          defaultInitializer(name, dataType.asInstanceOf[DataType[Any]])
        else
          initializer
      }
      val variable = makeGetter()(
        name, dataType, shape, actualInitializer, regularizer,
        trainable, reuse, collections, cachingDevice, null)
      variables += name -> variable.asUntyped
      // TODO: [LOGGING]
      // Run the regularizer if specified and save the resulting loss.
      if (regularizer != null) {
        Op.colocateWith(Set(variable.op), ignoreExisting = true) {
          val loss = Op.nameScope(s"$name/Regularizer")(regularizer(variable.value))
          if (loss != null)
            Op.currentGraph.addToCollection(Graph.Keys.REGULARIZATION_LOSSES)(loss.asUntyped)
        }
      }
      variable
    }
  }
}

object VariableStore {
  def current: VariableStore = {
    Op.currentGraph.variableStore
  }
}
