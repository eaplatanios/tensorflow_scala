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

package org.platanios.tensorflow.api.ops.variables

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.{InvalidDataTypeException, ShapeMismatchException}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}
import org.platanios.tensorflow.api.ops.variables.Variable._
import org.platanios.tensorflow.api.types.{DataType, FLOAT32}

import scala.collection.mutable.ArrayBuffer

/** Variable store that carries a number of named variables.
  *
  * @author Emmanouil Antonios Platanios
  */
case class VariableStore private[variables]() {
  /** Map with variable names as keys and the corresponding variables as values. */
  private[this] var variables: Map[String, Variable] = Map.empty[String, Variable]

  /** Map with partitioned variable names as keys and the corresponding partitioned variables as values. */
  private[this] var partitionedVariables: Map[String, PartitionedVariable] = Map.empty[String, PartitionedVariable]

  /** Map with variable scope names as keys and the corresponding use counts as values. */
  private[this] var variableScopeCounts: Map[String, Int] = Map.empty[String, Int]

  // TODO: [DOC] [VARIABLES]

  private[api] def enterVariableScope(scope: String): Unit = variableScopeCounts synchronized {
    variableScopeCounts += scope -> (variableScopeCounts.getOrElse(scope, 0) + 1)
  }

  private[api] def exitVariableScope(scope: String): Unit = variableScopeCounts synchronized {
    variableScopeCounts += scope -> (variableScopeCounts.getOrElse(scope, 1) - 1)
  }

  private[api] def setVariableScopeCounts(counts: Map[String, Int]): Unit = variableScopeCounts synchronized {
    variableScopeCounts ++= counts
  }

  private[api] def getVariableSubScopeCounts(scope: String): Map[String, Int] = variableScopeCounts synchronized {
    variableScopeCounts.filterKeys(_.startsWith(s"$scope/"))
  }

  private[api] def closeVariableSubScopes(scope: String): Unit = variableScopeCounts synchronized {
    variableScopeCounts.keySet.filter(_.startsWith(s"$scope/")).foreach(variableScopeCounts - _)
  }

  /** Returns the use count of the provided scope in this variable store.
    *
    * @param  scope Variable scope name.
    * @return Number of usages of the provided variable scope name, in this variable store.
    */
  private[api] def variableScopeCount(scope: String): Int = variableScopeCounts.getOrElse(scope, 0)

  /** Gets a name with the provided prefix that is unique in the current variable scope.
    *
    * @param  prefix Prefix.
    * @return Unique name with the provided prefix.
    */
  private[api] def uniqueVariableScope(prefix: String): String = {
    val currentScope = Op.convertNameScopeToName(Op.currentVariableScope.name)
    val name = {
      if (currentScope == null || currentScope == "")
        prefix
      else
        s"$currentScope/$prefix"
    }
    if (variableScopeCounts.getOrElse(name, 0) == 0) {
      prefix
    } else {
      var uniqueName = name
      var count = 1
      while (variableScopeCounts.getOrElse(uniqueName, 0) > 0) {
        uniqueName = s"${name}_$count"
        count += 1
      }
      uniqueName
    }
  }

  /** Gets or creates a variable.
    *
    * @param  name          Variable name.
    * @param  dataType      Variable data type.
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
    * @param  customGetter  Function that has the same signature as this function, except for its last argument and that
    *                       specifies custom variable getting behavior. For example, one can specify a custom variable
    *                       getter in order to automatically rename the variables, before calling the underlying getter.
    *                       The underlying variable getter (i.e., the one which is used by default), is provided as a
    *                       last argument to the `customGetter` function.
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
  def getVariable(
      name: String, dataType: DataType = FLOAT32, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, trainable: Boolean = true, reuse: Reuse = ReuseOrCreateNew,
      collections: Set[Graph.Key[Variable]] = Set.empty, cachingDevice: OpSpecification => String = null,
      customGetter: VariableGetter = null): Variable = {
    /** This function defines the main logic of 'getVariable'. However, 'customGetter' may override this logic. That is
      * why we pass it as an argument to the 'customGetter'. */
    val trueGetter: VariableGetter =
      new VariableGetter {
        override def apply(
            name: String, dataType: DataType, shape: Shape, initializer: Initializer, regularizer: Regularizer,
            trainable: Boolean, reuse: Reuse, collections: Set[Graph.Key[Variable]],
            cachingDevice: (OpSpecification) => String, customGetter: VariableGetter): Variable = {
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
            foundVariable
          } else {
            // Here we handle the case of creating a new variable.
            if (reuse == ReuseExistingOnly)
              throw new IllegalArgumentException(
                s"Variable '$name' does not exist, but variable scope re-use was set to 'ReuseExistingOnly'.")
            if (shape != null && !shape.isFullyDefined)
              throw new IllegalArgumentException(
                s"The shape of a new variable ('$name') must be fully defined, but instead it was set to '$shape'.")
            val actualInitializer = if (initializer == null) defaultInitializer(name, dataType) else initializer
            val variable = Variable(actualInitializer, dataType, shape, trainable, collections, cachingDevice, name)
            variables += name -> variable
            // TODO: [LOGGING]
            // Run the regularizer if specified and save the resulting loss.
            if (regularizer != null) {
              Op.createWith(colocationOps = Set[Op](variable.op)) {
                val loss = Op.createWithNameScope(s"$name/Regularizer")(regularizer(variable.value))
                if (loss != null)
                  Op.currentGraph.addToCollection(loss, Graph.Keys.REGULARIZATION_LOSSES)
              }
            }
            variable
          }
        }
      }

    if (customGetter != null) {
      customGetter(
        name, dataType, shape, initializer, regularizer, trainable, reuse, collections, cachingDevice, trueGetter)
    } else {
      trueGetter(name, dataType, shape, initializer, regularizer, trainable, reuse, collections, cachingDevice, null)
    }
  }

  /** Gets or creates a partitioned variable.
    *
    * @param  name          Variable name.
    * @param  dataType      Variable data type.
    * @param  shape         Variable shape.
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
    * @param  reuse         [[Reuse]] value indicating whether to re-use an existing variable with the same name, create
    *                       a new variable, or do either.
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
  def getPartitionedVariable(
      name: String, dataType: DataType = FLOAT32, shape: Shape = null, initializer: Initializer = null,
      regularizer: Regularizer = null, partitioner: Partitioner = null, trainable: Boolean = true,
      reuse: Reuse = ReuseOrCreateNew, collections: Set[Graph.Key[Variable]] = Set.empty,
      cachingDevice: OpSpecification => String = null): PartitionedVariable = {
    Op.createWithNameScope("") {
      if (variables.contains(name))
        throw new IllegalArgumentException(
          s"A partitioner was provided, but an unpartitioned version of the variable was found: '$name'. " +
              s"Perhaps a variable of the same name was already created without partitioning?")
      val partitions: Array[Int] = {
        if (partitioner == null) {
          null
        } else {
          val partitions = partitioner(dataType, shape)
          if (partitions.length != shape.rank)
            throw new IllegalArgumentException(
              s"The partitioner returned an array of length (${partitions.length}) that does not match the provided " +
                  s"shape rank (${shape.rank}).")
          if (partitions.exists(_ < 1))
            throw new IllegalArgumentException(
              s"The partitioner returned zero partitions for some axes (partitions = '$partitions').")
          partitions
        }
      }
      if (partitionedVariables.contains(name)) {
        // Here we handle the case of returning an existing variable.
        if (reuse == CreateNewOnly)
          throw new IllegalArgumentException(
            s"Partitioned variable '$name' already exists, but variable scope re-use was set to 'CreateNewOnly'.")
        val foundVariable = partitionedVariables(name)
        if (!shape.isCompatibleWith(foundVariable.shape))
          throw ShapeMismatchException(
            s"Trying to share partitioned variable '$name', but the specified shape '$shape' is not compatible " +
                s"with the existing variable shape '${foundVariable.shape}'.")
        if (dataType != foundVariable.dataType)
          throw InvalidDataTypeException(
            s"Trying to share partitioned variable '$name', but the specified data type '$dataType' is not compatible " +
                s"with the existing variable data type '${foundVariable.dataType}'.")
        if (partitions != null && !(foundVariable.partitions sameElements partitions))
          throw new IllegalArgumentException(
            s"Trying to reuse partitioned variable '$name', but specified partitions '$partitions' are not compatible " +
                s"with the existing variable partitions.")
        foundVariable
      } else {
        // Here we handle the case of creating a new variable.
        if (reuse == ReuseExistingOnly)
          throw new IllegalArgumentException(
            s"Partitioned variable '$name' does not exist, but variable scope re-use was set to 'ReuseExistingOnly'.")
        if (partitions == null)
          throw new IllegalArgumentException(
            s"Trying to create a new partitioned variable, but the partitioner returned '$partitions'.")
        val (sliceDimension, sliceShape) = computeSliceDimensionAndShape(shape, partitions)
        val numberOfParts = partitions(sliceDimension)

        // Check if the variable parts exist in the variable store and if they are of the same number.
        if (variables.contains(s"$name/part_0")) {
          if (!variables.contains(s"$name/part_${numberOfParts - 1}"))
            throw new IllegalArgumentException(
              s"The partitioner returned a different partitioning than what was already found in the variable store. " +
                  s"The partitioner returned $numberOfParts parts, and part '$name/part_0' was found in the " +
                  s"variable store, but part '$name/part_${numberOfParts - 1}' was not found in it.")
          if (variables.contains(s"$name/part_$numberOfParts"))
            throw new IllegalArgumentException(
              s"The partitioner returned a different partitioning than what was already found in the variable store. " +
                  s"The partitioner returned $numberOfParts parts, and part '$name/part_0' was found in the " +
                  s"variable store, but so was the extra part '$name/part_$numberOfParts'.")
        }

        // Get the variable parts.
        val variableParts = ArrayBuffer.empty[Variable]
        val partOffset = ArrayBuffer.fill[Int](shape.rank)(0)
        val numberOfPartsWithExcess = shape(sliceDimension) % numberOfParts
        for (i <- 0 until numberOfParts) {
          val partitionShape: Array[Int] = sliceShape.asArray.map(s => if (i < numberOfPartsWithExcess) s + 1 else s)
          val partitionOffsets: Array[Int] = partOffset.toArray
          partOffset(sliceDimension) += partitionShape(sliceDimension)
          val actualInitializer = if (initializer == null) defaultInitializer(name, dataType) else initializer
          val partitionInfo = PartitionInformation(name, shape, partitionOffsets, partitionShape)
          val variablePart = Op.createWithNameScope("") {
            getVariable(
              name = s"$name/part_$i",
              dataType = dataType,
              shape = Shape.fromSeq(partitionShape),
              initializer = InitializerWithPartitionInformation(actualInitializer, partitionInfo),
              regularizer = regularizer,
              trainable = trainable,
              reuse = reuse,
              collections = collections,
              cachingDevice = cachingDevice)
          }
          variablePart.partitionInformation = partitionInfo
          variableParts += variablePart
        }

        // Create and return the new partitioned variable.
        val partitionedVariable = PartitionedVariable(name, dataType, shape, variableParts, partitions)
        partitionedVariables += name -> partitionedVariable
        // TODO: [LOGGING]
        partitionedVariable
      }
    }
  }

  /** Returns a default variable initializer.
    *
    * @param  name     Variable name.
    * @param  dataType Variable data type.
    * @return Default initializer.
    * @throws IllegalArgumentException If no default initializer is defined for the specified data type.
    */
  @throws[IllegalArgumentException]
  private[this] def defaultInitializer(name: String, dataType: DataType = FLOAT32): Initializer = {
    if (dataType.isFloatingPoint)
      GlorotUniformInitializer()
    else if (dataType.isInteger || dataType.isUnsigned || dataType.isBoolean)
      ZerosInitializer
    else
      throw new IllegalArgumentException(s"A default initializer for variable '$name' of type '$dataType' is required.")
  }

  /** Computes which dimension is being sliced and the typical slice shape.
    *
    * @param  fullShape  Variable full shape.
    * @param  partitions Number of partitions along each dimension of the variable.
    * @return Tuple containing the slicing dimension and the corresponding slice shape.
    * @throws IllegalArgumentException If the provided partitions are invalid.
    */
  @throws[IllegalArgumentException]
  private[this] def computeSliceDimensionAndShape(fullShape: Shape, partitions: Array[Int]): (Int, Shape) = {
    val sliceShape = ArrayBuffer.fill[Int](fullShape.rank)(0)
    var sliceDimension = -1
    for (((dimensionPartitions, dimensionSize), dimension) <- partitions.zip(fullShape.asArray).zipWithIndex) {
      if (dimensionPartitions <= 0 || dimensionSize < dimensionPartitions)
        throw new IllegalArgumentException(
          s"Cannot create $dimensionPartitions slices for size $dimensionSize " +
              s"(shape = $fullShape, partitions = $partitions).")
      if (dimensionPartitions == 1) {
        // Not slicing along this dimension.
        sliceShape(dimension) = dimensionSize
      } else if (sliceDimension >= 0) {
        // We only support slicing along one of the dimensions.
        throw new IllegalArgumentException(
          s"Can only slice a variable along one dimension (shape = $fullShape, partitions = $partitions).")
      } else {
        // Note that we will add any extras onto the last slice later.
        sliceShape(dimension) = dimensionSize / dimensionPartitions
        sliceDimension = dimension
      }
    }
    // Degenerate case: If "partitions" is all ones, we pretend we are slicing along the first dimension.
    if (sliceDimension == -1)
      sliceDimension = 0
    (sliceDimension, Shape.fromSeq(sliceShape))
  }
}
