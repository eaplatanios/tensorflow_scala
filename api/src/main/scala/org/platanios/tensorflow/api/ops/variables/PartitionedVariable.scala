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

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.types.DataType

import scala.math.Ordering.Implicits._

/** Partitioned variable wrapper.
  *
  * Variables passed via `wrappedVariables` must contain a non-null save slice information field. Concatenation and
  * iteration is in lexicographic order according to the `variableOffset` property of the save slice information.
  *
  * Accessing this object as an [[Output]] returns the variable parts concatenated along the partition axis.
  *
  * This wrapper also acts as an iterator that allows accessing the underlying variables. This iterator is necessary to
  * control the order of access when variables are not partitioned in a standard way along a single axis.
  *
  * @param  name             Overall name of the variables.
  * @param  dataType         Data type of the variables.
  * @param  shape            Overall shape of the variables.
  * @param  wrappedVariables Variables that comprise this partitioned variable.
  * @param  partitions       Number of partitions for each axis/dimension.
  * @throws IllegalArgumentException If the provided variables sequence is empty, or if their shapes do not match with
  *                                  `shape`, or if their data types do not match `dataType`., or if they have
  *                                  `null`-valued save slice information
  *
  * @author Emmanouil Antonios Platanios
  */
@throws[IllegalArgumentException]
case class PartitionedVariable private[variables](
    override val name: String,
    override val dataType: DataType,
    override val shape: Shape,
    private val wrappedVariables: Seq[Variable],
    partitions: Array[Int]
) extends Iterable[Variable] with VariableLike {
  if (shape.rank != partitions.length)
    throw new IllegalArgumentException(
      s"The number of partitions provided (${partitions.length}) does not match the shape rank (${shape.rank}).")
  if (partitions.exists(_ <= 0))
    throw new IllegalArgumentException(s"All partitions must be positive: $partitions.")
  if (wrappedVariables.isEmpty)
    throw new IllegalArgumentException("The provided variables list may not be empty.")
  if (wrappedVariables.exists(_.partitionInformation == null))
    throw new IllegalArgumentException(s"All variables must have save slice information available.")
  if (wrappedVariables.exists(_.dataType != dataType))
    throw new IllegalArgumentException("All variables' data type must match the provided data type.")
  if (wrappedVariables.exists(_.partitionInformation.fullShape != shape))
    throw new IllegalArgumentException(
      "All variables' save slice information full shape must match the provided shape.")

  /** Graph where this variable is defined. */
  override val graph: Graph = wrappedVariables.head.graph

  val variables: Seq[Variable] = wrappedVariables.sortBy(_.partitionInformation.partitionOffsets.toList)

  /** Returns the overall concatenated value as an [[Output]].
    *
    * This is different from using the partitioned variable directly as an op output (through implicit conversion or
    * `toOutput`) in that it creates a new set of ops that retain the control dependencies from its scope.
    *
    * @return Concatenated op output.
    * @throws IllegalArgumentException If having more than one partition axes.
    */
  @throws[IllegalArgumentException]
  def concatenated: Output = {
    val concatenated: Output = {
      if (variables.length == 1) {
        variables.head.value
      } else {
        if (partitionAxes.length > 1)
          throw new IllegalArgumentException(
            s"Cannot concatenate along more than one dimension: $partitionAxes. " +
                "Multi-axis partition concatenation is not supported.")
        Op.createWithNameScope(s"$name/ConcatenatedPartitions") {
          Basic.concatenate(variables.map(_.value), partitionAxes(0))
        }
      }
    }
    Op.createWithNameScope("") {
      Basic.identity(concatenated, name)
    }
  }

  /** Returns a cached op which reads the last value of this partitioned variable.
    *
    * You can not assign a new value to the returned tensor as it is not a reference to the variable.
    *
    * The returned op output will not inherit the control dependencies from the scope where the value is used, which is
    * equivalent behavior to that of getting the value of a variable.
    *
    * NOTE: You usually do not need to call this method directly, as all ops that use variables do so by internally
    * converting them to tensors.
    */
  override val value: Output = {
    Op.createWith(controlDependencies = Set.empty[Op]) {
      concatenated
    }
  }

  /** Op responsible for initializing this variable. */
  override val initializer: Op = ControlFlow.group(variables.map(_.initializer).toSet)

  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  override val isInitialized: Output = Op.createWith(graph) {
    Math.all(Basic.stack(variables.map(_.isInitialized)), name = "IsInitialized")
  }

  /** Value of the initialized variable. You should use this instead of the variable itself to initialize
    * another variable with a value that depends on the value of this variable.
    *
    * Example:
    * {{{
    *   // Initialize `v` with random values, and then use `initializedValue` to guarantee that `v` has been initialized
    *   // before its value is used to initialize `w`. The random tensor will only be sampled once.
    *   val v = tf.variable("v", FLOAT32, Shape(10, 40), tf.RandomTruncatedNormalInitializer())
    *   val w = tf.variable("w", initializer = tf.ConstantInitializer(v.initializedValue * 2.0))
    * }}}
    */
  override val initializedValue: Output = Op.initialization {
    ControlFlow.cond(
      isInitialized,
      () => value,
      () => Op.createWith(controlDependencies = Set(initializer))(value))
  }

  /** Creates an op that reads the value of this variable.
    *
    * This method should be used when there are multiple reads, or when it is desirable to read the value only after
    * some condition is true.
    *
    * The returned value may be different from that of [[value]] depending on the device being used, the control
    * dependencies, etc.
    *
    * @return Created op.
    */
  override def read(name: String = "Read"): Output = {
    Op.createWith(graph) {
      // Return an identity op so that it can get placed on whatever device the context specifies instead of the device
      // where the variable is.
      Basic.identity(value)
    }
  }

  /** Creates an op that reads the value of this variable sparsely, using the provided `indices`.
    *
    * This method should be used when there are multiple reads, or when it is desirable to read the value only after
    * some condition is true.
    *
    * @param  indices Indices to use for the sparse read.
    * @param  name    Name for the created op.
    * @return Created op.
    */
  @throws[UnsupportedOperationException]
  override def gather(indices: Output, name: String = "Gather"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'gather' yet.")
  }

  /** Creates an op that assigns the provided value to this variable and returns its value.
    *
    * @param  value Value to assign the variable to.
    * @param  name  Name for created op.
    * @return Variable value read op, after the assignment.
    */
  @throws[UnsupportedOperationException]
  override def assign(value: Output, name: String = "Assign"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'assign' yet.")
  }

  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
    *
    * @param  value Value to add to the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignAdd(value: Output, name: String = "AssignAdd"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'assignAdd' yet.")
  }

  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
    *
    * @param  value Value to subtract from the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignSub(value: Output, name: String = "AssignAdd"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'assignSub' yet.")
  }

  /** Creates an op that applies updates the provided sparse value updates to this variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` used for the update.
    * @param  values  Values to use for updating, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignScatter(indices: Output, values: Output, name: String = "AssignScatter"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'assignScatter' yet.")
  }

  /** Creates an op that adds the provided sparse value to the current value of the variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` being added.
    * @param  values  Values to be added, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignScatterAdd(indices: Output, values: Output, name: String = "AssignScatterAdd"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'assignScatterAdd' yet.")
  }

  /** Creates an op that subtracts the provided sparse value from the current value of the variable and returns its
    * value.
    *
    * @param  indices Indices corresponding to the `values` being subtracted.
    * @param  values  Values to be subtracted, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  @throws[InvalidDataTypeException]
  override def assignScatterSub(indices: Output, values: Output, name: String = "AssignScatterSub"): Output = {
    throw new UnsupportedOperationException("Partitioned variables do not support 'assignScatterAdd' yet.")
  }

  /** Returns an array of integers containing the partition axes of this partitioned variable. */
  private[this] val partitionAxes: Array[Int] = {
    val filteredPartitions = partitions.zipWithIndex.filter(_._1 > 1).map(_._2)
    if (filteredPartitions.isEmpty)
      Array[Int](0)
    else
      filteredPartitions
  }

  /** Returns the number of partitions of this partitioned variable.
    *
    * @throws IllegalArgumentException If having more than one partition axes.
    */
  @throws[IllegalArgumentException]
  def length: Int = {
    if (partitionAxes.length > 1)
      throw new IllegalArgumentException(s"Cannot get a length for ${partitionAxes.length} > 1 partition axes.")
    variables.length
  }

  /** Returns an iterator for accessing the underlying partition variables.
    *
    * This iterator is necessary to control order of access when variables are not partitioned in a standard way along
    * a single axis.
    */
  override def iterator: Iterator[Variable] = new Iterator[Variable] {
    private[this] var index: Int = 0

    override def hasNext: Boolean = index < variables.length

    override def next(): Variable = {
      val nextVariable = variables(index)
      index += 1
      nextVariable
    }
  }
}
