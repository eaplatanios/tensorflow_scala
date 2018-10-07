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
import org.platanios.tensorflow.api.ops.{Output, UntypedOp}
import org.platanios.tensorflow.api.types.{DataType, IsInt32OrInt64, IsNumeric, TF}

/** Represents objects that can be used as variables (e.g., variables and partitioned variables).
  *
  * @author Emmanouil Antonios Platanios
  */
trait VariableLike[+T] {
  /** Graph where this variable is defined. */
  val graph: Graph

  /** Name of this variable. */
  val name: String

  /** Data type of this variable. */
  val dataType: DataType[T]

  /** Shape of this variable. */
  val shape: Shape

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
  val value: Output[T]

  /** Op responsible for initializing this variable. */
  val initializer: UntypedOp

  /** Op output that is `true` when the variable has been initialized and `false` otherwise. */
  val isInitialized: Output[Boolean]

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
  val initializedValue: Output[T]

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
  def read(name: String = "Read"): Output[T]

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
  def gather[I: IsInt32OrInt64 : TF](indices: Output[I], name: String = "Gather"): Output[T]

  /** Creates an op that assigns the provided value to this variable and returns its value.
    *
    * @param  value Value to assign the variable to.
    * @param  name  Name for created op.
    * @return Variable value read op, after the assignment.
    */
  @throws[UnsupportedOperationException]
  def assign[V >: T : TF](value: Output[V], name: String = "Assign"): Output[V]

  /** Creates an op that adds the provided value to the current value of the variable and returns its value.
    *
    * @param  value Value to add to the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  def assignAdd[V >: T : TF](value: Output[V], name: String = "AssignAdd"): Output[V]

  /** Creates an op that subtracts the provided value from the current value of the variable and returns its value.
    *
    * @param  value Value to subtract from the current variable value.
    * @param  name  Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  def assignSub[V >: T : TF](value: Output[V], name: String = "AssignAdd"): Output[V]

  /** Creates an op that applies updates the provided sparse value updates to this variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` used for the update.
    * @param  values  Values to use for updating, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  def assignScatter[V >: T : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatter"
  ): Output[V]

  /** Creates an op that adds the provided sparse value to the current value of the variable and returns its value.
    *
    * @param  indices Indices corresponding to the `values` being added.
    * @param  values  Values to be added, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the addition.
    */
  @throws[UnsupportedOperationException]
  def assignScatterAdd[V >: T : IsNumeric : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatterAdd"
  ): Output[V]

  /** Creates an op that subtracts the provided sparse value from the current value of the variable and returns its
    * value.
    *
    * @param  indices Indices corresponding to the `values` being subtracted.
    * @param  values  Values to be subtracted, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  def assignScatterSub[V >: T : IsNumeric : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatterSub"
  ): Output[V]

  /** Creates an op that multiplies the provided sparse value from the current value of the variable and returns its
    * value.
    *
    * @param  indices Indices corresponding to the `values` being multiplied.
    * @param  values  Values to multiply with, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  def assignScatterMul[V >: T : IsNumeric : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatterMul"
  ): Output[V]

  /** Creates an op that divides the current value of the variable by the provided sparse value and returns its
    * value.
    *
    * @param  indices Indices corresponding to the `values` dividing the current variable value.
    * @param  values  Values to divide by, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  def assignScatterDiv[V >: T : IsNumeric : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatterDiv"
  ): Output[V]

  /** Creates an op that computes the element-wise minimum between the current value of the variable and the provided
    * sparse value, and returns its value.
    *
    * @param  indices Indices corresponding to the provided `values`.
    * @param  values  Values to compute the minimum with respect to, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  def assignScatterMin[V >: T : IsNumeric : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatterMin"
  ): Output[V]

  /** Creates an op that computes the element-wise maximum between the current value of the variable and the provided
    * sparse value, and returns its value.
    *
    * @param  indices Indices corresponding to the provided `values`.
    * @param  values  Values to compute the maximum with respect to, corresponding to the provided `indices`.
    * @param  name    Name for created op.
    * @return Variable value read op, after the subtraction.
    */
  @throws[UnsupportedOperationException]
  def assignScatterMax[V >: T : IsNumeric : TF, I: IsInt32OrInt64 : TF](
      indices: Output[I],
      values: Output[V],
      name: String = "AssignScatterMax"
  ): Output[V]

  /** Converts this variable to an op output. This function simply returns an op corresponding to the variable value. */
  def toOutput: Output[T] = {
    value
  }
}
