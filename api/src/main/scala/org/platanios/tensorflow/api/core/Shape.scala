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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.ops.{Basic, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32}
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import org.tensorflow.framework.TensorShapeProto

import scala.collection.JavaConverters._

/** Represents the shape of a tensor computed by an op.
  *
  * A `Shape` represents a possibly-partial shape specification for an op output. It may be one of the following:
  *
  *  - Fully known shape: It has a known number of dimensions and a known size for each dimension.
  *  - Partially known shape: It has a known number of dimensions and an unknown size for one or more dimension.
  *  - Unknown shape: It has an unknown number of dimensions and an unknown size for all its dimensions.
  *
  * Unknown dimensions are represented as having a size of `-1` and two shapes are considered equal only if they are
  * fully known and all their dimension sizes match.
  *
  * If a tensor is produced by an op of type `"Foo"`, its shape may be inferred if there is a registered shape function
  * for `"Foo"`. See [[https://www.tensorflow.org/extend/adding_an_op#shape_functions_in_c Shape Functions in C++]] for
  * details on shape functions and how to register them.
  *
  * @param  array Array representation of the shape to create.
  *
  * @author Emmanouil Antonios Platanios
  */
final class Shape private (private val array: Array[Int]) extends ProtoSerializable {
  /** Returns a boolean value indicating whether this shape is fully defined.
    *
    * If the size of any dimension is equal to `-1` or if the shape is completely unknown, then it is not considered
    * fully defined..
    */
  def isFullyDefined: Boolean = array != null && !array.contains(-1)

  /** Gets the rank of this shape (i.e., number of dimensions). */
  def rank: Int = if (array == null) -1 else array.length

  /** Gets the size for a specific dimension of this shape.
    *
    * @param  dimension Dimension whose size to return.
    */
  def size(dimension: Int): Int = {
    if (dimension < 0)
      array(array.length + dimension)
    else
      array(dimension)
  }

  /** Gets the total number of elements in tensors of this shape.
    *
    * If the shape is not fully defined, then `-1` is returned, otherwise, the product of the sizes for each dimension
    * of this shape is returned.
    */
  def numElements: Long = {
    if (!isFullyDefined) {
      -1
    } else {
      var size: Long = 1L
      array.foreach(size *= _)
      size
    }
  }

  /** Reshapes this shape to the provided shape.
    *
    * This function first checks that this shape can be reshaped in the specified way and then:
    *   - If `shape` has an unknown dimension, then its value is computed, filled in, and the new shape is returned.
    *   - Otherwise, `shape` is returned.
    *
    * @param  shape Shape to reshape to.
    * @return New shape.
    * @throws IllegalArgumentException If this shape cannot be reshaped to `shape`.
    */
  @throws[IllegalArgumentException]
  def reshape(shape: Shape): Shape = {
    this.assertFullyDefined("Only fully defined shapes can be reshaped.")
    val unknownDimensions = shape.asArray.count(_ == -1)
    if (shape.rank == -1 || unknownDimensions > 1)
      throw new IllegalArgumentException(
        s"The new shape ($shape) must have known rank and at most one unknown dimension.")
    if (unknownDimensions == 0 && this.numElements != shape.numElements) {
      throw new IllegalArgumentException(
        s"Shape '$this' cannot be reshaped to '$shape' (different number of elements).")
    } else if (unknownDimensions == 0) {
      shape
    } else {
      val unknownIndex = shape.asArray.indexWhere(_ == -1)
      val otherNumElements = shape.asArray.filter(_ == -1).product
      if (this.numElements % otherNumElements != 0)
        throw new IllegalArgumentException(s"Shape '$this' cannot be reshaped to '$shape'.")
      val newShape = shape.asArray
      newShape(unknownIndex) = (this.numElements / otherNumElements).toInt
      new Shape(newShape)
    }
  }

  /** Returns an array representation of this shape. This method does not perform a copy or an array creation. It simply
    * returns the underlying array representation of this shape. Its cost is thus the same as that of a field access. */
  def asArray: Array[Int] = array

  /** Checks if `other` is compatible with this shape.
    *
    * Two shapes are compatible if either of them is completely unknown, or if they have the same rank and each one of
    * their dimensions is compatible. One dimension is compatible with another if either one is equal to `-1` or if they
    * have the same value.
    *
    * For example:
    *  - `Shape.unknown()` is compatible with every other shape.
    *  - `Shape.unknown(rank = r)` is compatible with every other shape which has rank `r`.
    *  - `Shape(-1, -1)` is compatible with all rank `2` shapes.
    *  - `Shape(32, -1)` is compatible with all rank `2` shapes whose first dimension size is equal to `-1` or `32`.
    *  - `Shape(32, 784)` is compatible with itself and `Shape(-1, 784)`, `Shape(32, -1)`, `Shape.unknown(rank = 2)`,
    * and `Shape.unknown()`.
    *
    * The compatibility relation is reflexive and symmetric, but not transitive. For example, `Shape(32, 784)` is
    * compatible with `Shape.unknown()`, and `Shape.unknown()` is compatible with `Shape(4, 4)`, but `Shape(32, 784)` is
    * not compatible with `Shape(4, 4)`.
    *
    * @param  other Shape to check compatibility with.
    * @return Boolean value indicating whether the two shapes are compatible.
    */
  def isCompatibleWith(other: Shape): Boolean = {
    this.rank == -1 || other.rank == -1 ||
        (this.rank == other.rank &&
            this.array != null && other.array != null &&
            this.array.zip(other.asArray).forall(t => t._1 == -1 || t._2 == -1 || t._1 == t._2))
  }

  /** Merges two shapes and returns the result as a new shape.
    *
    * Merging consists of first checking whether the shapes are compatible using the [[isCompatibleWith]] method and
    * then going through each dimension of this shape and keeping it if it not equal to `-1` (i.e., unknown), or setting
    * it equal to the respective dimension of `other`, otherwise. This effectively merges the information contained in
    * the two shapes.
    *
    * For example:
    * {{{
    *   val shape1 = Shape(2, 3, -1, 1)
    *   val shape2 = Shape(-1, 3, 5, -1)
    *   val mergedShape = shape1.mergeWith(shape2)
    *   assert(mergedShape == Shape(2, 3, 5, 1)
    * }}}
    *
    * The merging functionality is reflexive and symmetric, but not transitive, similar to the compatibility relation.
    *
    * @param  other Shape to merge with.
    * @throws InvalidShapeException If this shape is not compatible with `other`.
    */
  @throws[InvalidShapeException]
  def mergeWith(other: Shape): Shape = {
    if (this.rank == -1) {
      other
    } else if (other.rank == -1) {
      this
    } else {
      assertSameRank(other)
      assertIsCompatibleWith(other)
      new Shape(this.array.zip(other.asArray).map(t => {
        if (t._1 == -1) t._2
        else if (t._2 == -1) t._1
        else t._1
      }))
    }
  }

  def +(dimension: Int): Shape = new Shape(this.array :+ dimension)
  def ++(other: Shape): Shape = concatenateWith(other)

  // TODO: Support merging an unknown shape with a (partially) known one and vice-versa.
  /** Concatenates this shape with another shape and returns the result as a new shape.
    *
    * If any of the two shapes is completely unknown, then the result of the concatenation is also a completely unknown
    * shape. Otherwise, the two shapes are simply concatenated.
    *
    * For example:
    * {{{
    *   val shape1 = Shape(2, 3, -1, 1)
    *   val shape2 = Shape(-1, 3, 5, -1)
    *   val shape3 = Shape.unknown()
    *   val shape12 = shape1.concatenateWith(shape2)
    *   assert(shape12 == Shape(2, 3, -1, 1, -1, 3, 5, -1)
    *   val shape23 = shape2.concatenateWith(shape3)
    *   assert(shape23 == Shape.unknown())
    *   val shape31 = shape3.concatenateWith(shape1)
    *   assert(shape31 == Shape.unknown())
    * }}}
    *
    * @param  other Shape to concatenate with this shape.
    */
  def concatenateWith(other: Shape): Shape = {
    if (this.rank == -1 || other.rank == -1)
      new Shape(null)
    else
      new Shape(this.array ++ other.array)
  }

  /** Returns a shape with the specified rank that is based on the current shape.
    *
    * This method can be used to promote a completely unknown shape to one with a known rank.
    *
    * @param  rank Rank to use for the new shape.
    * @throws InvalidShapeException If this shape is fully or partially known and has a different rank than the
    *                               provided value.
    */
  @throws[InvalidShapeException]
  def withRank(rank: Int): Shape = mergeWith(Shape.unknown(rank))

  /** Returns a shape with at least the specified rank, that is based on the current shape.
    *
    * @param  rank Minimum rank to use for the new shape.
    * @throws InvalidShapeException If this shape is fully or partially known and has a rank that is smaller than the
    *                               provided value.
    */
  @throws[InvalidShapeException]
  def withRankAtLeast(rank: Int): Shape = {
    assertHasRank(rank)
    this
  }

  /** Asserts that this shape is fully defined (i.e., fully known). If it is not, an [[InvalidShapeException]] exception
    * is thrown.
    *
    * @throws InvalidShapeException If this shape is not fully defined.
    */
  @throws[InvalidShapeException]
  def assertFullyDefined(message: String = s"Shape '$this' must be fully defined."): Unit = {
    if (!this.isFullyDefined)
      throw InvalidShapeException(message)
  }

  /** Asserts that this shape has the specified rank.
    *
    * @param  rank Rank.
    * @throws InvalidShapeException If this shape has rank other than `rank`.
    */
  @throws[InvalidShapeException]
  def assertHasRank(rank: Int): Unit = {
    if (this.rank != -1 && this.rank != rank)
      throw InvalidShapeException(s"Shape '$this' must have rank $rank.")
  }

  /** Asserts that this shape has rank at least `rank` and throws an exception if it does not.
    *
    * @param  rank Rank lower bound.
    * @throws InvalidShapeException If this shape has rank lower than `rank`.
    */
  @throws[InvalidShapeException]
  def assertRankAtLeast(rank: Int): Unit = {
    if (this.rank < rank)
      throw InvalidShapeException(s"Shape '$this' must have rank at least $rank.")
  }

  /** Asserts that this shape has rank at most `rank` and throws an exception if it does not.
    *
    * @param  rank Rank upper bound.
    * @throws InvalidShapeException If this shape has rank higher than `rank`.
    */
  @throws[InvalidShapeException]
  def assertRankAtMost(rank: Int): Unit = {
    if (this.rank > rank)
      throw InvalidShapeException(s"Shape '$this' must have rank at most $rank.")
  }

  /** Asserts that this shape has the same rank as `other`. If the two shapes are not compatible, an
    * [[InvalidShapeException]] exception is thrown.
    *
    * @param  other Shape to assert having the same rank as.
    * @throws InvalidShapeException If this shape does not have the same rank as `other`.
    */
  @throws[InvalidShapeException]
  def assertSameRank(other: Shape): Unit = {
    if (this.rank != other.rank)
      throw InvalidShapeException(s"Shape '$this' must have the same rank as shape '$other'.")
  }

  /** Asserts that this shape is compatible with `other` using the [[isCompatibleWith]] method. If the two shapes are
    * not compatible, an [[InvalidShapeException]] exception is thrown.
    *
    * This method can be used to assert that there exists a shape that both this shape and `other` represent.
    *
    * @param  other Shape to assert compatibility with.
    * @throws InvalidShapeException If this shape is not compatible with `other`.
    */
  @throws[InvalidShapeException]
  def assertIsCompatibleWith(other: Shape): Unit = {
    if (!isCompatibleWith(other))
      throw InvalidShapeException(s"Shape '$this' must be compatible with shape '$other'.")
  }

  /** Gets the size for a specific dimension of this shape.
    *
    * @param  dimension Dimension whose size to return.
    */
  def apply(dimension: Int): Int = size(dimension)

  /** Gets a slice of this shape.
    *
    * @param  slice Slice to get.
    */
  def apply(slice: Slice): Shape = {
    if (slice == null)
      throw new IllegalArgumentException("The provided slice should not be 'null'.")
    if (array != null)
      Shape.fromSeq(slice.toArray(rank).map(i => array(i)))
    else
      Shape.unknown(slice.length(rank))
  }

  /** Converts this shape to a one-dimensional tensor.
    *
    * @param  dataType Data type to use for the tensor.
    * @return One-dimensional tensor representing this shape.
    */
  def toTensor(dataType: DataType = INT32): Tensor = {
    if (rank == 0)
      Tensor(dataType)
    else
      Tensor(dataType, asArray.head, asArray.tail: _*)
  }

  /** Converts this shape to a one-dimensional "symbolic" tensor (i.e., a constant-valued op output).
    *
    * @param  dataType Data type to use for the tensor.
    * @return One-dimensional op output tensor representing this shape.
    */
  def toOutput(dataType: DataType = INT32, name: String = "Shape"): Output = {
    Basic.constant(toTensor(dataType), name = name)
  }

  override def toProto: TensorShapeProto = toTensorShapeProto

  /** Constructs and returns a [[TensorShapeProto]] object that represents this shape.
    *
    * @return Constructed [[TensorShapeProto]].
    */
  def toTensorShapeProto: TensorShapeProto = {
    if (rank == -1) {
      TensorShapeProto.newBuilder().setUnknownRank(true).build()
    } else {
      val builder = TensorShapeProto.newBuilder()
      array.zipWithIndex.foreach(a => builder.addDim(a._2, TensorShapeProto.Dim.newBuilder().setSize(a._1)))
      builder.build()
    }
  }

  override def toString: String = if (array == null) "<unknown>" else s"[${array.mkString(", ").replace("-1", "?")}]"

  override def equals(that: Any): Boolean = that match {
    case that: Shape =>
      if ((this.rank != that.rank)
          || (this.array == null && that.array != null)
          || (this.array != null && that.array == null))
        false
      else if (this.array == null && that.array == null)
        true
      else
        this.array.sameElements(that.array)
    case _ => false
  }

  override def hashCode: Int = array.hashCode
}

/** Contains helper functions for creating [[Shape]] objects. */
object Shape {
  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def create(dimensions: Int*): Shape = new Shape(Array(dimensions: _*))

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def create(dimensions: Array[Int]): Shape = new Shape(dimensions)

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def fromSeq(dimensions: Seq[Int]): Shape = new Shape(dimensions.toArray)

  /** Creates an unknown shape, optionally with a known rank.
    *
    * @param  rank Optional rank of the shape to create. If set to `-1`, then it is considered unknown.
    */
  def unknown(rank: Int = -1): Shape = if (rank == -1) new Shape(null) else new Shape(Array.fill[Int](rank)(-1))

  /** Creates a shape representing a scalar. */
  def scalar(): Shape = Shape.create()

  /** Creates a shape representing a vector with the specified length.
    *
    * @param  length Vector length.
    */
  def vector(length: Int): Shape = Shape.create(length)

  /** Creates a shape representing a matrix with the specified number of rows and columns.
    *
    * @param  numRows    Matrix number of rows.
    * @param  numColumns Matrix number of columns.
    */
  def matrix(numRows: Int, numColumns: Int): Shape = Shape.create(numRows, numColumns)

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def apply(dimensions: Int*): Shape = create(dimensions: _*)

  /** Creates a shape with the specified dimension sizes.
    *
    * @param  dimensions Dimension sizes.
    */
  def apply(dimensions: Array[Int]): Shape = create(dimensions)

  /** Creates a shape from the provided [[TensorShapeProto]] object.
    *
    * @param  tensorShapeProto Serialized shape object.
    * @return Constructed [[Shape]].
    */
  def fromProto(tensorShapeProto: TensorShapeProto): Shape = fromTensorShapeProto(tensorShapeProto)

  /** Creates a shape from the provided [[TensorShapeProto]] object.
    *
    * @param  tensorShapeProto Serialized shape object.
    * @return Constructed [[Shape]].
    */
  def fromTensorShapeProto(tensorShapeProto: TensorShapeProto): Shape = {
    if (tensorShapeProto.getUnknownRank)
      Shape.unknown()
    else
      fromSeq(tensorShapeProto.getDimList.asScala.map(_.getSize.toInt))
  }

  // implicit def shapeToTensor(shape: Shape): Tensor = shape.toTensor()
  // implicit def shapeToOutput(shape: Shape): Output = shape.toOutput()

  // implicit def tupleToShape(t: Tuple1[Int]): Shape = create(t._1)
  // implicit def tupleToShape(t: (Int, Int)): Shape = create(t._1, t._2)
  // implicit def tupleToShape(t: (Int, Int, Int)): Shape = create(t._1, t._2, t._3)
  // implicit def tupleToShape(t: (Int, Int, Int, Int)): Shape = create(t._1, t._2, t._3, t._4)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int)): Shape = create(t._1, t._2, t._3, t._4, t._5)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int)): Shape = create(t._1, t._2, t._3, t._4, t._5, t._6)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14)
  // implicit def tupleToShape(t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(
  //     t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16, t._17)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(
  //     t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16, t._17,
  //     t._18)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(
  //     t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16, t._17,
  //     t._18, t._19)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int,
  //         Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(
  //     t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16, t._17,
  //     t._18, t._19, t._20)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int,
  //         Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(
  //     t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16, t._17,
  //     t._18, t._19, t._20, t._21)
  // implicit def tupleToShape(
  //     t: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int,
  //         Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int)): Shape =
  //   create(
  //     t._1, t._2, t._3, t._4, t._5, t._6, t._7, t._8, t._9, t._10, t._11, t._12, t._13, t._14, t._15, t._16, t._17,
  //     t._18, t._19, t._20, t._21, t._22)
}
