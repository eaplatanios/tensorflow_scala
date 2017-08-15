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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api.core
import org.platanios.tensorflow.api.core.exception.InvalidIndexerException
import org.platanios.tensorflow.api.ops.{Basic, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.INT32

import scala.language.postfixOps

/** Represents an indexer object. Indexers are used to index tensors.
  *
  * An indexer can be one of:
  *   - [[Ellipsis]]: Corresponds to a full slice over multiple dimensions of a tensor. Ellipses are used to represent
  *     zero or more dimensions of a full-dimension indexer sequence.
  *   - [[NewAxis]]: Corresponds to the addition of a new dimension.
  *   - [[Slice]]: Corresponds to a slice over a single dimension of a tensor.
  *
  * Examples of constructing and using indexers are provided in the [[Ellipsis]] and the  [[Slice]] class documentation.
  * Here we provide examples of indexing over tensors using indexers:
  * {{{
  *   // 't' is a tensor (i.e., Output) with shape [4, 2, 3, 8]
  *   t(::, ::, 1, ::)            // Tensor with shape [4, 2, 1, 8]
  *   t(1 :: -2, ---, 2)          // Tensor with shape [1, 2, 3, 1]
  *   t(---)                      // Tensor with shape [4, 2, 3, 8]
  *   t(1 :: -2, ---, NewAxis, 2) // Tensor with shape [1, 2, 3, 1, 1]
  *   t(1 ::, ---, NewAxis, 2)    // Tensor with shape [3, 2, 3, 1, 1]
  * }}}
  * where `---` corresponds to an ellipsis.
  *
  * Note that each indexing sequence is only allowed to contain at most one [[Ellipsis]]. Furthermore, if an ellipsis is
  * not provided, then one is implicitly appended at the end of indexing sequence. For example, `foo(2 :: 4)` is
  * equivalent to `foo(2 :: 4, ---)`.
  *
  * TODO: Add more usage examples.
  *
  * @author Emmanouil Antonios Platanios
  */
sealed trait Indexer

/** Helper trait for representing indexer construction phases. */
sealed trait IndexerConstruction

/** Helper class for representing a indexer construction phase that has already been provided one numbers. */
case class IndexerConstructionWithOneNumber private (n: Int) extends IndexerConstruction {
  def :: : Slice = Slice(start = n, end = -1, inclusive = true)

  def ::(leftSide: IndexerConstructionWithOneNumber): IndexerConstructionWithTwoNumbers = {
    IndexerConstructionWithTwoNumbers(leftSide.n, n)
  }
}

object IndexerConstructionWithOneNumber {
  implicit def indexerConstructionToIndex(construction: IndexerConstructionWithOneNumber): Index = {
    Index(index = construction.n)
  }
}

/** Helper class for representing a indexer construction phase that has already been provided two numbers. */
case class IndexerConstructionWithTwoNumbers private (n1: Int, n2: Int) extends IndexerConstruction {
  def :: : Slice = Slice(start = n1, end = -1, step = n2, inclusive = true)

  def ::(leftSide: IndexerConstructionWithOneNumber): IndexerConstructionWithThreeNumbers = {
    IndexerConstructionWithThreeNumbers(leftSide.n, n1, n2)
  }
}

object IndexerConstructionWithTwoNumbers {
  implicit def indexerConstructionToIndex(construction: IndexerConstructionWithTwoNumbers): Slice = {
    Slice(start = construction.n1, end = construction.n2)
  }
}

/** Helper class for representing a indexer construction phase that has already been provided three numbers. */
case class IndexerConstructionWithThreeNumbers private (n1: Int, n2: Int, n3: Int) extends IndexerConstruction

object IndexerConstructionWithThreeNumbers {
  implicit def indexerConstructionToIndex(construction: IndexerConstructionWithThreeNumbers): Slice = {
    Slice(start = construction.n1, end = construction.n3, step = construction.n2)
  }
}

/** Contains helper functions for dealing with indexers. */
object Indexer {
  val ---    : Indexer = core.Ellipsis
  val ::     : Slice   = core.Slice.::

  private[core] trait API extends Implicits {
    type Indexer = core.Indexer
    type Index = core.Index
    type Slice = core.Slice

    val ---    : Indexer = core.Ellipsis
    val NewAxis: Indexer = core.NewAxis
    val ::     : Slice   = core.Slice.::
  }

  private[core] trait Implicits {
    // TODO: Add begin mask support (not simple).

    implicit def intToIndex(index: Int): Index = Index(index = index)

    implicit def intToIndexerConstruction(n: Int): IndexerConstructionWithOneNumber = {
      IndexerConstructionWithOneNumber(n)
    }
  }

  private[api] object Implicits extends Implicits

  /** Decodes the provided indexers sequence into a new set of dimension sizes, begin offsets, end offsets, and strides,
    * for the provided tensor shape.
    *
    * This function returns a tuple of five integer arrays:
    *
    *   - Old dimension sizes (one for each dimension of the original tensor shape)
    *   - Dimension sizes (one for each dimension)
    *   - Begin offsets (one for each dimension)
    *   - End offsets (one for each dimension)
    *   - Strides (one for each dimension)
    *
    * @param  shape    Shape of the tensor being indexed.
    * @param  indexers Sequence of indexers to use.
    * @return Tuple containing the decoded indexing sequence information.
    * @throws InvalidIndexerException If an invalid indexing sequence is provided.
    */
  @throws[InvalidIndexerException]
  private[api] def decode(
      shape: Shape, indexers: Seq[Indexer]): (Array[Int], Array[Int], Array[Int], Array[Int], Array[Int]) = {
    // TODO: Make this more efficient.
    // TODO: Add tests for when providing an empty shape.
    val newAxesCount = indexers.count(_ == NewAxis)
    val ellipsesCount = if (indexers.contains(Ellipsis)) 1 else 0
    val newRank = Math.max(shape.rank + newAxesCount, shape.rank + newAxesCount - ellipsesCount)
    if (newRank + ellipsesCount < indexers.length)
      throw InvalidIndexerException(
        s"Provided indexing sequence (${indexers.mkString(", ")}) is too large for shape $shape.")
    val oldDimensions = Array.ofDim[Int](newRank)
    val dimensions = Array.ofDim[Int](newRank)
    val beginOffsets = Array.ofDim[Int](newRank)
    val endOffsets = Array.ofDim[Int](newRank)
    val strides = Array.ofDim[Int](newRank)
    var i: Int = 0
    var newAxesCounter: Int = 0
    var ellipsisFound = false
    while (i < indexers.length && !ellipsisFound) {
      val oldDimSize = shape(i - newAxesCounter)
      indexers(i) match {
        case Ellipsis =>
          var j: Int = newRank - 1
          newAxesCounter = 0
          while (indexers.length - newRank + j > i) {
            val oldDimSize = shape(shape.rank - newRank + j + newAxesCounter)
            indexers(indexers.length - newRank + j) match {
              case Ellipsis =>
                throw InvalidIndexerException("Only one ellipsis ('---') is allowed per indexing sequence.")
              case NewAxis =>
                beginOffsets(j) = 0
                endOffsets(j) = 1
                strides(j) = 1
                dimensions(j) = 1
                oldDimensions(j) = 1
                newAxesCounter += 1
              case Index(index) =>
                beginOffsets(j) = if (index < 0) index + oldDimSize else index
                endOffsets(j) = beginOffsets(j) + 1
                strides(j) = 1
                dimensions(j) = 1
                oldDimensions(j) = oldDimSize
              case s@Slice(begin, end, step, inclusive) =>
                beginOffsets(j) = if (begin < 0) begin + oldDimSize else begin
                val effectiveEnd = if (end < 0) end + oldDimSize else end
                endOffsets(j) = if (inclusive) effectiveEnd + 1 else effectiveEnd
                strides(j) = step
                dimensions(j) = s.length(oldDimSize)
                oldDimensions(j) = oldDimSize
            }
            if (beginOffsets(j) < 0 || beginOffsets(j) >= oldDimSize)
              throw InvalidIndexerException(
                s"Indexer '${indexers(j)}' is invalid for a dimension with size '$oldDimSize'.")
            if (endOffsets(j) < 0 || endOffsets(j) > oldDimSize)
              throw InvalidIndexerException(
                s"Indexer '${indexers(j)}' is invalid for a dimension with size '$oldDimSize'.")
            j -= 1
          }
          while (j >= i) {
            beginOffsets(j) = 0
            endOffsets(j) = shape(shape.rank - newRank + j + newAxesCounter)
            strides(j) = 1
            dimensions(j) = shape(shape.rank - newRank + j + newAxesCounter)
            oldDimensions(j) = dimensions(j)
            j -= 1
          }
          ellipsisFound = true
        case NewAxis =>
          beginOffsets(i) = 0
          endOffsets(i) = 1
          strides(i) = 1
          dimensions(i) = 1
          oldDimensions(i) = 1
          newAxesCounter += 1
        case Index(index) =>
          beginOffsets(i) = if (index < 0) index + oldDimSize else index
          endOffsets(i) = beginOffsets(i) + 1
          strides(i) = 1
          dimensions(i) = 1
          oldDimensions(i) = oldDimSize
        case s@Slice(begin, end, step, inclusive) =>
          beginOffsets(i) = if (begin < 0) begin + oldDimSize else begin
          val effectiveEnd = if (end < 0) end + oldDimSize else end
          endOffsets(i) = if (inclusive) effectiveEnd + 1 else effectiveEnd
          strides(i) = step
          dimensions(i) = s.length(oldDimSize)
          oldDimensions(i) = oldDimSize
      }
      if (!ellipsisFound && (beginOffsets(i) < 0 || beginOffsets(i) >= oldDimSize))
        throw InvalidIndexerException(
          s"Indexer '${indexers(i)}' is invalid for a dimension with size '$oldDimSize'.")
      if (!ellipsisFound && (endOffsets(i) < 0 || endOffsets(i) > oldDimSize))
        throw InvalidIndexerException(
          s"Indexer '${indexers(i)}' is invalid for a dimension with size '$oldDimSize'.")
      i += 1
    }
    if (!ellipsisFound) {
      while (i < newRank) {
        beginOffsets(i) = 0
        endOffsets(i) = shape(i - newAxesCounter)
        strides(i) = 1
        dimensions(i) = shape(i - newAxesCounter)
        oldDimensions(i) = dimensions(i)
        i += 1
      }
    }
    (oldDimensions, dimensions, beginOffsets, endOffsets, strides)
  }

  /** Converts a sequence of indexers into a function that takes an [[Output]], applies the strided slice native op on
    * it for the provided sequence of indexers, and returns a new (indexed) [[Output]].
    *
    * Note that `indexers` is only allowed to contain at most one [[Ellipsis]].
    *
    * @param  indexers Sequence of indexers to convert.
    * @return Function that indexes an [[Output]] and returns a new (indexed) [[Output]].
    */
  private[api] def toStridedSlice(indexers: Indexer*): Output => Output = {
    if (indexers.count(_ == Ellipsis) > 1)
      throw InvalidIndexerException("Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
    val begin = Tensor.fill(dataType = INT32, shape = Shape(indexers.length))(value = 0)
    val end = Tensor.fill(dataType = INT32, shape = Shape(indexers.length))(value = 0)
    val strides = Tensor.fill(dataType = INT32, shape = Shape(indexers.length))(value = 0)
    var beginMask: Int = 0 // TODO: Use this.
    var endMask: Int = 0
    var ellipsisMask: Int = 0
    var newAxisMask: Int = 0
    var shrinkAxisMask: Int = 0
    indexers.zipWithIndex foreach {
      case (Ellipsis, i) =>
        ellipsisMask |= (1 << i)
      case (NewAxis, i) =>
        newAxisMask |= (1 << i)
      case (Index(index), i) =>
        begin(i).fill(index)
        end(i).fill(index + 1)
        shrinkAxisMask |= (1 << i)
      case (Slice(sliceBegin, sliceEnd, sliceStep, false), i) =>
        begin(i).fill(sliceBegin)
        end(i).fill(sliceEnd)
        strides(i).fill(sliceStep)
      case (Slice(sliceBegin, sliceEnd, sliceStep, true), i) =>
        begin(i).fill(sliceBegin)
        end(i).fill(sliceEnd + 1)
        strides(i).fill(sliceStep)
        if (sliceEnd == -1)
          endMask |= (1 << i)
    }
    input: Output =>
      Basic.stridedSlice(
        input = input,
        begin = Basic.constant(begin),
        end = Basic.constant(end),
        strides = Basic.constant(strides),
        beginMask = beginMask,
        endMask = endMask,
        ellipsisMask = ellipsisMask,
        newAxisMask = newAxisMask,
        shrinkAxisMask = shrinkAxisMask)
  }
}

/** Ellipsis indexer used to represent a full slice over multiple dimensions of a tensor. Ellipses are used to represent
  * zero or more dimensions of a full-dimension indexer sequence. Usage examples are providedin the documentation of
  * [[Indexer]]. */
object Ellipsis extends Indexer {
  override def toString: String = "---"
}

/** New axis indexer used to represent the addition of a new axis in a sequence of indexers. Usage examples are provided
  * in the documentation of [[Indexer]]. */
object NewAxis extends Indexer {
  override def toString: String = "NewAxis"
}

case class Index private[api] (index: Int) extends Indexer {
  // TODO: Add an 'assertWithinBounds' method.
  override def toString: String = index.toString
}

// TODO: Assertions and index computations may be doable in a more efficient manner, in this class.
// TODO: Maybe use options for the one number case (for the 'end' argument). This would ensure more type safety.
/** Represents a slice object. A slice is a sequence of indices for some underlying sequence or tensor.
  *
  * The companion object provides some helpful implicit conversion functions that allow for easy creation of slices.
  *
  * For example:
  * {{{
  *   3               // Slice object representing the following indices: '[3]'.
  *   2 :: 6          // Slice object representing the following indices: '[2, 3, 4, 5]'.
  *   3 :: 4 :: 12    // Slice object representing the following indices: '[3, 7, 10]'.
  *   0 :: 1 :: -2    // Slice object representing the following indices: '[0, 3, 6, ..., length - 3]'.
  *   6 :: -2 :: 3    // Slice object representing the following indices: '[6, 4]'.
  *   -3 :: -2 :: -10 // Slice object representing the following indices: '[length - 3, length - 5, ..., length - 10]'.
  *   2 :: -          // Slice object representing the following indices: '[2, 3, ..., length - 1]'.
  *   0 :: 1 :: -     // Slice object representing the following indices: '[0, 3, 6, ..., length - 1]'.
  * }}}
  *
  * Note how the end index is exclusive and the step size is optional.
  *
  * @param  start         Start index for this slice.
  * @param  end           End index for this slice.
  * @param  step          Step for this slice.
  * @param  inclusive     Boolean value indicating whether this slice inclusive with respect to `end` or not.
  */
case class Slice private[api] (start: Int, end: Int, step: Int = 1, inclusive: Boolean = false) extends Indexer {
  if (step == 0)
    throw InvalidIndexerException("A slice step must not be '0'.")

  /** Helper method that returns the `end` value for this slice, if the slice was considered to be `exclusive`. This
    * method is used by other methods in this class.
    *
    * @param  underlyingSequenceLength Underlying sequence length.
    * @return Exclusive `end` value.
    */
  private[api] def exclusiveEnd(underlyingSequenceLength: Int = -1): Int = {
    if (inclusive) {
      if (step > 0) {
        if (end == -1)
          underlyingSequenceLength
        else
          end + 1
      } else {
        if (end == 0)
          underlyingSequenceLength
        else
          end - 1
      }
    } else {
      end
    }
  }

  /** Length of this slice.
    *
    * If `end - start` has the opposite sign of `step`, then the length of this slice is considered unknown and a value
    * of `-1` is returned. This can be avoided by providing the underlying sequence length in the [[length]] method of
    * this class.
    */
  def length: Int = length(underlyingSequenceLength = -1)

  /** Length of this slice.
    *
    * The length of a slice can only be computed if:
    *   - `end - start` has the same sign as `step`, or
    *   - the underlying sequence length is provided. Note that by underlying sequence we refer to the sequence that
    * this slice is used to index.
    *
    * @param  underlyingSequenceLength Underlying sequence length.
    * @throws IllegalArgumentException  If the underlying sequence length is not positive, or if the slice is invalid
    *                                   for some reason.
    * @throws IndexOutOfBoundsException If the indices specified by this slice are not within the valid size bounds for
    *                                   the provided underlying sequence length.
    */
  @throws[IllegalArgumentException]
  @throws[IndexOutOfBoundsException]
  def length(underlyingSequenceLength: Int = -1): Int = {
    if (underlyingSequenceLength != -1 && underlyingSequenceLength <= 0)
      throw new IllegalArgumentException(
        s"The underlying sequence length, '$underlyingSequenceLength' must be a positive integral number or unknown " +
            s"(i.e., '-1').")
    val end = exclusiveEnd(underlyingSequenceLength)
    if (underlyingSequenceLength < 0) {
      val result = Slice.ceilDiv(end - start, step)
      if (result >= 0)
        result
      else if ((start > 0 && end > 0 && step < 0) || (start < 0 && end < 0 && step > 0))
        throw new IllegalArgumentException(
          s"Slice '$this' is invalid. It can never get to its end from its start, using the specified step.")
      else
        throw new IllegalArgumentException(
          s"Slice '$this' length cannot be inferred without knowing the underlying sequence length.")
    } else {
      assertWithinBounds(underlyingSequenceLength)
      val floorStart = Math.floorMod(start, underlyingSequenceLength)
      val floorEnd = if (end < underlyingSequenceLength) Math.floorMod(end, underlyingSequenceLength) else end
      val result = Slice.ceilDiv(floorEnd - floorStart, step)
      if (result >= 0)
        result
      else
        throw new IllegalArgumentException(
          s"For the provided sequence length of '$underlyingSequenceLength', slice '$this' is invalid. It can never " +
              s"get to its end from its start, using the specified step.")
    }
  }

  /** Returns an array containing the indices represented by this slice, for the specified underlying sequence length.
    *
    * This method takes case of negative indexing and the returned array is guaranteed to only have positive elements.
    *
    * @param  underlyingSequenceLength Underlying sequence length.
    * @return Array containing the indices represented by this slice, for the specified underlying sequence length.
    * @throws IllegalArgumentException  If the underlying sequence length is not positive.
    * @throws IndexOutOfBoundsException If the indices specified by this slice are not within the valid size bounds for
    *                                   the provided underlying sequence length.
    */
  @throws[IllegalArgumentException]
  @throws[IndexOutOfBoundsException]
  def toArray(underlyingSequenceLength: Int): Array[Int] = {
    assertWithinBounds(underlyingSequenceLength)
    val start = Math.floorMod(this.start, underlyingSequenceLength)
    val end = {
      val exclusiveEnd = this.exclusiveEnd(underlyingSequenceLength)
      if (exclusiveEnd < underlyingSequenceLength)
        Math.floorMod(exclusiveEnd, underlyingSequenceLength)
      else
        exclusiveEnd
    }
    if (((end - start) > 0 && step < 0) || ((end - start) < 0 && step > 0))
      throw new IllegalArgumentException(
        s"For the provided sequence length of '$underlyingSequenceLength', slice '$this' is invalid. It can never " +
            s"get to its end from its start, using the specified step.")
    start until end by step toArray
  }

  /** Asserts that the indices specified by this slice are within the valid size bounds for the provided underlying
    * sequence length. Throws an [[IndexOutOfBoundsException]] exception if they are not.
    *
    * @param  underlyingSequenceLength Underlying sequence length.
    * @throws IllegalArgumentException  If the underlying sequence length is not positive.
    * @throws IndexOutOfBoundsException If the indices specified by this slice are not within the valid size bounds for
    *                                   the provided underlying sequence length.
    */
  @throws[IllegalArgumentException]
  @throws[IndexOutOfBoundsException]
  def assertWithinBounds(underlyingSequenceLength: Int): Unit = {
    if (underlyingSequenceLength <= 0)
      throw new IllegalArgumentException(
        s"The underlying sequence length, '$underlyingSequenceLength', must be a positive integral number.")
    if (Math.abs(start) >= underlyingSequenceLength)
      throw new IndexOutOfBoundsException(
        s"Slice start index '$start' is outside the bounds for a sequence length of '$underlyingSequenceLength'.")
    val end = exclusiveEnd(underlyingSequenceLength)
    if (end > underlyingSequenceLength || end <= -underlyingSequenceLength)
      throw new IndexOutOfBoundsException(
        s"Slice end index '$end' is outside the bounds for a sequence length of '$underlyingSequenceLength'.")
  }

  override def toString: String = {
    if (inclusive && start == 0 && end == -1 && step == 1) "::"
    else if (step == 1) s"[$start::$end" + (if (inclusive) "]" else ")")
    else s"[$start::$step::$end" + (if (inclusive) "]" else ")")
  }
}

/** Contains helper functions for creating [[Slice]] objects. */
object Slice {
  private def ceilDiv(numerator: Int, denominator: Int) = {
    numerator / denominator + (if (numerator % denominator == 0) 0 else 1)
  }

  /** Returns a slice object representing all indices. */
  private[api] val :: = Slice(start = 0, end = -1, inclusive = true)
}
