package org.platanios.tensorflow.api

import Slice._

import scala.collection.mutable.ArrayBuffer

// TODO: Support integer slices.
// TODO: Assertions and index computations may be doable in a more efficient manner, in this class.
/** Represents a slice object. A slice is a sequence of indices for some underlying sequence or tensor.
  *
  * The companion object provides some helpful implicit conversion functions that allow for easy creation of slices.
  *
  * For example:
  * {{{
  *   3               // Slice object representing the following indices: '[3]'.
  *   2 :: 6          // Slice object representing the following indices: '[2, 3, 4, 5]'.
  *   3 :: 4 :: 12    // Slice object representing the following indices: '[3, 7, 10]'.
  *   0 :: 1 :: -2    // Slice object representing the following indices: '[0, 3, 6, ..., length - 2]'.
  *   6 :: -2 :: 3    // Slice object representing the following indices: '[6, 4]'.
  *   -3 :: -2 :: -10 // Slice object representing the following indices: '[length - 3, length - 5, ..., length - 9]'.
  * }}}
  *
  * Note how the end index is exclusive and the step size is optional.
  *
  * @note This class supports [[Int]]-based and [[Long]]-based slice creation, but the underlying representation always
  *       uses [[Long]] because the TensorFlow library shapes are defined using [[Long]]s.
  *
  * @param  start Start index for this slice.
  * @param  end   End index for this slice.
  * @param  step  Step for this slice.
  *
  * @author Emmanouil Antonios Platanios
  */
case class Slice(start: Long, end: Long, step: Long = 1) {
  /** Length of this slice.
    *
    * If `end - start` has the opposite sign of `step`, then the length of this slice is considered unknown and a value
    * of `-1` is returned. This can be avoided by providing the underlying sequence length in the [[length]] method of
    * this class.
    */
  def length: Long = length(underlyingSequenceLength = -1)

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
  def length(underlyingSequenceLength: Long = -1): Long = {
    if (underlyingSequenceLength != -1 && underlyingSequenceLength <= 0)
      throw new IllegalArgumentException(
        s"The underlying sequence length, '$underlyingSequenceLength' must be a positive integral number or unknown " +
            s"(i.e., '-1').")
    if (underlyingSequenceLength < 0) {
      val result = ceilDiv(end - start, step)
      if (result >= 0)
        result
      else if ((start > 0 && end > 0 && step < 0) || (start < 0 && end < 0 && step > 0))
        throw new IllegalArgumentException(
          s"Slice '$this' is invalid. It can never get to its end from its start, using the specified step.")
      else
        -1
    } else {
      assertWithinBounds(underlyingSequenceLength)
      val floorStart = Math.floorMod(start, underlyingSequenceLength)
      val floorEnd = Math.floorMod(end, underlyingSequenceLength)
      val result = ceilDiv(floorEnd - floorStart, step)
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
  def toArray(underlyingSequenceLength: Long): Array[Long] = {
    assertWithinBounds(underlyingSequenceLength)
    var start = Math.floorMod(this.start, underlyingSequenceLength)
    val end = Math.floorMod(this.end, underlyingSequenceLength)
    if (((end - start) > 0 && step < 0) || ((end - start) < 0 && step > 0))
      throw new IllegalArgumentException(
        s"For the provided sequence length of '$underlyingSequenceLength', slice '$this' is invalid. It can never " +
            s"get to its end from its start, using the specified step.")
    val resultArrayBuffer = ArrayBuffer[Long](start)
    start += step
    while ((start < end && step > 0) || (start > end && step < 0)) {
      resultArrayBuffer += start
      start += step
    }
    resultArrayBuffer.toArray
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
  def assertWithinBounds(underlyingSequenceLength: Long): Unit = {
    if (underlyingSequenceLength <= 0)
      throw new IllegalArgumentException(
        s"The underlying sequence length, '$underlyingSequenceLength', must be a positive integral number.")
    if (Math.abs(start) >= underlyingSequenceLength)
      throw new IndexOutOfBoundsException(
        s"Slice start index '$start' is outside the bounds for a sequence length of '$underlyingSequenceLength'.")
    if (Math.abs(end) >= underlyingSequenceLength)
      throw new IndexOutOfBoundsException(
        s"Slice end index '$end' is outside the bounds for a sequence length of '$underlyingSequenceLength'.")
  }

  override def toString: String = s"($start::$step::$end)"
}

/** Helper trait for representing slice construction phases. */
sealed trait SliceConstruction

/** Helper class for representing a slice construction phase that has already been provided three numbers. */
case class SliceWithThreeNumbers private (n1: Long, n2: Long, n3: Long = 1) extends SliceConstruction

/** Helper class for representing a slice construction phase that has already been provided two numbers. */
case class SliceWithTwoNumbers private (n1: Long, n2: Long) extends SliceConstruction {
  def ::(slice: SliceWithOneNumber): SliceWithThreeNumbers = SliceWithThreeNumbers(slice.n, n1, n2)
}

/** Helper class for representing a slice construction phase that has already been provided one numbers. */
case class SliceWithOneNumber private (n: Long) extends SliceConstruction {
  def ::(slice: SliceWithOneNumber): SliceWithTwoNumbers = SliceWithTwoNumbers(slice.n, n)
}

/** Contains helper functions for creating [[Slice]] objects. */
object Slice {
  private def ceilDiv(numerator: Long, denominator: Long) = {
    numerator / denominator + (if (numerator % denominator == 0) 0 else 1)
  }

  /** Returns a slice object representing all indices. */
  private[api] def :: = Slice(start = 0, end = -1)

  private[api] implicit def intToSlice(int: Int): Slice = Slice(start = int, end = int + 1)
  private[api] implicit def longToSlice(long: Long): Slice = Slice(start = long, end = long + 1)
  private[api] implicit def intToSliceWithOneNumber(int: Int): SliceWithOneNumber = SliceWithOneNumber(int)
  private[api] implicit def longToSliceWithOneNumber(long: Long): SliceWithOneNumber = SliceWithOneNumber(long)

  private[api] implicit def sliceConstructionToSlice(sliceConstruction: SliceConstruction): Slice =
    sliceConstruction match {
      case SliceWithOneNumber(start) => Slice(start = start, end = start + 1)
      case SliceWithTwoNumbers(start, end) => Slice(start = start, end = end)
      case SliceWithThreeNumbers(start, step, end) => Slice(start = start, end = end, step = step)
    }
}
