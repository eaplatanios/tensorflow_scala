package org.platanios.tensorflow.api

import Slice._
import org.platanios.tensorflow.api.Exception.InvalidIndexerException
import org.platanios.tensorflow.api.ops.ArrayOps

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
  *   // 't' is a tensor (i.e., Op.Output) with shape [4, 2, 3, 8]
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

/** Helper class for representing a indexer construction phase that has already been provided two numbers. */
case class IndexerConstructionWithTwoNumbers private (n1: Int, n2: Int) extends IndexerConstruction {
  def :: : Slice = Slice(start = n1, end = -1, step = n2, inclusive = true)

  def ::(leftSide: IndexerConstructionWithOneNumber): IndexerConstructionWithThreeNumbers = {
    IndexerConstructionWithThreeNumbers(leftSide.n, n1, n2)
  }
}

/** Helper class for representing a indexer construction phase that has already been provided three numbers. */
case class IndexerConstructionWithThreeNumbers private (n1: Int, n2: Int, n3: Int) extends IndexerConstruction

/** Contains helper functions for dealing with indexers. */
object Indexer {
  private[api] val --- : Indexer = Ellipsis

  //region Implicits

  // TODO: Add begin mask support (this one is tough).

  private[api] implicit def intToIndex(index: Int): Index = Index(index = index)
  private[api] implicit def intToIndexerConstructionWithOneNumber(n: Int): IndexerConstructionWithOneNumber =
    IndexerConstructionWithOneNumber(n)
  private[api] implicit def indexerConstructionWithOneNumberToIndex(
      construction: IndexerConstructionWithOneNumber): Index =
    Index(index = construction.n)
  private[api] implicit def indexerConstructionWithTwoNumbersToSlice(
      construction: IndexerConstructionWithTwoNumbers): Slice =
    Slice(start = construction.n1, end = construction.n2)
  private[api] implicit def indexerConstructionWithThreeNumbersToSlice(
      construction: IndexerConstructionWithThreeNumbers): Slice =
    Slice(start = construction.n1, end = construction.n3, step = construction.n2)

  //endregion Implicits

  /** Converts a sequence of indexers into a function that takes an [[Op.Output]], applies the strided slice native op
    * on it for the provided sequence of indexers, and returns a new (indexed) [[Op.Output]].
    *
    * Note that `indexers` is only allowed to contain at most one [[Ellipsis]].
    *
    * @param  indexers Sequence of indexers to convert.
    * @return Function that indexes an [[Op.Output]] and returns a new (indexed) [[Op.Output]].
    */
  private[api] def toStridedSlice(indexers: Indexer*): Op.Output => Op.Output = {
    if (indexers.count(_ == Ellipsis) > 1)
      throw InvalidIndexerException("Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
    val begin = Array.fill[Int](indexers.length)(0)
    val end = Array.fill[Int](indexers.length)(0)
    val strides = Array.fill[Int](indexers.length)(1)
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
        begin(i) = index
        end(i) = index + 1
        shrinkAxisMask |= (1 << i)
      case (Slice(sliceBegin, sliceEnd, sliceStep, false), i) =>
        begin(i) = sliceBegin
        end(i) = sliceEnd
        strides(i) = sliceStep
      case (Slice(sliceBegin, sliceEnd, sliceStep, true), i) =>
        begin(i) = sliceBegin
        end(i) = sliceEnd + 1
        strides(i) = sliceStep
        if (sliceEnd == -1)
          endMask |= (1 << i)
    }
    input: Op.Output =>
      ArrayOps.stridedSlice(
        input = input,
        begin = ArrayOps.constant(begin),
        end = ArrayOps.constant(end),
        strides = ArrayOps.constant(strides),
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
object Ellipsis extends Indexer

/** New axis indexer used to represent the addition of a new axis in a sequence of indexers. Usage examples are provided
  * in the documentation of [[Indexer]]. */
object NewAxis extends Indexer

case class Index private[api] (index: Int) extends Indexer

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
      val result = ceilDiv(end - start, step)
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

  override def toString: String = s"[$start::$step::$end" + (if (inclusive) "]" else ")")
}

/** Contains helper functions for creating [[Slice]] objects. */
object Slice {
  private def ceilDiv(numerator: Int, denominator: Int) = {
    numerator / denominator + (if (numerator % denominator == 0) 0 else 1)
  }

  /** Returns a slice object representing all indices. */
  private[api] val :: = Slice(start = 0, end = -1, inclusive = true)
}
