package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.Exception.InvalidIndexerException
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class IndexerSpec extends FlatSpec with Matchers {
  private def indexerImplicitsHelper(indexer: Indexer): Indexer = indexer

  "Ellipsis" must "be representable using '---'" in {
    assert(Ellipsis === ---)
  }

  "NewAxis" must "be representable using 'NewAxis'" in {
    assert(NewAxis === NewAxis) // TODO: Redundant test. Maybe test usage instead?
  }

  "Indexer construction" must "work when specifying a single index" in {
    assert(indexerImplicitsHelper(45) === Index(index = 45))
    assert(indexerImplicitsHelper(2) === Index(index = 2))
  }

  it must "work when specifying a start and an end" in {
    assert(indexerImplicitsHelper(45 :: 67) === Slice(start = 45, end = 67))
    assert(indexerImplicitsHelper(2 :: 34) === Slice(start = 2, end = 34))
  }

  it must "work when specifying a start and an end and a step" in {
    assert(indexerImplicitsHelper(45 :: 2 :: 67) === Slice(start = 45, end = 67, step = 2))
    assert(indexerImplicitsHelper(2 :: 5 :: 34) === Slice(start = 2, end = 34, step = 5))
  }

  it must "work when specifying a complete slice" in {
    assert(indexerImplicitsHelper(::) === Slice(start = 0, end = -1, inclusive = true))
  }

  it must "work when no 'end' is provided" in {
    assert(indexerImplicitsHelper(45 ::) === Slice(start = 45, end = -1, inclusive = true))
    assert(indexerImplicitsHelper(34 :: 2 ::) === Slice(start = 34, step = 2, end = -1, inclusive = true))
  }

  it must "fail to compile in all other cases" in {
    assertDoesNotCompile("val s: Slice = 45 :: 67 :: 34 :: 23")
    assertDoesNotCompile("val s: Slice = :: 23")
    assertDoesNotCompile("val s: Slice = 32 :: 2 :: 1 ::")
  }

  it must "throw an 'InvalidSliceException' when a step size of 0 is provided" in {
    assert(intercept[InvalidIndexerException](Slice(start = 0, end = 0, step = 0)).getMessage ===
               "A slice step must not be '0'.")
    assert(intercept[InvalidIndexerException](indexerImplicitsHelper(5 :: 0 :: 7)).getMessage ===
               "A slice step must not be '0'.")
  }

  "The slice length" must "be computed correctly when `end - start` and `step` have the same sign" in {
    assert((45 :: 46).length === 1)
    assert((45 :: 67).length === 22)
    assert((-12 :: -5).length === 7)
    assert((-3 :: 1).length === 4)
  }

  it must "be computed correctly when `end - start` and `step` have the same sign and a step is provided" in {
    assert((45 :: 2 :: 67).length === 11)
    assert((-12 :: 2 :: -5).length === 4)
    assert((12 :: -2 :: 5).length === 4)
    assert((45 :: 3 :: 67).length === 8)
  }

  it must "be computed correctly when the underlying sequence length is provided" in {
    // Check using the ground truth length
    assert((45 :: 46).length(175) === 1)
    assert((45 :: 67).length(175) === 22)
    assert((-12 :: -5).length(175) === 7)
    assert((45 :: 2 :: 67).length(175) === 11)
    assert((-12 :: 2 :: -5).length(175) === 4)
    assert((12 :: -2 :: 5).length(175) === 4)
    assert((45 :: 3 :: 67).length(175) === 8)
    assert((3 :: -1).length(175) === 171)
    assert(::.length(175) === 175)

    // Check using the 'Slice.toArray' method (this is checking for self-consistency of the Slice class)
    assert((45 :: 46).length(175) === (45 :: 46).toArray(175).length)
    assert((45 :: 67).length(175) === (45 :: 67).toArray(175).length)
    assert((-12 :: -5).length(175) === (-12 :: -5).toArray(175).length)
    assert((45 :: 2 :: 67).length(175) === (45 :: 2 :: 67).toArray(175).length)
    assert((-12 :: 2 :: -5).length(175) === (-12 :: 2 :: -5).toArray(175).length)
    assert((12 :: -2 :: 5).length(175) === (12 :: -2 :: 5).toArray(175).length)
    assert((45 :: 3 :: 67).length(175) === (45 :: 3 :: 67).toArray(175).length)
    assert((3 :: -1).length(175) === (3 :: -1).toArray(175).length)
    assert(::.length(175) === ::.toArray(175).length)
  }

  it must "be computed correctly when no 'end' is provided" in {
    assert(Slice(start = 0, end = 0, inclusive = true).length === 1)
    assert(Slice(start = 45, end = 46, inclusive = true).length === 2)
    assert(Slice(start = 45, end = 67, inclusive = true).length === 23)
    assert(Slice(start = -3, end = 2, inclusive = true).length === 6)
    assert(Slice(start = -3, step = 2, end = 2, inclusive = true).length === 3)
    assert(Slice(start = 3, end = -1, inclusive = true).length(175) === 172)
  }

  it must "throw an exception when an error occurs" in {
    assert(intercept[IllegalArgumentException]((3 :: -1).length).getMessage ===
               "Slice '[3::1::-1)' length cannot be inferred without knowing the underlying sequence length.")
    assert(intercept[IllegalArgumentException]((5 :: -1 :: 7).length).getMessage ===
               "Slice '[5::-1::7)' is invalid. It can never get to its end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-6 :: 2 :: -8).length).getMessage ===
               "Slice '[-6::2::-8)' is invalid. It can never get to its end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-3 :: 1).length(175)).getMessage ===
               "For the provided sequence length of '175', slice '[-3::1::1)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IndexOutOfBoundsException]((4 :: 3 :: -10).length(8)).getMessage ===
               "Slice end index '-10' is outside the bounds for a sequence length of '8'.")
  }

  "'Slice.toArray'" must "return an array with the indices represented by this slice, for the underlying sequence" in {
    assert((45 :: 46).toArray(175) === Array(45))
    assert((45 :: 67).toArray(175) === (45 until 67).toArray)
    assert((-12 :: -5).toArray(175) === (163 until 170).toArray)
    assert((45 :: 2 :: 67).toArray(175) === (45 until 67 by 2).toArray)
    assert((-12 :: 2 :: -5).toArray(175) === (163 until 170 by 2).toArray)
    assert((12 :: -2 :: 5).toArray(175) === (12 until 5 by -2).toArray)
    assert((45 :: 3 :: 67).toArray(175) === (45 until 67 by 3).toArray)
    assert((3 :: -1).toArray(175) === (3 until 174).toArray)
    assert(::.toArray(175) === (0 to 174).toArray)
    assert(Slice(start = 45, end = 46, inclusive = true).toArray(175) === (45 to 46).toArray)
    assert(Slice(start = 45, end = 67, inclusive = true).toArray(175) === (45 to 67).toArray)
    assert(Slice(start = -12, end = -5, inclusive = true).toArray(175) === (163 to 170).toArray)
    assert(Slice(start = 45, end = 67, step = 2, inclusive = true).toArray(175) === (45 to 67 by 2).toArray)
    assert(Slice(start = -12, end = -5, step = 2, inclusive = true).toArray(175) === (163 to 170 by 2).toArray)
    assert(Slice(start = 12, end = 5, step = -2, inclusive = true).toArray(175) === (12 to 5 by -2).toArray)
    assert(Slice(start = 45, end = 67, step = 3, inclusive = true).toArray(175) === (45 to 67 by 3).toArray)
    assert(Slice(start = 3, end = -1).toArray(175) === (3 to 173).toArray)
  }

  it must "throw an exception when an error occurs" in {
    assert(intercept[IllegalArgumentException]((5 :: -1 :: 7).toArray(175)).getMessage ===
               "For the provided sequence length of '175', slice '[5::-1::7)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-6 :: 2 :: -8).toArray(175)).getMessage ===
               "For the provided sequence length of '175', slice '[-6::2::-8)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-3 :: 1).toArray(175)).getMessage ===
               "For the provided sequence length of '175', slice '[-3::1::1)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IndexOutOfBoundsException]((4 :: 3 :: -10).toArray(8)).getMessage ===
               "Slice end index '-10' is outside the bounds for a sequence length of '8'.")
  }

  "'Slice.assertWithinBounds'" must "throw an exception exception whenever appropriate" in {
    val slice: Slice = 4 :: 3 :: -10
    assert(intercept[IllegalArgumentException](slice.assertWithinBounds(0)).getMessage ===
               "The underlying sequence length, '0', must be a positive integral number.")
    assert(intercept[IndexOutOfBoundsException](slice.assertWithinBounds(3)).getMessage ===
               "Slice start index '4' is outside the bounds for a sequence length of '3'.")
    assert(intercept[IndexOutOfBoundsException](slice.assertWithinBounds(8)).getMessage ===
               "Slice end index '-10' is outside the bounds for a sequence length of '8'.")
    assert(intercept[IndexOutOfBoundsException](slice.assertWithinBounds(10)).getMessage ===
               "Slice end index '-10' is outside the bounds for a sequence length of '10'.")
  }

  // TODO: Add tests for "toStridedSlice".

  "'Indexer.toStridedSlice'" must "throw an 'InvalidIndexerException' when an ellipsis is used more than once" in {
    assert(Indexer.toStridedSlice(---).isInstanceOf[Op.Output => Op.Output])
    assert(Indexer.toStridedSlice(::, ---).isInstanceOf[Op.Output => Op.Output])
    assert(Indexer.toStridedSlice(0 :: -1, 0 ::, ---, 3 :: -1 :: 1, -1).isInstanceOf[Op.Output => Op.Output])
    assert(intercept[InvalidIndexerException](Indexer.toStridedSlice(---, ---)).getMessage ===
               "Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
    assert(intercept[InvalidIndexerException](Indexer.toStridedSlice(::, ---, 0, ---)).getMessage ===
               "Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
    assert(intercept[InvalidIndexerException](Indexer.toStridedSlice(0 ::, ---, 3 :: -1 :: 1, ---, -1)).getMessage ===
               "Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
  }
}
