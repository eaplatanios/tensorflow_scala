package org.platanios.tensorflow.api

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class SliceSpec extends FlatSpec with Matchers {
  private def sliceImplicitsHelper(slice: Slice): Slice = slice

  "Slice construction" must "work when specifying a single index" in {
    assert(sliceImplicitsHelper(45) === Slice(start = 45, end = 46))
    assert(sliceImplicitsHelper(2) === Slice(start = 2, end = 3))
  }

  it must "work when specifying a start and an end" in {
    assert(sliceImplicitsHelper(45 :: 67) === Slice(start = 45, end = 67))
    assert(sliceImplicitsHelper(2 :: 34) === Slice(start = 2, end = 34))
  }

  it must "work when specifying a start and an end and a step" in {
    assert(sliceImplicitsHelper(45 :: 2 :: 67) === Slice(start = 45, end = 67, step = 2))
    assert(sliceImplicitsHelper(2 :: 5 :: 34) === Slice(start = 2, end = 34, step = 5))
  }

  it must "work when specifying a complete slice" in {
    assert(sliceImplicitsHelper(::) === Slice(start = 0, end = -1))
  }

  it must "fail to compile in all other cases" in {
    assertDoesNotCompile("val s: Slice = 45 :: 67 :: 34 :: 23")
    assertDoesNotCompile("val s: Slice = :: 23")
    assertDoesNotCompile("val s: Slice = 32 ::")
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

  it must "be unknown when `end - start` and `step` have the opposite signs and no sequence length is provided" in {
    assert((3 :: -1).length === -1)
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

    // Check using the 'Slice.toArray' method (this is checking for self-consistency of the Slice class)
    assert((45 :: 46).length(175) === (45 :: 46).toArray(175).length)
    assert((45 :: 67).length(175) === (45 :: 67).toArray(175).length)
    assert((-12 :: -5).length(175) === (-12 :: -5).toArray(175).length)
    assert((45 :: 2 :: 67).length(175) === (45 :: 2 :: 67).toArray(175).length)
    assert((-12 :: 2 :: -5).length(175) === (-12 :: 2 :: -5).toArray(175).length)
    assert((12 :: -2 :: 5).length(175) === (12 :: -2 :: 5).toArray(175).length)
    assert((45 :: 3 :: 67).length(175) === (45 :: 3 :: 67).toArray(175).length)
    assert((3 :: -1).length(175) === (3 :: -1).toArray(175).length)
  }

  it must "throw appropriate exceptions when errors occur" in {
    assert(intercept[IllegalArgumentException]((5 :: -1 :: 7).length).getMessage ===
               "Slice '(5::-1::7)' is invalid. It can never get to its end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-6 :: 2 :: -8).length).getMessage ===
               "Slice '(-6::2::-8)' is invalid. It can never get to its end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-3 :: 1).length(175)).getMessage ===
               "For the provided sequence length of '175', slice '(-3::1::1)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IndexOutOfBoundsException]((4 :: 3 :: -10).length(8)).getMessage ===
               "Slice end index '-10' is outside the bounds for a sequence length of '8'.")
  }

  "'toArray'" must "return an array with the indices represented by this slice, for the underlying sequence" in {
    assert((45 :: 46).toArray(175) === Array[Long](45))
    assert((45 :: 67).toArray(175) === (45 until 67).map(_.asInstanceOf[Long]).toArray)
    assert((-12 :: -5).toArray(175) === (163 until 170).map(_.asInstanceOf[Long]).toArray)
    assert((45 :: 2 :: 67).toArray(175) === (45 until 67 by 2).map(_.asInstanceOf[Long]).toArray)
    assert((-12 :: 2 :: -5).toArray(175) === (163 until 170 by 2).map(_.asInstanceOf[Long]).toArray)
    assert((12 :: -2 :: 5).toArray(175) === (12 until 5 by -2).map(_.asInstanceOf[Long]).toArray)
    assert((45 :: 3 :: 67).toArray(175) === (45 until 67 by 3).map(_.asInstanceOf[Long]).toArray)
    assert((3 :: -1).toArray(175) === (3 until 174).map(_.asInstanceOf[Long]).toArray)
  }

  it must "throw appropriate exceptions when errors occur" in {
    assert(intercept[IllegalArgumentException]((5 :: -1 :: 7).toArray(175)).getMessage ===
               "For the provided sequence length of '175', slice '(5::-1::7)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-6 :: 2 :: -8).toArray(175)).getMessage ===
               "For the provided sequence length of '175', slice '(-6::2::-8)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-3 :: 1).toArray(175)).getMessage ===
               "For the provided sequence length of '175', slice '(-3::1::1)' is invalid. It can never get to its " +
                   "end from its start, using the specified step.")
    assert(intercept[IndexOutOfBoundsException]((4 :: 3 :: -10).toArray(8)).getMessage ===
               "Slice end index '-10' is outside the bounds for a sequence length of '8'.")
  }

  "'assertWithinBounds'" must "throw an 'IndexOutOfBoundsException' exception whenever appropriate" in {
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
}
