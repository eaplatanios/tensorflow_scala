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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.exception.InvalidIndexerException

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
               "Slice '[3::-1)' length cannot be inferred without knowing the underlying sequence length.")
    assert(intercept[IllegalArgumentException]((5 :: -1 :: 7).length).getMessage ===
               "Slice '[5::-1::7)' is invalid. It can never get to its end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-6 :: 2 :: -8).length).getMessage ===
               "Slice '[-6::2::-8)' is invalid. It can never get to its end from its start, using the specified step.")
    assert(intercept[IllegalArgumentException]((-3 :: 1).length(175)).getMessage ===
               "For the provided sequence length of '175', slice '[-3::1)' is invalid. It can never get to its " +
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
               "For the provided sequence length of '175', slice '[-3::1)' is invalid. It can never get to its " +
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

  "'Indexer.decode'" must "work correctly for valid inputs" in {
    val shape1 = Shape(10, 25, 3, 1)
    val index1 = Seq[Indexer](0)
    val (oldDimensions1, dimensions1, beginOffsets1, endOffsets1, strides1) = Indexer.decode(shape1, index1)
    assert(oldDimensions1 === Array(10, 25, 3, 1))
    assert(dimensions1 === Array(1, 25, 3, 1))
    assert(beginOffsets1 === Array(0, 0, 0, 0))
    assert(endOffsets1 === Array(1, 25, 3, 1))
    assert(strides1 === Array(1, 1, 1, 1))
    val index2 = Seq[Indexer](3 :: 6, ---)
    val (oldDimensions2, dimensions2, beginOffsets2, endOffsets2, strides2) = Indexer.decode(shape1, index2)
    assert(oldDimensions2 === Array(10, 25, 3, 1))
    assert(dimensions2 === Array(3, 25, 3, 1))
    assert(beginOffsets2 === Array(3, 0, 0, 0))
    assert(endOffsets2 === Array(6, 25, 3, 1))
    assert(strides2 === Array(1, 1, 1, 1))
    val index3 = Seq[Indexer](---)
    val (oldDimensions3, dimensions3, beginOffsets3, endOffsets3, strides3) = Indexer.decode(shape1, index3)
    assert(oldDimensions3 === Array(10, 25, 3, 1))
    assert(dimensions3 === Array(10, 25, 3, 1))
    assert(beginOffsets3 === Array(0, 0, 0, 0))
    assert(endOffsets3 === Array(10, 25, 3, 1))
    assert(strides3 === Array(1, 1, 1, 1))
    val index4 = Seq[Indexer](---, 1 :: 2, ::)
    val (oldDimensions4, dimensions4, beginOffsets4, endOffsets4, strides4) = Indexer.decode(shape1, index4)
    assert(oldDimensions1 === Array(10, 25, 3, 1))
    assert(dimensions4 === Array(10, 25, 1, 1))
    assert(beginOffsets4 === Array(0, 0, 1, 0))
    assert(endOffsets4 === Array(10, 25, 2, 1))
    assert(strides4 === Array(1, 1, 1, 1))
    val shape2 = Shape(10, 25, 3, 5)
    val index5 = Seq[Indexer](2 :: 2 :: 7, ---, 1 :: 4)
    val (oldDimensions5, dimensions5, beginOffsets5, endOffsets5, strides5) = Indexer.decode(shape2, index5)
    assert(oldDimensions5 === Array(10, 25, 3, 5))
    assert(dimensions5 === Array(3, 25, 3, 3))
    assert(beginOffsets5 === Array(2, 0, 0, 1))
    assert(endOffsets5 === Array(7, 25, 3, 4))
    assert(strides5 === Array(2, 1, 1, 1))
    val index6 = Seq[Indexer](2 :: 2 :: 8, ::, NewAxis, ---, ::, 1 :: 4)
    val (oldDimensions6, dimensions6, beginOffsets6, endOffsets6, strides6) = Indexer.decode(shape2, index6)
    assert(oldDimensions6 === Array(10, 25, 1, 3, 5))
    assert(dimensions6 === Array(3, 25, 1, 3, 3))
    assert(beginOffsets6 === Array(2, 0, 0, 0, 1))
    assert(endOffsets6 === Array(8, 25, 1, 3, 4))
    assert(strides6 === Array(2, 1, 1, 1, 1))
    val index7 = Seq[Indexer](2 :: 2 :: 8, ::, NewAxis, ---, NewAxis, ::, 1 :: 4)
    val (oldDimensions7, dimensions7, beginOffsets7, endOffsets7, strides7) = Indexer.decode(shape2, index7)
    assert(oldDimensions7 === Array(10, 25, 1, 1, 3, 5))
    assert(dimensions7 === Array(3, 25, 1, 1, 3, 3))
    assert(beginOffsets7 === Array(2, 0, 0, 0, 0, 1))
    assert(endOffsets7 === Array(8, 25, 1, 1, 3, 4))
    assert(strides7 === Array(2, 1, 1, 1, 1, 1))
    val index8 = Seq[Indexer](2 :: 2 :: 8, ::, NewAxis, ---, ::, NewAxis, 1 :: 4)
    val (oldDimensions8, dimensions8, beginOffsets8, endOffsets8, strides8) = Indexer.decode(shape2, index8)
    assert(oldDimensions8 === Array(10, 25, 1, 3, 1, 5))
    assert(dimensions8 === Array(3, 25, 1, 3, 1, 3))
    assert(beginOffsets8 === Array(2, 0, 0, 0, 0, 1))
    assert(endOffsets8 === Array(8, 25, 1, 3, 1, 4))
    assert(strides8 === Array(2, 1, 1, 1, 1, 1))
    val index9 = Seq[Indexer](NewAxis)
    val (oldDimensions9, dimensions9, beginOffsets9, endOffsets9, strides9) = Indexer.decode(shape1, index9)
    assert(oldDimensions9 === Array(1, 10, 25, 3, 1))
    assert(dimensions9 === Array(1, 10, 25, 3, 1))
    assert(beginOffsets9 === Array(0, 0, 0, 0, 0))
    assert(endOffsets9 === Array(1, 10, 25, 3, 1))
    assert(strides9 === Array(1, 1, 1, 1, 1))
    val index10 = Seq[Indexer](---, NewAxis)
    val (oldDimensions10, dimensions10, beginOffsets10, endOffsets10, strides10) = Indexer.decode(shape1, index10)
    assert(oldDimensions10 === Array(10, 25, 3, 1, 1))
    assert(dimensions10 === Array(10, 25, 3, 1, 1))
    assert(beginOffsets10 === Array(0, 0, 0, 0, 0))
    assert(endOffsets10 === Array(10, 25, 3, 1, 1))
    assert(strides10 === Array(1, 1, 1, 1, 1))
    val index11 = Seq[Indexer](NewAxis, NewAxis, ---, NewAxis)
    val (oldDimensions11, dimensions11, beginOffsets11, endOffsets11, strides11) = Indexer.decode(shape1, index11)
    assert(oldDimensions11 === Array(1, 1, 10, 25, 3, 1, 1))
    assert(dimensions11 === Array(1, 1, 10, 25, 3, 1, 1))
    assert(beginOffsets11 === Array(0, 0, 0, 0, 0, 0, 0))
    assert(endOffsets11 === Array(1, 1, 10, 25, 3, 1, 1))
    assert(strides11 === Array(1, 1, 1, 1, 1, 1, 1))
    val index12 = Seq[Indexer](-1 :: -2 :: -8, ::, NewAxis, ---, ::, NewAxis, 1 :: 4)
    val (oldDimensions12, dimensions12, beginOffsets12, endOffsets12, strides12) = Indexer.decode(shape2, index12)
    assert(oldDimensions12 === Array(10, 25, 1, 3, 1, 5))
    assert(dimensions12 === Array(4, 25, 1, 3, 1, 3))
    assert(beginOffsets12 === Array(9, 0, 0, 0, 0, 1))
    assert(endOffsets12 === Array(2, 25, 1, 3, 1, 4))
    assert(strides12 === Array(-2, 1, 1, 1, 1, 1))
  }

  it must "throw an 'InvalidIndexerException' for invalid inputs" in {
    val shape = Shape(10, 25, 3, 5)
    val index1 = Seq[Indexer](-1 :: -2 :: -8, ::, NewAxis, ---, ::, ::, NewAxis, 1 :: 4)
    assert(intercept[InvalidIndexerException](Indexer.decode(shape, index1)).getMessage ===
               s"Provided indexing sequence ([-1::-2::-8), ::, NewAxis, ---, ::, ::, NewAxis, [1::4)) is too large " +
                   s"for shape [10, 25, 3, 5].")
    val index2 = Seq[Indexer](0, 0, 0, 0, 0)
    assert(intercept[InvalidIndexerException](Indexer.decode(shape, index2)).getMessage ===
               s"Provided indexing sequence (0, 0, 0, 0, 0) is too large for shape [10, 25, 3, 5].")
    val index3 = Seq[Indexer](---, ---)
    assert(intercept[InvalidIndexerException](Indexer.decode(shape, index3)).getMessage ===
               "Only one ellipsis ('---') is allowed per indexing sequence.")
    val index4 = Seq[Indexer](::, 27, ---)
    assert(intercept[InvalidIndexerException](Indexer.decode(shape, index4)).getMessage ===
               "Indexer '27' is invalid for a dimension with size '25'.")
    val index5 = Seq[Indexer](-11)
    assert(intercept[InvalidIndexerException](Indexer.decode(shape, index5)).getMessage ===
               "Indexer '-11' is invalid for a dimension with size '10'.")
    val index6 = Seq[Indexer](0 :: 12)
    assert(intercept[IndexOutOfBoundsException](Indexer.decode(shape, index6)).getMessage ===
               "Slice end index '12' is outside the bounds for a sequence length of '10'.")
  }

  // TODO: Add tests for "toStridedSlice".

  "'Indexer.toStridedSlice'" must "throw an 'InvalidIndexerException' when an ellipsis is used more than once" in {
    assert(Indexer.toStridedSlice(---).isInstanceOf[tf.Output => tf.Output])
    assert(Indexer.toStridedSlice(::, ---).isInstanceOf[tf.Output => tf.Output])
    assert(Indexer.toStridedSlice(0 :: -1, 0 ::, ---, 3 :: -1 :: 1, -1).isInstanceOf[tf.Output => tf.Output])
    assert(intercept[InvalidIndexerException](Indexer.toStridedSlice(---, ---)).getMessage ===
               "Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
    assert(intercept[InvalidIndexerException](Indexer.toStridedSlice(::, ---, 0, ---)).getMessage ===
               "Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
    assert(intercept[InvalidIndexerException](Indexer.toStridedSlice(0 ::, ---, 3 :: -1 :: 1, ---, -1)).getMessage ===
               "Only one 'Ellipsis' ('---') is allowed per indexing sequence.")
  }
}
