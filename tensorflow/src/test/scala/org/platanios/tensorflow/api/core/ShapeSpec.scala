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
import org.platanios.tensorflow.api.tf.InvalidShapeException

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class ShapeSpec extends FlatSpec with Matchers {
  "Shape construction" must "work for completely unknown shapes" in {
    val shape = Shape.unknown()
    assert(shape.rank === -1)
    assert(shape.asArray === null)
  }

  it must "work for unknown shapes with known rank" in {
    val shape = Shape.unknown(rank = 145)
    assert(shape.rank === 145)
    assert(shape.asArray === Array.fill[Int](145)(-1))
  }

  it must "work for scalar shapes" in {
    val shape = Shape.scalar()
    assert(shape.rank === 0)
    assert(shape.asArray === Array.empty[Int])
  }

  it must "work for vector shapes" in {
    val shape = Shape.vector(length = 67)
    assert(shape.rank === 1)
    assert(shape.asArray === Array[Int](67))
  }

  it must "work for matrix shapes" in {
    val shape = Shape.matrix(numRows = 32, numColumns = 784)
    assert(shape.rank === 2)
    assert(shape.asArray === Array[Int](32, 784))
  }

  it must "work for arbitrary shapes" in {
    val shape = Shape(45, 2356, 54, 2)
    assert(shape.rank === 4)
    assert(shape.asArray === Array[Int](45, 2356, 54, 2))
  }

  it must "work when creating shapes from sequences" in {
    val shape = Shape.fromSeq(Array[Int](45, 2356, 54, 2))
    assert(shape.rank === 4)
    assert(shape.asArray === Array[Int](45, 2356, 54, 2))
  }

  "'Shape.isFullyDefined'" must "always work correctly" in {
    assert(Shape().isFullyDefined === true)
    assert(Shape(0).isFullyDefined === true)
    assert(Shape(1).isFullyDefined === true)
    assert(Shape(34, 6, 356, 89).isFullyDefined === true)
    assert(Shape(-1, 4, 3).isFullyDefined === false)
    assert(Shape(34, 4, -1).isFullyDefined === false)
    assert(Shape.unknown().isFullyDefined === false)
    assert(Shape.unknown(rank = 5).isFullyDefined === false)
  }

  "'Shape.rank'" must "always work correctly" in {
    assert(Shape().rank === 0)
    assert(Shape(0).rank === 1)
    assert(Shape(1).rank === 1)
    assert(Shape(34, 6, 356, 89).rank === 4)
    assert(Shape(-1, 4, 3).rank === 3)
    assert(Shape(34, 4, -1).rank === 3)
    assert(Shape.unknown().rank === -1)
    assert(Shape.unknown(rank = 5).rank === 5)
  }

  "'Shape.numElements'" must "always work correctly" in {
    assert(Shape().numElements === Some(1))
    assert(Shape(0).numElements === Some(0))
    assert(Shape(1).numElements === Some(1))
    assert(Shape(34, 6, 356, 89).numElements === Some(6463536))
    assert(Shape(-1, 4, 3).numElements === None)
    assert(Shape(34, 4, -1).numElements === None)
    assert(Shape.unknown().numElements === None)
    assert(Shape.unknown(rank = 5).numElements === None)
  }

  "'Shape.isCompatibleWith'" must "always work correctly" in {
    val shape1 = Shape(0)
    val shape2 = Shape.unknown()
    val shape3 = Shape.unknown(rank = 4)
    val shape4 = Shape.scalar()
    val shape5 = Shape(1)
    val shape6 = Shape(34, 6, 356, 89)
    val shape7 = Shape(-1, 4, 3)
    val shape8 = Shape(34, 4, 3)
    val shape9 = Shape(-1, 4, -1)
    assert(shape2.isCompatibleWith(shape1) === true)
    assert(shape2.isCompatibleWith(shape3) === true)
    assert(shape2.isCompatibleWith(shape4) === true)
    assert(shape2.isCompatibleWith(shape6) === true)
    assert(shape3.isCompatibleWith(shape6) === true)
    assert(shape3.isCompatibleWith(shape1) === false)
    assert(shape3.isCompatibleWith(shape9) === false)
    assert(shape1.isCompatibleWith(shape5) === false)
    assert(shape6.isCompatibleWith(shape3) === true)
    assert(shape7.isCompatibleWith(shape2) === true)
    assert(shape7.isCompatibleWith(shape8) === true)
    assert(shape7.isCompatibleWith(shape9) === true)
    assert(shape7.isCompatibleWith(shape3) === false)
    assert(shape7.isCompatibleWith(shape5) === false)
  }

  it must "be reflexive and symmetric" in {
    val shapes = Array(
      Shape(0), Shape.unknown(), Shape.unknown(rank = 4), Shape.scalar(), Shape(1), Shape(34, 6, 356, 89),
      Shape(-1, 4, 3), Shape(34, 4, 3), Shape(-1, 4, -1))
    val pairs = for (s1 <- shapes; s2 <- shapes) yield (s1, s2)
    pairs.foreach(pair => assert(pair._1.isCompatibleWith(pair._2) === pair._2.isCompatibleWith(pair._1)))
  }

  "'Shape.mergeWith'" must "always work correctly" in {
    val shape1 = Shape(0)
    val shape2 = Shape(1)
    val shape3 = Shape(-1)
    val shape4 = Shape.unknown()
    val shape5 = Shape.unknown(rank = 4)
    val shape6 = Shape(34, 6, 356, 89)
    val shape7 = Shape(-1, 4, 3)
    val shape8 = Shape(34, 4, 3)
    val shape9 = Shape(-1, 4, -1)
    assert(shape1.mergeWith(shape3) === shape1)
    assert(shape2.mergeWith(shape3) === shape2)
    assert(shape1.mergeWith(shape4) === shape1)
    assert(shape2.mergeWith(shape4) === shape2)
    assert(shape5.mergeWith(shape4) === shape5)
    assert(shape6.mergeWith(shape4) === shape6)
    assert(shape9.mergeWith(shape4) === shape9)
    assert(shape5.mergeWith(shape6) === shape6)
    assert(shape7.mergeWith(shape9) === shape7)
    assert(shape7.mergeWith(shape8) === shape8)
  }

  it must "be reflexive and symmetric" in {
    val shape1 = Shape(0)
    val shape2 = Shape(1)
    val shape3 = Shape(-1)
    val shape4 = Shape.unknown()
    val shape5 = Shape.unknown(rank = 4)
    val shape6 = Shape(34, 6, 356, 89)
    val shape7 = Shape(-1, 4, 3)
    val shape8 = Shape(34, 4, 3)
    val shape9 = Shape(-1, 4, -1)
    val pairs = Array(
      (shape1, shape3), (shape2, shape3), (shape4, shape1), (shape4, shape2), (shape4, shape5), (shape4, shape6),
      (shape4, shape9), (shape5, shape6), (shape7, shape8), (shape7, shape9), (shape8, shape9))
    pairs.foreach(pair => assert(pair._1.mergeWith(pair._2) === pair._2.mergeWith(pair._1)))
  }

  "'Shape.concatenateWith'" must "always work correctly" in {
    val shape1 = Shape(0)
    val shape2 = Shape(1)
    val shape3 = Shape(-1)
    val shape4 = Shape.unknown(rank = 4)
    val shape5 = Shape(34, 6, 356, 89)
    val shape6 = Shape(-1, 4, 3)
    val shape7 = Shape(34, 4, 3)
    val shape8 = Shape(-1, 4, -1)
    assert(shape1.concatenateWith(shape2) === Shape(0, 1))
    assert(shape2.concatenateWith(shape3) === Shape(1, -1))
    assert(shape4.concatenateWith(shape6) === Shape(-1, -1, -1, -1, -1, 4, 3))
    assert(shape2.concatenateWith(shape4) === Shape(1, -1, -1, -1, -1))
    assert(shape5.concatenateWith(shape7) === Shape(34, 6, 356, 89, 34, 4, 3))
    assert(shape8.concatenateWith(shape7) === Shape(-1, 4, -1, 34, 4, 3))
  }

  "'Shape.withRank'" must "always work correctly" in {
    assert(Shape.unknown().withRank(5) === Shape(-1, -1, -1, -1, -1))
  }

  "All shape methods" must "throw exceptions when appropriate" in {
    assert(intercept[InvalidShapeException](Shape.unknown(4).mergeWith(Shape(3, 4))).getMessage ===
               "Shape '[?, ?, ?, ?]' must have the same rank as shape '[3, 4]'.")
    assert(intercept[InvalidShapeException](Shape(4, -1).mergeWith(Shape(3, 4))).getMessage ===
               "Shape '[4, ?]' must be compatible with shape '[3, 4]'.")
    assert(intercept[InvalidShapeException](Shape.unknown(4).withRank(5)).getMessage ===
               "Shape '[?, ?, ?, ?]' must have the same rank as shape '[?, ?, ?, ?, ?]'.")
    assert(intercept[InvalidShapeException](Shape.unknown(4).assertSameRank(Shape(3, 4))).getMessage ===
               "Shape '[?, ?, ?, ?]' must have the same rank as shape '[3, 4]'.")
    assert(intercept[InvalidShapeException](Shape(4, -1).assertIsCompatibleWith(Shape(3, 4))).getMessage ===
               "Shape '[4, ?]' must be compatible with shape '[3, 4]'.")
    assert(intercept[InvalidShapeException](Shape(2, -1).assertRankAtLeast(3)).getMessage ===
               "Shape '[2, ?]' must have rank at least 3.")
    assert(intercept[InvalidShapeException](Shape(2, -1).assertRankAtMost(1)).getMessage ===
               "Shape '[2, ?]' must have rank at most 1.")
  }

  "Shape slicing" must "always work correctly" in {
    val shape = Shape(-1, 4, -1, 34, 2, 98, -1, 3)
    assert(shape(3) === 34)
    assert(shape(4) === 2)
    assert(shape(1 :: 5) === Shape(4, -1, 34, 2))
    assert(shape(6 :: -2 :: 2) === Shape(-1, 2))
  }
}
