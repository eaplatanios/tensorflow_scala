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

package org.platanios.tensorflow.api.utilities

import org.junit.Test
import org.scalatest.junit.JUnitSuite

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
class ReservoirSuite extends JUnitSuite {
  @Test def testEmptyReservoir(): Unit = {
    val r = Reservoir[String, Int](1)
    assert(r.keys.isEmpty)
  }

  @Test def testReservoirRespectsSize(): Unit = {
    val r = Reservoir[String, Int](42)
    r.add("meaning of life", 12)
    assert(r.buckets("meaning of life").maxSize === 42)
  }

  @Test def testReservoirItemsAndKeys(): Unit = {
    val r = Reservoir[String, Int](42)
    r.add("foo", 4)
    r.add("bar", 9)
    r.add("foo", 19)
    assert(r.keys.toSet === Set("foo", "bar"))
    assert(r.items("foo") === List(4, 19))
    assert(r.items("bar") === List(9))
  }

  @Test def testReservoirExceptions(): Unit = {
    assertThrows[IllegalArgumentException](Reservoir[String, Int](-1))
    assertThrows[NoSuchElementException](Reservoir[String, Int](12).items("missing key"))
  }

  @Test def testReservoirDeterminism(): Unit = {
    val r1 = Reservoir[String, Int](10)
    val r2 = Reservoir[String, Int](10)
    (0 until 100).foreach(i => {
      r1.add("key", i)
      r2.add("key", i)
    })
    assert(r1.items("key") === r2.items("key"))
  }

  @Test def testReservoirBucketDeterminism(): Unit = {
    val r1 = Reservoir[String, Int](10)
    val r2 = Reservoir[String, Int](10)
    (0 until 100).foreach(i => r1.add("key1", i))
    (0 until 100).foreach(i => r1.add("key2", i))
    (0 until 100).foreach(i => {
      r2.add("key1", i)
      r2.add("key2", i)
    })
    assert(r1.items("key1") === r2.items("key1"))
    assert(r1.items("key2") === r2.items("key2"))
  }

  @Test def testReservoirUsesSeed(): Unit = {
    val r1 = Reservoir[String, Int](10, seed = 0L)
    val r2 = Reservoir[String, Int](10, seed = 1L)
    (0 until 100).foreach(i => {
      r1.add("key", i)
      r2.add("key", i)
    })
    assert(r1.items("key") !== r2.items("key"))
  }

  @Test def testReservoirFilterItemsByKey(): Unit = {
    val r = Reservoir[String, Int](100, seed = 0L)
    (0 until 10).foreach(i => {
      r.add("key1", i)
      r.add("key2", i)
    })
    assert(r.items("key1").size === 10)
    assert(r.items("key2").size === 10)
    assert(r.filter(_ <= 7, Some("key2")) === 2)
    assert(r.items("key1").size === 10)
    assert(r.items("key2").size === 8)
    assert(r.filter(_ <= 3, Some("key1")) === 6)
    assert(r.items("key1").size === 4)
    assert(r.items("key2").size === 8)
  }

  @Test def testEmptyReservoirBucket(): Unit = {
    val b = ReservoirBucket[Int](1)
    assert(b.items.isEmpty)
  }

  @Test def testReservoirBucketFillToSize(): Unit = {
    val b = ReservoirBucket[Int](100)
    (0 until 100).foreach(b.add(_))
    assert(b.items === (0 until 100).toList)
    assert(b.numItemsSeen === 100)
  }

  @Test def testReservoirBucketDoesNotOverfill(): Unit = {
    val b = ReservoirBucket[Int](10)
    (0 until 1000).foreach(b.add(_))
    assert(b.items.size === 10)
    assert(b.numItemsSeen === 1000)
  }

  @Test def testReservoirBucketMaintainsOrder(): Unit = {
    val b = ReservoirBucket[Int](100)
    (0 until 10000).foreach(b.add(_))
    val items = b.items
    var previous = -1
    items.foreach(item => {
      assert(item > previous)
      previous = item
    })
  }

  @Test def testReservoirBucketKeepsLastItem(): Unit = {
    val b = ReservoirBucket[Int](5)
    (0 until 100).foreach(i => {
      b.add(i)
      assert(b.items.last === i)
    })
  }

  @Test def testReservoirBucketWithMaxSize1(): Unit = {
    val b = ReservoirBucket[Int](1)
    (0 until 20).foreach(i => {
      b.add(i)
      assert(b.items === List(i))
    })
    assert(b.numItemsSeen === 20)
  }

  @Test def testReservoirBucketWithMaxSize0(): Unit = {
    val b = ReservoirBucket[Int](0)
    (0 until 20).foreach(i => {
      b.add(i)
      assert(b.items === (0 to i).toList)
    })
    assert(b.numItemsSeen === 20)
  }

  @Test def testReservoirBucketMaxSizeRequirement(): Unit = {
    assertThrows[IllegalArgumentException](ReservoirBucket[Int](-1))
  }

  @Test def testReservoirBucketFilterItems(): Unit = {
    val b = ReservoirBucket[Int](100)
    (0 until 10).foreach(b.add(_))
    assert(b.items.size === 10)
    assert(b.numItemsSeen === 10)
    assert(b.filter(_ <= 7) === 2)
    assert(b.items.size === 8)
    assert(b.numItemsSeen === 8)
  }

  @Test def testReservoirBucketRemovesItemsWhenItemsAreReplaced(): Unit = {
    val b = ReservoirBucket[Int](100)
    (0 until 10000).foreach(b.add(_))
    assert(b.numItemsSeen === 10000)
    val numFiltered = b.filter(_ <= 7)
    assert(numFiltered > 92)
    assert(!b.items.exists(_ > 7))
    assert(b.numItemsSeen === 10000 * (1 - numFiltered.toFloat / 100))
  }

  @Test def testReservoirBucketLazyFunctionEvaluationAndAlwaysKeepLast(): Unit = {
    class FakeRandom extends Random {
      override def nextInt(n: Int): Int = 999
    }

    class Incrementer {
      var n: Int = 0

      def incrementAndDouble(x: Int): Int = {
        n += 1
        x * 2
      }
    }

    // We have mocked the random number generator, so that once it is full, the last item will never get durable
    // reservoir inclusion. Since `alwaysKeepLast` is set to `false`, the function should only get invoked 100 times
    // while filling up the reservoir. This laziness property is an essential performance optimization.
    val b1 = ReservoirBucket[Int](100, new FakeRandom(), alwaysKeepLast = false)
    val i1 = new Incrementer()
    (0 until 1000).foreach(b1.add(_, i1.incrementAndDouble))
    assert(i1.n === 100)
    assert(b1.items === (0 until 100).map(_ * 2).toList)

    // Now, we will always keep the last item, meaning that the function should get invoked once for every item we add.
    val b2 = ReservoirBucket[Int](100, new FakeRandom(), alwaysKeepLast = true)
    val i2 = new Incrementer()
    (0 until 1000).foreach(b2.add(_, i2.incrementAndDouble))
    assert(i2.n === 1000)
    assert(b2.items === (0 until 99).map(_ * 2).toList :+ 999 * 2)
  }

  @Test def testReservoirBucketStatisticalDistribution(): Unit = {
    val numTotal = 10000
    val numSamples = 100
    val numBuckets = 10
    val totalPerBucket = numTotal / numBuckets
    assert(numTotal % numBuckets === 0)
    assert(numTotal > numSamples)

    def assertBinomialQuantity(measured: Int): Unit = {
      val p = 1.0 * numBuckets / numSamples
      val mean = p * numSamples
      val variance = p * (1 - p) * numSamples
      val error = measured - mean
      // Given that the buckets were actually binomially distributed, this fails with probability ~2e-9.
      assert(error * error <= 36.0 * variance)
    }

    // Not directly related to a `ReservoirBucket`, but instead we put samples into a specific number of buckets for
    // testing the shape of the distribution.
    val b = ReservoirBucket[Int](numSamples)
    // We add one extra item because we always keep the most recent item, which would skew the distribution; we can just
    // slice it off the end instead.
    (0 until numTotal + 1).foreach(b.add(_))
    val divBins = ArrayBuffer.fill(numBuckets)(0)
    val modBins = ArrayBuffer.fill(numBuckets)(0)
    // Slice off the last item when we iterate.
    b.items.dropRight(1).foreach(item => {
      divBins(item / totalPerBucket) += 1
      modBins(item % numBuckets) += 1
    })
    divBins.foreach(assertBinomialQuantity)
    modBins.foreach(assertBinomialQuantity)
  }
}
