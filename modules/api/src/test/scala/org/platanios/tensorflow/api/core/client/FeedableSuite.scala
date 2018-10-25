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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.ops.{Basic, Op, OutputIndexedSlices}
import org.platanios.tensorflow.api.tensors.TensorIndexedSlices

import org.scalatest.junit.JUnitSuite
import org.junit.Test

/**
  * @author Emmanouil Antonios Platanios
  */
class FeedableSuite extends JUnitSuite {
  def feedMapIdentity(feedMap: FeedMap): FeedMap = feedMap

  @Test def testFeedMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val tensor0 = Tensor(0.0f)
      val tensor1 = Tensor(1)
      val tensor2 = Tensor(2.0f)
      val tensor3 = Tensor(3)
      val feedable1 = Basic.placeholder[Float]()
      val feedable2 = OutputIndexedSlices(
        Basic.placeholder[Int](), Basic.placeholder[Float](), Basic.placeholder[Int]())
      val feedable1FeedMap = feedMapIdentity(Map(feedable1 -> tensor0))
      val feedable2FeedMap = feedMapIdentity(Map(feedable2 -> TensorIndexedSlices(tensor1, tensor2, tensor3)))
      assert(feedable1FeedMap.values === Map(feedable1 -> tensor0))
      assert(feedable2FeedMap.values ===
                 Map(feedable2.indices -> tensor1, feedable2.values -> tensor2, feedable2.denseShape -> tensor3))
    }
  }

  @Test def testHeterogeneousFeedMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val tensor0 = Tensor(0.0f)
      val tensor1 = Tensor(1)
      val tensor2 = Tensor(2.0f)
      val tensor3 = Tensor(3)
      val feedable1 = Basic.placeholder[Float]()
      val feedable2 = OutputIndexedSlices(
        Basic.placeholder[Int](), Basic.placeholder[Float](), Basic.placeholder[Int]())
      val feedable1FeedMap: FeedMap = feedMapIdentity(Map(feedable1 -> tensor0))
      val feedable2FeedMap: FeedMap = feedMapIdentity(Map(feedable2 -> TensorIndexedSlices(tensor1, tensor2, tensor3)))
      val feedMap = feedMapIdentity(feedable1FeedMap ++ feedable2FeedMap)
      assert(feedMap.values === Map(
        feedable1 -> tensor0,
        feedable2.indices -> tensor1, feedable2.values -> tensor2, feedable2.denseShape -> tensor3))
    }
  }
}
