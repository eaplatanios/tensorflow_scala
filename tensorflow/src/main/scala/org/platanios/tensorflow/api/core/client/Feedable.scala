// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.tensors.Tensor

/**
  * @author Emmanouil Antonios Platanios
  */
trait Feedable[T] {
  def toFeedMap(value: T): Map[Op.Output, Tensor]
}

case class FeedMap(values: Map[Op.Output, Tensor] = Map.empty) {
  def feed[T](feedable: Feedable[T], value: T): FeedMap = {
    FeedMap(values ++ feedable.toFeedMap(value))
  }
}

object FeedMap {
  val empty = FeedMap()

  trait Implicits {
    implicit def feedMap(feeds: Map[Op.Output, Tensor]): FeedMap = FeedMap(feeds)
  }

  object Implicits extends Implicits
}
