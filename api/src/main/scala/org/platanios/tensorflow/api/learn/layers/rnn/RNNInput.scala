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

package org.platanios.tensorflow.api.learn.layers.rnn

import org.platanios.tensorflow.api.ops.Output

import scala.collection.TraversableLike
import scala.collection.generic.CanBuildFrom

trait RNNInput[T] {
  def outputs(input: T): Seq[Output]
}

object RNNInput {
  def apply[T: RNNInput]: RNNInput[T] = implicitly[RNNInput[T]]

  implicit def outputExecutable: RNNInput[Output] = new RNNInput[Output] {
    override def outputs(input: Output): Seq[Output] = Seq(input)
  }

  implicit def traversableExecutable[T: RNNInput, CC[A] <: TraversableLike[A, CC[A]]](implicit
    cbf: CanBuildFrom[CC[T], Output, Seq[Output]]
  ): RNNInput[CC[T]] = {
    new RNNInput[CC[T]] {
      override def outputs(input: CC[T]): Seq[Output] = input.flatMap(e => RNNInput[T].outputs(e))(cbf)
    }
  }
}
