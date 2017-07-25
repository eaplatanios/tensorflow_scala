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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tensors.RealNumericTensor
import org.scalactic.{Equality, Equivalence, TolerantNumerics, TypeCheckedTripleEquals}
import org.scalatest._

/**
  * @author Sören Brunk
  */
class RandomSpec extends FlatSpec with Matchers with TypeCheckedTripleEquals{

  val epsilon = 1e-4f
  val randomSeed = 42

  implicit val tolerantNumericTensorEq = new Equivalence[RealNumericTensor] {
    override def areEquivalent(a: RealNumericTensor, b: RealNumericTensor): Boolean = {
      a.shape == b.shape &&
      a.dataType == b.dataType &&
      //a.entriesIterator.map{case v: Float => v}.zip(b.entriesIterator.map{case v: Float => v}).forall(p => p._1 === p._2) &&
      a == b +- epsilon // TODO cleaner tolerant equals
  }}

  "The random normal distribution op" must "produce correct and predictable FLOAT32 values" in {
    val session = tf.Session
    val rnd = tf.randomNormal()(tf.Shape(2,2), seed = Some(randomSeed))
    val result = rnd.evaluate()
    val expectedResult = tf.Tensor(tf.FLOAT32,
      tf.Tensor(-0.28077507, -0.1377521),
      tf.Tensor(-0.67632961, 0.02458041))
    assert(result.asRealNumeric === expectedResult.asRealNumeric)
  }

  "The random uniform distribution op" must "produce correct and predictable FLOAT32 values" in {
    val session = tf.Session
    val rnd = tf.randomUniform()(tf.Shape(2,2), seed = Some(randomSeed))
    val result = rnd.evaluate()
    val expectedResult = tf.Tensor(tf.FLOAT32,
      tf.Tensor(0.95227146, 0.67740774),
      tf.Tensor(0.79531825, 0.75578177))
    assert(result.asRealNumeric === expectedResult.asRealNumeric)
  }
}
