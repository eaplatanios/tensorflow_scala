/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.scalatest.matchers.should.Matchers
import org.scalatestplus.junit.JUnitSuite
import org.junit.Test

/**
 * @author Emmanouil Antonios Platanios
 */
class NNSpec extends JUnitSuite with Matchers {
  @Test def testLogSoftmax(): Unit = {
    val tensor = Tensor(Tensor(Tensor(2, 3), Tensor(0, 0), Tensor(5, 7)),
      Tensor(Tensor(1, 23), Tensor(4, -5), Tensor(7, 9)),
      Tensor(Tensor(56, 1), Tensor(-2, -4), Tensor(-7, -9)))
    val constant = tf.constant(tensor).toFloat
    val logSoftmaxLastAxis = tf.logSoftmax(constant, axis = -1)
    val logSoftmaxPenultimateAxis = tf.logSoftmax(constant, axis = 1)
    val session = Session()
    assertApproximatelyEqual(
      session.run(fetches = logSoftmaxLastAxis).toArray,
      Array(
        -1.3132616f, -0.31326163f, -0.6931472f, -0.6931472f, -2.126928f, -0.12692805f,
        -22.0f, 0.0f, -1.23374e-4f, -9.000123f, -2.126928f, -0.12692805f, 0.0f, -55.0f,
        -0.12692805f, -2.126928f, -0.12692805f, -2.126928f,
      ),
    )
    assertApproximatelyEqual(
      session.run(fetches = logSoftmaxPenultimateAxis).toArray,
      Array(
        -3.0549853f, -4.019045f, -5.054985f, -7.019045f, -0.054985214f, -0.019044992f,
        -6.0509458f, -8.344647e-7f, -3.0509458f, -28.0f, -0.05094571f, -14.000001f, 0.0f,
        -0.0067604627f, -58.0f, -5.0067606f, -63.0f, -10.006761f,
      ),
    )
  }

  def assertApproximatelyEqual(x: Array[Float], y: Array[Float]): Unit = {
    x.zip(y).foreach { case (xElement, yElement) =>
      assert(xElement === yElement +- 1e-6f)
    }
  }
}
