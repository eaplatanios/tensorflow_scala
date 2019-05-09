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

package org.platanios.tensorflow.jni

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorFlowSpec extends FlatSpec {
  "The TensorFlow library version" must "have non-zero length" in {
    assert(TensorFlow.version.length > 0)
  }

  "The TensorFlow library data type sizes" must "be correct" in {
    assert(TensorFlow.dataTypeSize(-1) === 0)
    assert(TensorFlow.dataTypeSize(0) === 0)
    assert(TensorFlow.dataTypeSize(1) === 4)
    assert(TensorFlow.dataTypeSize(2) === 8)
    assert(TensorFlow.dataTypeSize(3) === 4)
    assert(TensorFlow.dataTypeSize(4) === 1)
    assert(TensorFlow.dataTypeSize(5) === 2)
    assert(TensorFlow.dataTypeSize(6) === 1)
    assert(TensorFlow.dataTypeSize(7) === 0)
    assert(TensorFlow.dataTypeSize(8) === 8)
    assert(TensorFlow.dataTypeSize(9) === 8)
    assert(TensorFlow.dataTypeSize(10) === 1)
    assert(TensorFlow.dataTypeSize(11) === 1)
    assert(TensorFlow.dataTypeSize(12) === 1)
    assert(TensorFlow.dataTypeSize(13) === 4)
    assert(TensorFlow.dataTypeSize(14) === 2)
    assert(TensorFlow.dataTypeSize(15) === 2)
    assert(TensorFlow.dataTypeSize(16) === 2)
    assert(TensorFlow.dataTypeSize(17) === 2)
    assert(TensorFlow.dataTypeSize(18) === 16)
    assert(TensorFlow.dataTypeSize(19) === 2)
    assert(TensorFlow.dataTypeSize(20) === 0)
    assert(TensorFlow.dataTypeSize(101) === 0)
    assert(TensorFlow.dataTypeSize(102) === 0)
    assert(TensorFlow.dataTypeSize(103) === 0)
    assert(TensorFlow.dataTypeSize(104) === 0)
    assert(TensorFlow.dataTypeSize(105) === 0)
    assert(TensorFlow.dataTypeSize(106) === 0)
    assert(TensorFlow.dataTypeSize(107) === 0)
    assert(TensorFlow.dataTypeSize(108) === 0)
    assert(TensorFlow.dataTypeSize(109) === 0)
    assert(TensorFlow.dataTypeSize(110) === 0)
    assert(TensorFlow.dataTypeSize(111) === 0)
    assert(TensorFlow.dataTypeSize(112) === 0)
    assert(TensorFlow.dataTypeSize(113) === 0)
    assert(TensorFlow.dataTypeSize(114) === 0)
    assert(TensorFlow.dataTypeSize(115) === 0)
    assert(TensorFlow.dataTypeSize(116) === 0)
    assert(TensorFlow.dataTypeSize(117) === 0)
    assert(TensorFlow.dataTypeSize(118) === 0)
    assert(TensorFlow.dataTypeSize(119) === 0)
    assert(TensorFlow.dataTypeSize(120) === 0)
    assert(TensorFlow.dataTypeSize(121) === 0)
    assert(TensorFlow.dataTypeSize(167) === 0)
  }
}
