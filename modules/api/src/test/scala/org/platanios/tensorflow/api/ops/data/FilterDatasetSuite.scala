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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.{Math, Op}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.using

import org.junit.Test
import org.scalatest.junit.JUnitSuite

/**
  * @author Emmanouil Antonios Platanios
  */
class FilterDatasetSuite extends JUnitSuite {
  @Test def testFilterRange(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val dataset = Data.datasetFromRange(0, 100).filter(x => {
        Math.notEqual(Math.mod(x, 3L), 2L)
      })
      val iterator = dataset.createInitializableIterator()
      val initOp = iterator.initializer
      val nextOutput = iterator.next()
      assert(nextOutput.shape === Shape.scalar())
      val session = Session()
      session.run(targets = initOp)
      assert(session.run(fetches = nextOutput) == (0L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (3L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (4L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (6L: Tensor[Long]))
    }
  }
}
