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

package org.platanios.tensorflow.api.ops.data

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.client.Session
import org.platanios.tensorflow.api.core.exception.OutOfRangeException
import org.platanios.tensorflow.api.core.types.INT32
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToTensor}
import org.platanios.tensorflow.api.ops.{Op, Output}
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.using

import org.junit.Test
import org.scalatestplus.junit.JUnitSuite

import scala.collection.compat.immutable.LazyList

/**
  * @author Emmanouil Antonios Platanios
  */
class DatasetSuite extends JUnitSuite {
  // Implicit helper for Scala 2.11
  val evOutputStructureIntIntDouble: OutputStructure[(Output[Int], Output[Int], Output[Double])] = OutputStructure[(Output[Int], Output[Int], Output[Double])]
  val evOutputToTensorIntIntDouble: OutputToTensor.Aux[(Output[Int], Output[Int], Output[Double]), (Tensor[Int], Tensor[Int], Tensor[Double])] = OutputToTensor[(Output[Int], Output[Int], Output[Double])]

  @Test def testTensorDataset(): Unit = using(Graph()) { graph =>
    Op.createWith(graph, device = "/cpu:0") {
      val components = Tensor(1, 2, 3)
      val iterator = Data.datasetFromTensors(components).createInitializableIterator()
      val initOp = iterator.initializer
      val nextOutput = iterator.next()
      assert(nextOutput.shape == components.shape)
      val session = Session()
      session.run(targets = initOp)
      assert(components == session.run(fetches = nextOutput))
      assertThrows[OutOfRangeException](session.run(fetches = nextOutput))
    }
  }

  @Test def testTensorTupleDataset(): Unit = using(Graph()) { graph =>
    // Implicit helper for Scala 2.11
    implicit val evOutputStructureIntIntDouble: OutputStructure[(Output[Int], Output[Int], Output[Double])] = this.evOutputStructureIntIntDouble
    implicit val evOutputToTensorIntIntDouble: OutputToTensor.Aux[(Output[Int], Output[Int], Output[Double]), (Tensor[Int], Tensor[Int], Tensor[Double])] = this.evOutputToTensorIntIntDouble

    Op.createWith(graph) {
      val components = (Tensor(1), Tensor(1, 2, 3), Tensor(37.0))
      val dataset = Data.datasetFromTensors(components)
      val iterator = dataset.createInitializableIterator()
      val initOp = iterator.initializer
      val nextOp = iterator.next()
      assert(components._1.shape == nextOp._1.shape)
      assert(components._2.shape == nextOp._2.shape)
      assert(components._3.shape == nextOp._3.shape)
      val session = Session()
      session.run(targets = initOp)
      val results = session.run(fetches = nextOp)
      assert(components._1 == results._1)
      assert(components._2 == results._2)
      assert(components._3 == results._3)
      assertThrows[OutOfRangeException](session.run(fetches = nextOp))
    }
  }

  @Test def testRangeDataset(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val dataset = Data.datasetFromRange(0, 4, 1)
      val iterator = dataset.createInitializableIterator()
      val initOp = iterator.initializer
      val nextOutput = iterator.next()
      assert(nextOutput.shape == Shape.scalar())
      val session = Session()
      session.run(targets = initOp)
      assert(session.run(fetches = nextOutput) == (0L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (2L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (3L: Tensor[Long]))
      assertThrows[OutOfRangeException](session.run(fetches = nextOutput))
    }
  }

  @Test def testMapDataset(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val dataset = Data.datasetFromRange(0, 4, 1).map(v => 2 * v)
      val iterator = dataset.createInitializableIterator()
      val initOp = iterator.initializer
      val nextOutput = iterator.next()
      assert(nextOutput.shape == Shape.scalar())
      val session = Session()
      session.run(targets = initOp)
      assert(session.run(fetches = nextOutput) == (0L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (2L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (4L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (6L: Tensor[Long]))
      assertThrows[OutOfRangeException](session.run(fetches = nextOutput))
    }
  }

  @Test def testFlatMapDataset(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val dataset = Data.datasetFromRange(0, 4, 1).flatMap(v => {
        Data.datasetFromOutputSlices(Basic.stack[Long](Seq(v, 1L)))
      })
      val iterator = dataset.createInitializableIterator()
      val initOp = iterator.initializer
      val nextOutput = iterator.next()
      assert(nextOutput.shape == Shape.scalar())
      val session = Session()
      session.run(targets = initOp)
      assert(session.run(fetches = nextOutput) == (0L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (2L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (3L: Tensor[Long]))
      assert(session.run(fetches = nextOutput) == (1L: Tensor[Long]))
      assertThrows[OutOfRangeException](session.run(fetches = nextOutput))
    }
  }

  @Test def testDatasetFromGenerator(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val dataset = Data.datasetFromGenerator(() => {
        LazyList(0, 1, 2, 3).map(Tensor[Int](_))
      }, INT32, Shape(1))
      val iterator = dataset.createInitializableIterator()
      val initOp = iterator.initializer
      val nextOutput = iterator.next()
      assert(nextOutput.shape == Shape(1))
      val session = Session()
      session.run(targets = initOp)
      assert(session.run(fetches = nextOutput) == Tensor[Int](0))
      assert(session.run(fetches = nextOutput) == Tensor[Int](1))
      assert(session.run(fetches = nextOutput) == Tensor[Int](2))
      assert(session.run(fetches = nextOutput) == Tensor[Int](3))
      assertThrows[OutOfRangeException](session.run(fetches = nextOutput))
    }
  }
}
