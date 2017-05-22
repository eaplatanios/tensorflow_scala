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

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.ops.{Basic, Op}
import org.platanios.tensorflow.api.tensors.Tensor

import org.scalatest.junit.JUnitSuite
import org.junit.Test

import scala.collection.immutable.TreeMap

/**
  * @author Emmanouil Antonios Platanios
  */
class FetchableSuite extends JUnitSuite {
  def process[F, R](fetchable: F)(implicit ev: Fetchable.Aux[F, R]): (Seq[Op.Output], Seq[Tensor] => R) = {
    Fetchable.process(fetchable)(ev)
  }

  @Test def testFetchable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = Basic.constant(1.0)
      val fetchable2 = Op.OutputIndexedSlices(Basic.constant(2.0), Basic.constant(2.0), Basic.constant(2.0))
      val fetchable3 = Op.SparseOutput(
        Basic.constant(Tensor(Tensor(2L), Tensor(1L))), Basic.constant(Tensor(2L, 1L)), Basic.constant(Tensor(3L)))
      val processed1 = process(fetchable1)
      val processed2 = process(fetchable2)
      val processed3 = process(fetchable3)
      assert(processed1._1.length === 1)
      assert(processed1._1(0).name === "Constant:0")
      assert(processed1._2(Seq.fill(1)(Tensor(0))).isInstanceOf[Tensor])
      assert(processed2._1.length === 3)
      assert(processed2._1(0).name === "Constant_1:0")
      assert(processed2._1(1).name === "Constant_2:0")
      assert(processed2._1(2).name === "Constant_3:0")
      assert(processed2._2(Seq.fill(3)(Tensor(0))).isInstanceOf[(Tensor, Tensor, Tensor)])
      assert(processed3._1.length === 3)
      assert(processed3._1(0).name === "Constant_4:0")
      assert(processed3._1(1).name === "Constant_5:0")
      assert(processed3._1(2).name === "Constant_6:0")
      assert(processed3._2(Seq.fill(3)(Tensor(0))).isInstanceOf[(Tensor, Tensor, Tensor)])
    }
  }

  @Test def testFetchableSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedSeq = process(Seq(Basic.constant(1.0), Basic.constant(2.0), Basic.constant(3.0)))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor(0)))
      assert(results.length === 3)
      assert(results(0).isInstanceOf[Tensor])
      assert(results(1).isInstanceOf[Tensor])
      assert(results(2).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableList(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedList = process(List(Basic.constant(1.0), Basic.constant(2.0), Basic.constant(3.0)))
      assert(processedList._1.length === 3)
      assert(processedList._1(0).name === "Constant:0")
      assert(processedList._1(1).name === "Constant_1:0")
      assert(processedList._1(2).name === "Constant_2:0")
      val results = processedList._2(Seq.fill(3)(Tensor(0)))
      assert(results.length === 3)
      assert(results(0).isInstanceOf[Tensor])
      assert(results(1).isInstanceOf[Tensor])
      assert(results(2).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedList = process(Array(Basic.constant(1.0), Basic.constant(2.0), Basic.constant(3.0)))
      assert(processedList._1.length === 3)
      assert(processedList._1(0).name === "Constant:0")
      assert(processedList._1(1).name === "Constant_1:0")
      assert(processedList._1(2).name === "Constant_2:0")
      val results = processedList._2(Seq.fill(3)(Tensor(0)))
      assert(results.length === 3)
      assert(results(0).isInstanceOf[Tensor])
      assert(results(1).isInstanceOf[Tensor])
      assert(results(2).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedMap = process(
        TreeMap("1" -> Basic.constant(1.0), "2" -> Basic.constant(2.0), "3" -> Basic.constant(3.0)))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor(0)))
      assert(results.size === 3)
      assert(results("1").isInstanceOf[Tensor])
      assert(results("2").isInstanceOf[Tensor])
      assert(results("3").isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableSeqWithDuplicates(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = Basic.constant(1.0)
      val fetchable2 = Basic.constant(2.0)
      val fetchable3 = Basic.constant(3.0)
      val processedSeq = process(Seq(fetchable1, fetchable1, fetchable2, fetchable2, fetchable3))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor(0)))
      assert(results.length === 5)
      assert(results(0).isInstanceOf[Tensor])
      assert(results(1).isInstanceOf[Tensor])
      assert(results(2).isInstanceOf[Tensor])
      assert(results(3).isInstanceOf[Tensor])
      assert(results(4).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableMapWithDuplicates(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = Basic.constant(1.0)
      val fetchable2 = Basic.constant(2.0)
      val fetchable3 = Basic.constant(3.0)
      val processedMap = process(
        TreeMap("1_1" -> fetchable1, "1_2" -> fetchable1, "2_1" -> fetchable2, "2_2" -> fetchable2, "3" -> fetchable3))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor(0)))
      assert(results.size === 5)
      assert(results("1_1").isInstanceOf[Tensor])
      assert(results("1_2").isInstanceOf[Tensor])
      assert(results("2_1").isInstanceOf[Tensor])
      assert(results("2_2").isInstanceOf[Tensor])
      assert(results("3").isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedSeq = process(Seq(Seq(Basic.constant(1.0)), Seq(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor(0)))
      assert(results.length === 2)
      assert(results(0).length === 1)
      assert(results(1).length === 2)
      assert(results(0)(0).isInstanceOf[Tensor])
      assert(results(1)(0).isInstanceOf[Tensor])
      assert(results(1)(1).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedSeqArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedSeq = process(Seq(Array(Basic.constant(1.0)), Array(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor(0)))
      assert(results.length === 2)
      assert(results(0).length === 1)
      assert(results(1).length === 2)
      assert(results(0)(0).isInstanceOf[Tensor])
      assert(results(1)(0).isInstanceOf[Tensor])
      assert(results(1)(1).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedMapArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedMap = process(
        TreeMap("1" -> TreeMap("1" -> Basic.constant(1.0)),
                "2" -> TreeMap("2" -> Basic.constant(2.0), "3" -> Basic.constant(3.0))))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor(0)))
      assert(results.size === 2)
      assert(results("1").size === 1)
      assert(results("2").size === 2)
      assert(results("1")("1").isInstanceOf[Tensor])
      assert(results("2")("2").isInstanceOf[Tensor])
      assert(results("2")("3").isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedMapSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedMap = process(
        TreeMap("1" -> Seq(Basic.constant(1.0)), "2" -> Seq(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor(0)))
      assert(results.size === 2)
      assert(results("1").length === 1)
      assert(results("2").length === 2)
      assert(results("1")(0).isInstanceOf[Tensor])
      assert(results("2")(0).isInstanceOf[Tensor])
      assert(results("2")(1).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableTuple(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = process((Basic.constant(1.0), Basic.constant(2.0)))
      assert(processedTuple._1.length === 2)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      val results = processedTuple._2(Seq.fill(2)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Tensor)])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2.isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedTuple(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = process((Basic.constant(1.0), (Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedTuple._1.length === 3)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      assert(processedTuple._1(2).name === "Constant_2:0")
      val results = processedTuple._2(Seq.fill(3)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, (Tensor, Tensor))])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2._1.isInstanceOf[Tensor])
      assert(results._2._2.isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedTupleArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = process((Basic.constant(1.0), Array(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedTuple._1.length === 3)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      assert(processedTuple._1(2).name === "Constant_2:0")
      val results = processedTuple._2(Seq.fill(3)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Array[Tensor])])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2(0).isInstanceOf[Tensor])
      assert(results._2(1).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableNestedTupleMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = process((Basic.constant(1.0), Map("2" -> Basic.constant(2.0), "3" -> Basic.constant(3.0))))
      assert(processedTuple._1.length === 3)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      assert(processedTuple._1(2).name === "Constant_2:0")
      val results = processedTuple._2(Seq.fill(3)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Map[String, Tensor])])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2("2").isInstanceOf[Tensor])
      assert(results._2("3").isInstanceOf[Tensor])
    }
  }
}
