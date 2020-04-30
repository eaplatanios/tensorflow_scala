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

package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Op, OutputIndexedSlices, SparseOutput}
import org.platanios.tensorflow.api.tensors.{SparseTensor, Tensor, TensorIndexedSlices}

import org.scalatestplus.junit.JUnitSuite
import org.junit.Test

/**
  * @author Emmanouil Antonios Platanios
  */
class FetchableSuite extends JUnitSuite {
  @Test def testFetchable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = Basic.constant[Double](1.0)
      val fetchable2 = OutputIndexedSlices(Basic.constant(2), Basic.constant(2L), Basic.constant(2))
      val fetchable3 = SparseOutput(
        Basic.constant(Tensor(Tensor(2L), Tensor(1L))), Basic.constant(Tensor(2L, 1L)), Basic.constant(Tensor(3L)))
      val processed1 = Session.processFetches(fetchable1)
      val processed2 = Session.processFetches(fetchable2)
      val processed3 = Session.processFetches(fetchable3)
      assert(processed1._1.length === 1)
      assert(processed1._1(0).name === "Constant:0")
      assert(processed1._2(Seq(Tensor[Double](0))).isInstanceOf[Tensor[_]])
      assert(processed2._1.length === 3)
      assert(processed2._1(0).name === "Constant_1:0")
      assert(processed2._1(1).name === "Constant_2:0")
      assert(processed2._1(2).name === "Constant_3:0")
      assert(processed2._2(Seq.fill(3)(Tensor[Long](0L))).isInstanceOf[TensorIndexedSlices[_]])
      assert(processed3._1.length === 3)
      assert(processed3._1(0).name === "Constant_4:0")
      assert(processed3._1(1).name === "Constant_5:0")
      assert(processed3._1(2).name === "Constant_6:0")
      assert(processed3._2(Seq(
        Tensor[Long](
          Tensor[Long](2L),
          Tensor[Long](1L)),
        Tensor[Long](2L, 1L),
        Tensor[Long](3L)
      )).isInstanceOf[SparseTensor[_]])
    }
  }

  @Test def testFetchableSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedSeq = Session.processFetches(Seq(Basic.constant(1.0), Basic.constant(2.0), Basic.constant(3.0)))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.length === 3)
      assert(results(0).isInstanceOf[Tensor[_]])
      assert(results(1).isInstanceOf[Tensor[_]])
      assert(results(2).isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedArray = Session.processFetches(Seq(Basic.constant(1.0), Basic.constant(2.0), Basic.constant(3.0)))
      assert(processedArray._1.length === 3)
      assert(processedArray._1(0).name === "Constant:0")
      assert(processedArray._1(1).name === "Constant_1:0")
      assert(processedArray._1(2).name === "Constant_2:0")
      val results = processedArray._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.length === 3)
      assert(results(0).isInstanceOf[Tensor[_]])
      assert(results(1).isInstanceOf[Tensor[_]])
      assert(results(2).isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedMap = Session.processFetches(
        Map("1" -> Basic.constant(1.0), "2" -> Basic.constant(2.0), "3" -> Basic.constant(3.0)))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.size === 3)
      assert(results("1").isInstanceOf[Tensor[_]])
      assert(results("2").isInstanceOf[Tensor[_]])
      assert(results("3").isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableSeqWithDuplicates(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = Basic.constant(1.0)
      val fetchable2 = Basic.constant(2.0)
      val fetchable3 = Basic.constant(3.0)
      val processedSeq = Session.processFetches(Seq(fetchable1, fetchable1, fetchable2, fetchable2, fetchable3))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.length === 5)
      assert(results(0).isInstanceOf[Tensor[_]])
      assert(results(1).isInstanceOf[Tensor[_]])
      assert(results(2).isInstanceOf[Tensor[_]])
      assert(results(3).isInstanceOf[Tensor[_]])
      assert(results(4).isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableMapWithDuplicates(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = Basic.constant(1.0)
      val fetchable2 = Basic.constant(2.0)
      val fetchable3 = Basic.constant(3.0)
      val processedMap = Session.processFetches(
        Map("1_1" -> fetchable1, "1_2" -> fetchable1, "2_1" -> fetchable2, "2_2" -> fetchable2, "3" -> fetchable3))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant_1:0")
      assert(processedMap._1(1).name === "Constant:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.size === 5)
      assert(results("1_1").isInstanceOf[Tensor[_]])
      assert(results("1_2").isInstanceOf[Tensor[_]])
      assert(results("2_1").isInstanceOf[Tensor[_]])
      assert(results("2_2").isInstanceOf[Tensor[_]])
      assert(results("3").isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableNestedSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedSeq = Session.processFetches(Seq(Seq(Basic.constant(1.0)), Seq(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.length === 2)
      assert(results(0).length === 1)
      assert(results(1).length === 2)
      assert(results(0)(0).isInstanceOf[Tensor[_]])
      assert(results(1)(0).isInstanceOf[Tensor[_]])
      assert(results(1)(1).isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableNestedSeqArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedSeq = Session.processFetches(
        Seq(Seq(Basic.constant(1.0)), Seq(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedSeq._1.length === 3)
      assert(processedSeq._1(0).name === "Constant:0")
      assert(processedSeq._1(1).name === "Constant_1:0")
      assert(processedSeq._1(2).name === "Constant_2:0")
      val results = processedSeq._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.length === 2)
      assert(results(0).length === 1)
      assert(results(1).length === 2)
      assert(results(0)(0).isInstanceOf[Tensor[_]])
      assert(results(1)(0).isInstanceOf[Tensor[_]])
      assert(results(1)(1).isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableNestedMapArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedMap = Session.processFetches(
        Map("1" -> Map("1" -> Basic.constant(1.0)),
          "2" -> Map("2" -> Basic.constant(2.0), "3" -> Basic.constant(3.0))))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.size === 2)
      assert(results("1").size === 1)
      assert(results("2").size === 2)
      assert(results("1")("1").isInstanceOf[Tensor[_]])
      assert(results("2")("2").isInstanceOf[Tensor[_]])
      assert(results("2")("3").isInstanceOf[Tensor[_]])
    }
  }

  @Test def testFetchableNestedMapSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedMap = Session.processFetches(
        Map("1" -> Seq(Basic.constant(1.0)), "2" -> Seq(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedMap._1.length === 3)
      assert(processedMap._1(0).name === "Constant:0")
      assert(processedMap._1(1).name === "Constant_1:0")
      assert(processedMap._1(2).name === "Constant_2:0")
      val results = processedMap._2(Seq.fill(3)(Tensor[Double](0).asUntyped))
      assert(results.size === 2)
      assert(results("1").length === 1)
      assert(results("2").length === 2)
      assert(results("1")(0).isInstanceOf[Tensor[Double]])
      assert(results("2")(0).isInstanceOf[Tensor[Double]])
      assert(results("2")(1).isInstanceOf[Tensor[Double]])
    }
  }

  @Test def testFetchableTuple(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = Session.processFetches((Basic.constant(1.0), Basic.constant(2.0)))
      assert(processedTuple._1.length === 2)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      val results = processedTuple._2(Seq.fill(2)(Tensor[Double](0)))
      assert(results.isInstanceOf[(Tensor[_], Tensor[_])])
    }
  }

  @Test def testFetchableNestedTuple(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = Session.processFetches((Basic.constant(1.0), (Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedTuple._1.length === 3)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      assert(processedTuple._1(2).name === "Constant_2:0")
      val results = processedTuple._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.isInstanceOf[(Tensor[_], (Tensor[_], Tensor[_]))])
    }
  }

  @Test def testFetchableNestedTupleArray(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = Session.processFetches((Basic.constant(1.0), Seq(Basic.constant(2.0), Basic.constant(3.0))))
      assert(processedTuple._1.length === 3)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      assert(processedTuple._1(2).name === "Constant_2:0")
      val results = processedTuple._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.isInstanceOf[(Tensor[_], Seq[_])])
    }
  }

  @Test def testFetchableNestedTupleMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val processedTuple = Session.processFetches(
        (Basic.constant(1.0), Map("2" -> Basic.constant(2.0), "3" -> Basic.constant(3.0))))
      assert(processedTuple._1.length === 3)
      assert(processedTuple._1(0).name === "Constant:0")
      assert(processedTuple._1(1).name === "Constant_1:0")
      assert(processedTuple._1(2).name === "Constant_2:0")
      val results = processedTuple._2(Seq.fill(3)(Tensor[Double](0)))
      assert(results.isInstanceOf[(Tensor[_], Map[String, Tensor[_]])])
    }
  }
}
