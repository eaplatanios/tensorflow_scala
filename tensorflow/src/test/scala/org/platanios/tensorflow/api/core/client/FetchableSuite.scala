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
  case class DummyFetchable(uniqueFetchesNumber: Int) extends Fetchable[Tensor] {
    private[this] val fetches = (0 until uniqueFetchesNumber).map(Basic.constant(_))

    override def uniqueFetches: Seq[Op.Output] = fetches
    override def buildResult(values: Seq[Tensor]): Tensor = Tensor(values: _*)
  }

  @Test def testFetchable(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      assert(fetchable1.uniqueFetches.length === 1)
      assert(fetchable1.uniqueFetches(0).name === "Constant:0")
      assert(fetchable1.buildResult(Seq.fill(1)(Tensor(0))).isInstanceOf[Tensor])
      assert(fetchable2.uniqueFetches.length === 2)
      assert(fetchable2.uniqueFetches(0).name === "Constant_1:0")
      assert(fetchable2.uniqueFetches(1).name === "Constant_2:0")
      assert(fetchable2.buildResult(Seq.fill(2)(Tensor(0))).isInstanceOf[Tensor])
      assert(fetchable3.uniqueFetches.length === 3)
      assert(fetchable3.uniqueFetches(0).name === "Constant_3:0")
      assert(fetchable3.uniqueFetches(1).name === "Constant_4:0")
      assert(fetchable3.uniqueFetches(2).name === "Constant_5:0")
      assert(fetchable3.buildResult(Seq.fill(3)(Tensor(0))).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableSeq(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchableSeqUnique: Fetchable[Seq[Tensor]] = Seq(fetchable1, fetchable2, fetchable3)
      assert(fetchableSeqUnique.uniqueFetches.length === 6)
      assert(fetchableSeqUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableSeqUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableSeqUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableSeqUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableSeqUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableSeqUnique.uniqueFetches(5).name === "Constant_5:0")
      val results = fetchableSeqUnique.buildResult(Seq.fill(6)(Tensor(0)))
      assert(results.length === 3)
      assert(results(0).isInstanceOf[Tensor])
      assert(results(1).isInstanceOf[Tensor])
      assert(results(2).isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableMap(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchableSeqUnique: Fetchable[Map[String, Tensor]] =
        Map("1" -> fetchable1, "2" -> fetchable2, "3" -> fetchable3)
      assert(fetchableSeqUnique.uniqueFetches.length === 6)
      assert(fetchableSeqUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableSeqUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableSeqUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableSeqUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableSeqUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableSeqUnique.uniqueFetches(5).name === "Constant_5:0")
      val results = fetchableSeqUnique.buildResult(Seq.fill(6)(Tensor(0)))
      assert(results.size === 3)
      assert(results("1").isInstanceOf[Tensor])
      assert(results("2").isInstanceOf[Tensor])
      assert(results("3").isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableSeqWithDuplicates(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchableSeqUnique: Fetchable[Seq[Tensor]] = Seq(fetchable1, fetchable1, fetchable2, fetchable2, fetchable3)
      assert(fetchableSeqUnique.uniqueFetches.length === 6)
      assert(fetchableSeqUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableSeqUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableSeqUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableSeqUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableSeqUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableSeqUnique.uniqueFetches(5).name === "Constant_5:0")
      val results = fetchableSeqUnique.buildResult(Seq.fill(6)(Tensor(0)))
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
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchableSeqUnique: Fetchable[Map[String, Tensor]] =
        TreeMap("1_1" -> fetchable1, "1_2" -> fetchable1, "2_1" -> fetchable2, "2_2" -> fetchable2, "3" -> fetchable3)
      assert(fetchableSeqUnique.uniqueFetches.length === 6)
      assert(fetchableSeqUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableSeqUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableSeqUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableSeqUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableSeqUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableSeqUnique.uniqueFetches(5).name === "Constant_5:0")
      val results = fetchableSeqUnique.buildResult(Seq.fill(6)(Tensor(0)))
      assert(results.size === 5)
      assert(results("1_1").isInstanceOf[Tensor])
      assert(results("1_2").isInstanceOf[Tensor])
      assert(results("2_1").isInstanceOf[Tensor])
      assert(results("2_2").isInstanceOf[Tensor])
      assert(results("3").isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableTuple1(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchableTupleUnique: Fetchable[Tuple1[Tensor]] = Tuple1(fetchable1)
      assert(fetchableTupleUnique.uniqueFetches.length === 1)
      assert(fetchableTupleUnique.uniqueFetches(0).name === "Constant:0")
      val results = fetchableTupleUnique.buildResult(Seq.fill(1)(Tensor(0)))
      assert(results.isInstanceOf[Tuple1[Tensor]])
      assert(results._1.isInstanceOf[Tensor])
    }
  }
Tuple10
  @Test def testFetchableTuple2(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchableTupleUnique: Fetchable[(Tensor, Tensor)] = (fetchable1, fetchable2)
      assert(fetchableTupleUnique.uniqueFetches.length === 3)
      assert(fetchableTupleUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableTupleUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableTupleUnique.uniqueFetches(2).name === "Constant_2:0")
      val results = fetchableTupleUnique.buildResult(Seq.fill(3)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Tensor)])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2.isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableTuple3(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchableTupleUnique: Fetchable[(Tensor, Tensor, Tensor)] = (fetchable1, fetchable2, fetchable3)
      assert(fetchableTupleUnique.uniqueFetches.length === 6)
      assert(fetchableTupleUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableTupleUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableTupleUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableTupleUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableTupleUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableTupleUnique.uniqueFetches(5).name === "Constant_5:0")
      val results = fetchableTupleUnique.buildResult(Seq.fill(6)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Tensor, Tensor)])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2.isInstanceOf[Tensor])
      assert(results._3.isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableTuple4(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchable4 = DummyFetchable(1)
      val fetchableTupleUnique: Fetchable[(Tensor, Tensor, Tensor, Tensor)] =
        (fetchable1, fetchable2, fetchable3, fetchable4)
      assert(fetchableTupleUnique.uniqueFetches.length === 7)
      assert(fetchableTupleUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableTupleUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableTupleUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableTupleUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableTupleUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableTupleUnique.uniqueFetches(5).name === "Constant_5:0")
      assert(fetchableTupleUnique.uniqueFetches(6).name === "Constant_6:0")
      val results = fetchableTupleUnique.buildResult(Seq.fill(7)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Tensor, Tensor, Tensor)])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2.isInstanceOf[Tensor])
      assert(results._3.isInstanceOf[Tensor])
      assert(results._4.isInstanceOf[Tensor])
    }
  }

  @Test def testFetchableTuple5(): Unit = using(Graph()) { graph =>
    Op.createWith(graph) {
      val fetchable1 = DummyFetchable(1)
      val fetchable2 = DummyFetchable(2)
      val fetchable3 = DummyFetchable(3)
      val fetchable4 = DummyFetchable(1)
      val fetchable5 = DummyFetchable(1)
      val fetchableTupleUnique: Fetchable[(Tensor, Tensor, Tensor, Tensor, Tensor)] =
        (fetchable1, fetchable2, fetchable3, fetchable4, fetchable5)
      assert(fetchableTupleUnique.uniqueFetches.length === 8)
      assert(fetchableTupleUnique.uniqueFetches(0).name === "Constant:0")
      assert(fetchableTupleUnique.uniqueFetches(1).name === "Constant_1:0")
      assert(fetchableTupleUnique.uniqueFetches(2).name === "Constant_2:0")
      assert(fetchableTupleUnique.uniqueFetches(3).name === "Constant_3:0")
      assert(fetchableTupleUnique.uniqueFetches(4).name === "Constant_4:0")
      assert(fetchableTupleUnique.uniqueFetches(5).name === "Constant_5:0")
      assert(fetchableTupleUnique.uniqueFetches(6).name === "Constant_6:0")
      assert(fetchableTupleUnique.uniqueFetches(7).name === "Constant_7:0")
      val results = fetchableTupleUnique.buildResult(Seq.fill(8)(Tensor(0)))
      assert(results.isInstanceOf[(Tensor, Tensor, Tensor, Tensor, Tensor)])
      assert(results._1.isInstanceOf[Tensor])
      assert(results._2.isInstanceOf[Tensor])
      assert(results._3.isInstanceOf[Tensor])
      assert(results._4.isInstanceOf[Tensor])
      assert(results._5.isInstanceOf[Tensor])
    }
  }
}
