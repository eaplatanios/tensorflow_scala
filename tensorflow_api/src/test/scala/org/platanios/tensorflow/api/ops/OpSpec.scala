package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Exception.IllegalNameException
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.ops.ArrayOps.constant
import org.platanios.tensorflow.api.ops.Op._
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class OpSpec extends FlatSpec with Matchers {
  "The 'usingGraph' function" must "change the default graph (only) for its code block" in {
    val graph1 = Graph()
    val graph2 = Graph()
    usingGraph(graph1) {
      val graph1Constant = constant(1.0)
      assert(graph1Constant.graph === graph1)
    }
    usingGraph(graph2) {
      val graph2Constant1 = constant(2.0)
      assert(graph2Constant1.graph === graph2)
      usingGraph(graph1) {
        val graph1NestedConstant = constant(3.0)
        assert(graph1NestedConstant.graph === graph1)
      }
      val graph2Constant2 = constant(4.0)
      assert(graph2Constant2.graph === graph2)
    }
  }

  "The 'usingNameScope' function" must "change the name scope for newly created ops (only) for its code block" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "c")
      assert(c.op.name === "c")
      val c1 = constant(2.0, name = "c_1")
      assert(c1.op.name === "c_1")
      usingNameScope("nested") {
        val nestedC = constant(3.0, name = "c")
        assert(nestedC.op.name === "nested/c")
      }
      val c2 = constant(4.0, name = "c_2")
      assert(c2.op.name === "c_2")
    }
  }

  it must "allow for creation of nested name scopes" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "c")
      assert(c.op.name === "c")
      usingNameScope("nested") {
        val nestedC = constant(2.0, name = "c")
        assert(nestedC.op.name === "nested/c")
        usingNameScope("inner") {
          val nestedInnerC = constant(3.0, name = "c")
          assert(nestedInnerC.op.name === "nested/inner/c")
        }
        usingNameScope("inner_1") {
          val nestedInner1C = constant(4.0, name = "c")
          assert(nestedInner1C.op.name === "nested/inner_1/c")
        }
      }
    }
  }

  it must "reset the name scope when provided an empty string or 'null'" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "c")
      assert(c.op.name === "c")
      usingNameScope("nested") {
        val nestedC = constant(2.0, name = "c")
        assert(nestedC.op.name === "nested/c")
        usingNameScope("inner") {
          val nestedInnerC1 = constant(3.0, name = "c_1")
          assert(nestedInnerC1.op.name === "nested/inner/c_1")
          usingNameScope("") {
            val c1 = constant(4.0, name = "c_1")
            assert(c1.op.name === "c_1")
          }
          val nestedInnerC2 = constant(5.0, name = "c_2")
          assert(nestedInnerC2.op.name == "nested/inner/c_2")
          usingNameScope(null) {
            val c2 = constant(6.0, name = "c_2")
            assert(c2.op.name === "c_2")
          }
        }
      }
    }
  }

  "Ops created using the same name" must "have their name made unique by appending an index to it" in {
    usingGraph(Graph()) {
      val c = constant(1.0, name = "c")
      assert(c.op.name === "c")
      val c1 = constant(2.0, name = "c")
      assert(c1.op.name === "c_1")
      val c2 = constant(3.0, name = "c")
      assert(c2.op.name === "c_2")
      val c3 = constant(4.0, name = "c_3")
      assert(c3.op.name === "c_3")
      val c4 = constant(5.0, name = "c")
      assert(c4.op.name === "c_4")
    }
  }

  "An 'IllegalNameException'" must "be thrown when invalid characters are used in an op's name" in {
    usingGraph(Graph()) {
      assertThrows[IllegalNameException](constant(1.0, name = "c!"))
      assertThrows[IllegalNameException](constant(1.0, name = "_c"))
      assertThrows[IllegalNameException](constant(1.0, name = "\\c"))
      assertThrows[IllegalNameException](constant(1.0, name = "-c"))
      assertThrows[IllegalNameException](constant(1.0, name = "/c"))
    }
  }
}
