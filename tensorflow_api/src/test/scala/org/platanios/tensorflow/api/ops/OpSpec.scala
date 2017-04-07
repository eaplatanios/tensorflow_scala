package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Exception.IllegalNameException
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.ops.ArrayOps.constant
import org.platanios.tensorflow.api.ops.MathOps.matMul
import org.platanios.tensorflow.api.ops.Op._
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class OpSpec extends FlatSpec with Matchers {
  //region createWith(...) Specification

  "The 'createWith' function" must "change the default graph (only) for its code block" in {
    val graph1 = Graph()
    val graph2 = Graph()
    createWith(graph = graph1) {
      val graph1Constant = constant(1.0)
      assert(graph1Constant.graph === graph1)
    }
    createWith(graph = graph2) {
      val graph2Constant1 = constant(2.0)
      assert(graph2Constant1.graph === graph2)
      createWith(graph = graph1) {
        val graph1NestedConstant = constant(3.0)
        assert(graph1NestedConstant.graph === graph1)
      }
      val graph2Constant2 = constant(4.0)
      assert(graph2Constant2.graph === graph2)
    }
  }

  //region createWith(nameScope = ...) Specification

  it must "change the name scope for newly created ops (only) for its code block" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      val c1 = constant(2.0, name = "C_1")
      assert(c1.op.name === "C_1")
      createWith(nameScope = "Nested") {
        val nestedC = constant(3.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
      }
      val c2 = constant(4.0, name = "C_2")
      assert(c2.op.name === "C_2")
    }
  }

  it must "allow for creation of nested name scopes" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      createWith(nameScope = "Nested") {
        val nestedC = constant(2.0, name = "C")
        assert(nestedC.op.name === "Nested/C")
        createWith(nameScope = "Inner") {
          val nestedInnerC = constant(3.0, name = "C")
          assert(nestedInnerC.op.name === "Nested/Inner/C")
        }
        createWith(nameScope = "Inner_1") {
          val nestedInner1C = constant(4.0, name = "C")
          assert(nestedInner1C.op.name === "Nested/Inner_1/C")
        }
      }
    }
  }

  it must "reset the name scope when provided an empty string" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      createWith(nameScope = "Nested") {
        val nestedC1 = constant(2.0, name = "C_1")
        assert(nestedC1.op.name === "Nested/C_1")
        createWith(nameScope = "Inner") {
          val nestedInnerC = constant(3.0, name = "C_1")
          assert(nestedInnerC.op.name === "Nested/Inner/C_1")
          createWith(nameScope = "") {
            val c1 = constant(4.0, name = "C_1")
            assert(c1.op.name === "C_1")
            createWith(nameScope = "Nested") {
              val nestedC2 = constant(5.0, name = "C_2")
              assert(nestedC2.op.name === "Nested/C_2")
            }
          }
        }
      }
    }
  }

  //endregion

  //region createWith(device = ...) Specification

  it must "change the device in which newly created ops from its code block (only) are placed" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/CPU:0") {
        val c2 = constant(2.0)
        assert(c2.device === "/device:CPU:0")
        createWith(device = "/GPU:0") {
          val c3 = constant(3.0)
          assert(c3.device === "/device:GPU:0")
        }
      }
      val c4 = constant(4.0)
      assert(c4.device === "")
    }
  }

  it must "allow for nesting of device scopes" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/job:worker/replica:2") {
        val c2 = constant(2.0)
        assert(c2.device === "/job:worker/replica:2")
        createWith(device = "/job:worker/replica:3/task:0") {
          val c3 = constant(3.0)
          assert(c3.device === "/job:worker/replica:3/task:0")
        }
      }
      val c4 = constant(4.0)
      assert(c4.device === "")
    }
  }

  it must "nest device scopes by appropriately overriding device specifications" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/job:worker/replica:2/device:CPU:1") {
        val c2 = constant(2.0)
        assert(c2.device === "/job:worker/replica:2/device:CPU:1")
        createWith(device = "/job:worker/replica:2/device:GPU:2") {
          val c3 = constant(3.0)
          assert(c3.device === "/job:worker/replica:2/device:GPU:2")
        }
      }
      val c4 = constant(4.0)
      assert(c4.device === "")
    }
  }

  it must "nest device scopes by appropriately merging device specifications" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0)
      assert(c1.device === "")
      createWith(device = "/device:GPU:*") {
        val c2 = constant(2.0)
        assert(c2.device === "/device:GPU:*")
        createWith(device = "/job:worker") {
          val c3 = constant(3.0)
          assert(c3.device === "/job:worker/device:GPU:*")
          createWith(device = "/device:CPU:0") {
            val c4 = constant(4.0)
            assert(c4.device === "/job:worker/device:CPU:0")
            createWith(device = "/job:ps") {
              val c5 = constant(5.0)
              assert(c5.device === "/job:ps/device:CPU:0")
            }
          }
        }
        createWith(device = "/device:GPU:5") {
          val c6 = constant(6.0)
          assert(c6.device === "/device:GPU:5")
        }
      }
      val c7 = constant(7.0)
      assert(c7.device === "")
    }
  }

  it must "reset the device whenever an empty string is provided for its device argument" in {
    createWith(graph = Graph()) {
      val c1 = constant(1.0, name = "C_1")
      assert(c1.device === "")
      createWith(device = "/CPU:0") {
        val c2 = constant(2.0)
        assert(c2.device === "/device:CPU:0")
        createWith(device = null) {
          val c3 = constant(3.0)
          assert(c3.device === "")
        }
      }
    }
  }

  it must "be able to use device functions for setting op devices individually" in {
    def matMulOnGPU(opSpecification: OpSpecification): String = {
      if (opSpecification.opType == "MatMul")
        "/GPU:0"
      else
        "/CPU:0"
    }
    createWith(device = matMulOnGPU) {
      val c = constant(1.0)
      assert(c.device === "/device:CPU:0")
      val m = matMul(c, constant(2.0))
      assert(m.device === "/device:GPU:0")
    }
  }

  //endregion

  //region createWith(colocationOps = ...) Specification

  it must "be able to colocate ops" in {
    val a = createWith(device = "/CPU:0")(constant(1.0, name = "A"))
    val b = createWith(device = "/GPU:0")(constant(1.0, name = "B"))
    assert(a.colocationGroups === Set.empty[String])
    assert(b.colocationGroups === Set.empty[String])
    val c = createWith(colocationOps = Set(a))(constant(1.0, name = "C"))
    assert(c.colocationGroups === Set("loc:@A"))
    createWith(colocationOps = Set(b)) {
      val d = constant(1.0, name = "D")
      assert(d.colocationGroups === Set("loc:@B"))
      createWith(colocationOps = Set(a, d)) {
        val e = constant(1.0, name = "E")
        assert(e.colocationGroups === Set("loc:@A", "loc:@B", "loc:@D"))
      }
    }
  }

  //endregion

  //region createWith(controlDependencies = ...) Specification

  it must "change the control dependencies for newly created ops (only) for its code block" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      val b = constant(1.0)
      assert(a.controlInputs.toSet === Set.empty[Op])
      assert(a.controlOutputs.toSet === Set.empty[Op])
      assert(b.controlInputs.toSet === Set.empty[Op])
      assert(b.controlOutputs.toSet === Set.empty[Op])
      val c = createWith(controlDependencies = Set[Op](a)) {
        val c = constant(1.0)
        assert(c.controlInputs.toSet === Set[Op](a))
        assert(c.controlOutputs.toSet === Set.empty[Op])
        assert(a.controlOutputs.toSet === Set[Op](c))
        assert(b.controlOutputs.toSet === Set.empty[Op])
        c
      }
      createWith(controlDependencies = Set[Op](a, b)) {
        val d = constant(1.0)
        assert(d.controlInputs.toSet === Set[Op](a, b))
        assert(d.controlOutputs.toSet === Set.empty[Op])
        assert(a.controlOutputs.toSet === Set[Op](c, d))
        assert(b.controlOutputs.toSet === Set[Op](d))
      }
    }
  }

  it must "allow for nesting of control dependencies specifications" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      val b = constant(1.0)
      assert(a.controlInputs.toSet === Set.empty[Op])
      assert(a.controlOutputs.toSet === Set.empty[Op])
      assert(b.controlInputs.toSet === Set.empty[Op])
      assert(b.controlOutputs.toSet === Set.empty[Op])
      createWith(controlDependencies = Set[Op](a)) {
        val c = constant(1.0)
        assert(c.controlInputs.toSet === Set[Op](a))
        assert(c.controlOutputs.toSet === Set.empty[Op])
        assert(a.controlOutputs.toSet === Set[Op](c))
        assert(b.controlOutputs.toSet === Set.empty[Op])
        createWith(controlDependencies = Set[Op](b)) {
          val d = constant(1.0)
          assert(d.controlInputs.toSet === Set[Op](a, b))
          assert(d.controlOutputs.toSet === Set.empty[Op])
          assert(a.controlOutputs.toSet === Set[Op](c, d))
          assert(b.controlOutputs.toSet === Set[Op](d))
        }
      }
    }
  }

  it must "reset the control dependencies (only) for its code block when provided an empty set" in {
    createWith(graph = Graph()) {
      val a = constant(1.0)
      val b = constant(1.0)
      assert(a.controlInputs.toSet === Set.empty[Op])
      assert(a.controlOutputs.toSet === Set.empty[Op])
      assert(b.controlInputs.toSet === Set.empty[Op])
      assert(b.controlOutputs.toSet === Set.empty[Op])
      createWith(controlDependencies = Set[Op](a)) {
        val c = constant(1.0)
        assert(c.controlInputs.toSet === Set[Op](a))
        assert(c.controlOutputs.toSet === Set.empty[Op])
        assert(a.controlOutputs.toSet === Set[Op](c))
        assert(b.controlOutputs.toSet === Set.empty[Op])
        createWith(controlDependencies = Set.empty[Op]) {
          val d = constant(1.0)
          assert(d.controlInputs.toSet === Set.empty[Op])
          assert(d.controlOutputs.toSet === Set.empty[Op])
          assert(a.controlOutputs.toSet === Set[Op](c))
          assert(b.controlOutputs.toSet === Set.empty[Op])
          createWith(controlDependencies = Set[Op](b)) {
            val e = constant(1.0)
            assert(e.controlInputs.toSet === Set[Op](b))
            assert(e.controlOutputs.toSet === Set.empty[Op])
            assert(a.controlOutputs.toSet === Set[Op](c))
            assert(b.controlOutputs.toSet === Set[Op](e))
          }
        }
      }
    }
  }

  //endregion

  it must "allow changing, the graph, the name scope, and the device used for its code block simultaneously" in {
    val graph1 = Graph()
    val graph2 = Graph()
    createWith(graph = graph1) {
      val graph1Constant = constant(1.0)
      assert(graph1Constant.graph === graph1)
    }
    createWith(graph = graph2, nameScope = "Nested", device = "/GPU:0") {
      val graph2Constant1 = constant(2.0, name = "C")
      assert(graph2Constant1.graph === graph2)
      assert(graph2Constant1.op.name === "Nested/C")
      assert(graph2Constant1.device === "/device:GPU:0")
      createWith(graph = graph1, nameScope = "Inner") {
        val graph1NestedConstant = constant(3.0, name = "C")
        assert(graph1NestedConstant.graph === graph1)
        assert(graph1NestedConstant.op.name === "Nested/Inner/C")
        assert(graph1NestedConstant.device === "/device:GPU:0")
      }
      val graph2Constant2 = constant(4.0)
      assert(graph2Constant2.graph === graph2)
      assert(graph2Constant2.device === "/device:GPU:0")
    }
  }

  //endregion

  "Ops created using the same name" must "have their name made unique by appending an index to it" in {
    createWith(graph = Graph()) {
      val c = constant(1.0, name = "C")
      assert(c.op.name === "C")
      val c1 = constant(2.0, name = "C")
      assert(c1.op.name === "C_1")
      val c2 = constant(3.0, name = "C")
      assert(c2.op.name === "C_2")
      val c3 = constant(4.0, name = "C_3")
      assert(c3.op.name === "C_3")
      val c4 = constant(5.0, name = "C")
      assert(c4.op.name === "C_4")
    }
  }

  "An 'IllegalNameException'" must "be thrown when invalid characters are used in an op's name" in {
    createWith(graph = Graph()) {
      assertThrows[IllegalNameException](constant(1.0, name = "C!"))
      assertThrows[IllegalNameException](constant(1.0, name = "_C"))
      assertThrows[IllegalNameException](constant(1.0, name = "\\C"))
      assertThrows[IllegalNameException](constant(1.0, name = "-C"))
      assertThrows[IllegalNameException](constant(1.0, name = "/C"))
    }
  }

  // TODO: Add name scope exceptions spec.
}
