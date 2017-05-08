package org.platanios.tensorflow.api.ops.optimizers

import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor}
import org.platanios.tensorflow.api.ops.{Basic, Op, Variable}
import org.platanios.tensorflow.api.types.DataType

import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientDescentSpec extends FlatSpec with Matchers {
  "Gradient descent" must "work for dense updates to resource-based variables" in {
    for (dataType <- Set[DataType](DataType.Float32, DataType.Float64)) {
      val value0 = Tensor(dataType, 1.0, 2.0)
      val value1 = Tensor(dataType, 3.0, 4.0)
      val updatedValue0 = Tensor(dataType, 1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1)
      val updatedValue1 = Tensor(dataType, 3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01)
      val graph = Graph()
      val (variable0, variable1, gdOp) = Op.createWith(graph) {
        val variable0 = Variable(Variable.ConstantInitializer(Tensor(1, 2)), shape = Shape(2), dataType = dataType)
        val variable1 = Variable(Variable.ConstantInitializer(Tensor(3, 4)), shape = Shape(2), dataType = dataType)
        val gradient0 = Basic.constant(Tensor(0.1, 0.1), dataType = dataType)
        val gradient1 = Basic.constant(Tensor(0.01, 0.01), dataType = dataType)
        val gdOp = GradientDescent(3.0).applyGradients(Seq((gradient0, variable0), (gradient1, variable1)))
        (variable0, variable1, gdOp)
      }
      val session = Session(graph)
      session.run(targets = Array(Variable.initializer(graph.trainableVariables)))
      var variable0Value = session.run(fetches = Array(variable0.value)).head
      var variable1Value = session.run(fetches = Array(variable1.value)).head
      assert(variable0Value === value0 +- 1e-6)
      assert(variable1Value === value1 +- 1e-6)
      session.run(targets = Array(gdOp))
      variable0Value = session.run(fetches = Array(variable0.value)).head
      variable1Value = session.run(fetches = Array(variable1.value)).head
      assert(variable0Value === updatedValue0 +- 1e-6)
      assert(variable1Value === updatedValue1 +- 1e-6)
    }
  }
}
