package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.ops.Op.OutputLike

/**
  * @author Emmanouil Antonios Platanios
  */
package object ops {
  def registerGradientFunction(opType: String, function: (Op, Seq[OutputLike]) => Seq[OutputLike]): Unit = {
    Gradients.registerGradientFunction(opType, function)
  }

  def registerNonDifferentiable(opType: String): Unit = Gradients.registerNonDifferentiable(opType)
}
