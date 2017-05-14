package org.platanios.tensorflow.api.ops.optimizers

import org.platanios.tensorflow.api.ops.{Basic, Math, Op}
import org.platanios.tensorflow.api.tf.Variable

/** Optimizer that implements the gradient descent algorithm.
  *
  * @author Emmanouil Antonios Platanios
  */
case class GradientDescent(
    learningRate: Double, useLocking: Boolean = false, name: String = "GradientDescentOptimizer") extends Optimizer {
  private[this] var learningRateTensor: Op.Output = _

  private[this] def getLearningRate(variable: Variable): Op.Output = {
    if (learningRateTensor eq null)
      throw new IllegalStateException("Method 'prepare' has not been called on this optimizer.")
    Math.cast(learningRateTensor, variable.dataType)
  }

  override def prepare(): Unit = {
    learningRateTensor = Basic.constant(learningRate, name = "LearningRate")
  }

  override def applyDense(gradient: Op.Output, variable: Variable): Op = {
    GradientDescent.resourceApplyDense(variable, getLearningRate(variable), gradient, useLocking)
  }

  override def applySparse(gradient: Op.OutputIndexedSlices, variable: Variable): Op = {
    variable.assignScatterSub(gradient.indices, -gradient.values * getLearningRate(variable)).op
  }

  override def applySparseDuplicateIndices(gradient: Op.OutputIndexedSlices, variable: Variable): Op = {
    applySparse(gradient, variable)
  }
}

object GradientDescent {
  /** Creates an op that updates the value of `variable` by subtracting `stepSize * gradient` from it.
    *
    * @param  variable   Variable whose value to update.
    * @param  stepSize   Step size to use for the gradient descent update.
    * @param  gradient   Gradient to apply.
    * @param  useLocking If `true`, the subtraction will be protected by a lock. Otherwise, the behavior is undefined,
    *                    but may exhibit less contention.
    * @param  name       Name for the created op.
    * @return Created op.
    */
  private[GradientDescent] def resourceApplyDense(
      variable: Variable, stepSize: Op.Output, gradient: Op.Output, useLocking: Boolean = false,
      name: String = "ResourceApplyGradientDescent"): Op = {
    Op.Builder(opType = "ResourceApplyGradientDescent", name = name)
        .addInput(variable.handle)
        .addInput(stepSize)
        .addInput(gradient)
        .setAttribute("use_locking", useLocking)
        .build()
  }
}
