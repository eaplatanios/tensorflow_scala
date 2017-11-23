/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
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

package org.platanios.tensorflow.api.ops.control_flow

import org.platanios.tensorflow.api.ops.{Basic, Op, Output, OutputIndexedSlices, SparseOutput}

import scala.collection.mutable

/** Maintains the mapping from while-loops to their gradient states.
  *
  * @author Emmanouil Antonios Platanios
  */
private[ops] class GradientState private[control_flow] () {
  private[this] val map: mutable.Map[Context, GradientLoopState] = mutable.Map.empty[Context, GradientLoopState]

  /** Returns the gradient loop state for `op`, if it is in a forward loop context. */
  private[ops] def getGradientLoopState(op: Op, before: Boolean): Option[GradientLoopState] = {
    val forwardContext = {
      if (before && ControlFlow.isLoopExit(op))
        op.controlFlowContext.flatMap(_.outerContext).flatMap(_.whileLoopContext)
      else
        WhileLoopContext.getWhileLoopContext(op)
    }
    forwardContext.flatMap(map.get)
  }

  /** Enters the appropriate while-loop context used for gradient computation. */
  private[ops] def enterGradientWhileLoopContext(op: Op, before: Boolean): Unit = {
    getGradientLoopState(op, before).foreach(_.backwardContext.enter())
  }

  /** Exits the appropriate while-loop context used for gradient computation. */
  private[ops] def exitGradientWhileLoopContext(op: Op, before: Boolean): Unit = {
    getGradientLoopState(op, before).foreach(_.backwardContext.exit())
  }

  /** Adds the gradient loop state for the while loop that `op` belongs to.
    *
    * Note that `op` must be an exit op, and this method must be called in the same control flow context where
    * `gradients()` is called from.
    *
    * Note that this method modifies `between` and `betweenList`. */
  private[control_flow] def addWhileLoopContext(
      op: Op, between: mutable.Set[Op], betweenList: mutable.ListBuffer[Op]): Unit = {
    WhileLoopContext.getWhileLoopContext(op).foreach(forwardContext => {
      if (!map.contains(forwardContext)) {
        // This is a new while loop and so we create a new gradient loop state for it.
        val outerForwardContext = forwardContext.outerContext.flatMap(_.whileLoopContext)
        val outerGradientLoopState = outerForwardContext.flatMap(map.get)
        val gradientLoopState = GradientLoopState(forwardContext, outerGradientLoopState)
        map += forwardContext -> gradientLoopState
        // We need to include all exit ops of a loop for back-propagation.
        gradientLoopState.forwardLoopExits.filter(e => !between.contains(e.op)).foreach(exit => {
          between += exit.op
          betweenList += exit.op
        })
      }
    })
  }

  /** Creates a zeros tensor with the same data type and shape as the provided loop exit.
    *
    * If the result of a loop variable is not used but is involved in computing the result of some needed loop variable,
    * we create a zero-valued tensor that is fed as gradient for the exit node of that loop variable. Note that
    * `value.op` must be an exit op, and this method must be called in the same control flow context where `gradients()`
    * is called from.
    *
    * @param  value Output of an exit op.
    * @return Zeros tensor with the same data type and shape as `value`.
    */
  private[ops] def zerosLikeForExit(value: Output): Output = {
    val forwardContext = value.op.controlFlowContext
    val outerForwardContext = forwardContext.flatMap(_.outerContext).flatMap(_.whileLoopContext)
    val outerGradientLoopState = outerForwardContext.flatMap(map.get)
    outerGradientLoopState match {
      case Some(gradientLoopState) =>
        // This is a nested loop.
        if (value.shape.isFullyDefined) {
          // If the shape is known statically, we just create a zeros tensor with the right shape in the right context.
          gradientLoopState.backwardContext.enter()
          val result = Basic.zeros(value.dataType, value.shape)
          gradientLoopState.backwardContext.exit()
          result
        } else {
          // Only the shape of `value` is needed for back-propagation.
          forwardContext.flatMap(_.outerContext).foreach(_.enter())
          val shape = Basic.shape(value, optimize = false)
          forwardContext.flatMap(_.outerContext).foreach(_.exit())
          // Save the shape to a stack.
          val historyShape = gradientLoopState.addForwardAccumulator(shape)
          // Get the shape back from the stack.
          gradientLoopState.backwardContext.enter()
          val realShape = gradientLoopState.addBackwardAccumulatedValue(historyShape, shape)
          val result = Basic.zeros(value.dataType, realShape)
          gradientLoopState.backwardContext.exit()
          result
        }
      case None =>
        // This is not a nested loop.
        if (value.shape.isFullyDefined) {
          // If the shape is known statically, we just create a zeros tensor with the right shape.
          Basic.zeros(value.dataType, value.shape)
        } else {
          Basic.zerosLike(value, optimize = false)
        }
    }
  }

  /** Creates a zeros tensor with the same data type and shape as the provided op output.
    *
    * If `op` is in a while loop that is part of `gradients()`, this method must be called in its gradient loop context.
    *
    * @param  op    Op.
    * @param  index Op output index.
    * @return Zeros tensor with the same data type and shape as `op.outputs(index)`.
    */
  private[ops] def zerosLike(op: Op, index: Int): Option[Output] = {
    if (ControlFlow.isLoopSwitch(op)) {
      None
    } else {
      val deadBranch = ControlFlow.isSwitch(op)
      val forwardContext = WhileLoopContext.getWhileLoopContext(op)
      forwardContext.flatMap(map.get) match {
        case Some(gradientLoopState) =>
          // `op` is in a while loop that is part of `gradients()`.
          val value = op.outputs(index)
          if (value.shape.isFullyDefined) {
            // If the shape is known statically, we just create a zeros tensor with the right shape in the right context.
            val result = Basic.zeros(value.dataType, value.shape)
            if (deadBranch) {
              // `op` is a conditional switch and so we guard the zero tensor with a switch.
              op.controlFlowContext.flatMap(c => {
                val condContext = c.asInstanceOf[CondContext]
                gradientLoopState.historyMap.get(condContext.predicate.name).map((_, condContext.branch))
              }) match {
                case Some((predicate, branch)) => Some(
                  branch.other.selectSwitchResult(ControlFlow.colocatedSwitch(result, predicate)))
                case None => Some(result)
              }
            } else {
              Some(result)
            }
          } else {
            // Unknown shape and so we keep a history of the shape at runtime.
            val shape = {
              if (deadBranch) {
                // `op` is a conditional switch and so we guard the zero tensor with a switch.
                op.controlFlowContext.map(c => {
                  val condContext = c.asInstanceOf[CondContext]
                  condContext.outerContext.foreach(_.enter())
                  val value = condContext.branch.other.selectSwitchResult(
                    ControlFlow.colocatedSwitch(op.inputs(0), condContext.predicate))
                  val shape = Basic.shape(value, optimize = false)
                  condContext.outerContext.foreach(_.exit())
                  value.op.controlFlowContext = Some(condContext)
                  shape.op.controlFlowContext = Some(condContext)
                  shape
                })
              } else {
                op.controlFlowContext.foreach(_.enter())
                val shape = Basic.shape(value, optimize = false)
                op.controlFlowContext.foreach(_.exit())
                Some(shape)
              }
            }
            // Add forward accumulator for the shape.
            gradientLoopState.backwardContext.exit()
            val historyShape = gradientLoopState.addForwardAccumulator(shape.get, deadBranch)
            gradientLoopState.backwardContext.enter()
            // Create a zeros tensor with the right shape.
            val realShape = gradientLoopState.addBackwardAccumulatedValue(historyShape, shape.get, deadBranch)
            Some(Basic.zeros(value.dataType, realShape))
          }
        case None =>
          // `op` is not in a while loop that is part of `gradients()`.
          Some(Context.zerosLikeOutsideLoop(op, index))
      }
    }
  }

  /** Processes all of the "unused" loop exits.
    *
    * The "unused" exits of the loops are added to `unusedExits`. An exit is unused if its pending count is 0. If there
    * is an exit with a real gradient, all these deferred exits will enter the back-propagation loop with zero gradient.
    * Otherwise, they will enter the back-propagation loop with `None`. As an example, people often write:
    * {{{
    *   val loopResult = tf.whileLoop(p, b, Seq(x1, x2))
    *   val result = tf.gradients(loopResult._1, x1)
    * }}}
    * The exit node for `x2` is not included because of the between-ness analysis. However, we need to back-propagate
    * `x2` if `x2` is involved in computing `loopResult._1`.
    *
    * @param  pendingCounts  Number of back-propagation inputs for each op.
    * @param  destinationOps Set of destination ops in the gradient computation.
    * @return Set of unused loop exits that we know at this point that we need to back-propagate.
    */
  private[ops] def processUnusedLoopExits(
      pendingCounts: mutable.Map[Op, Int], destinationOps: Set[Op]): Set[Output] = {
    val loopExits = mutable.Set.empty[Output]
    map.values.foreach(gradientLoopState => {
      gradientLoopState.forwardLoopExits.filter(e => pendingCounts.getOrElse(e.op, 0) == 0).foreach(exit => {
        gradientLoopState.pendingExitsCount -= 1
        if (!destinationOps.contains(exit.op))
          gradientLoopState.unusedExits += exit
        if (gradientLoopState.pendingExitsCount == 0)
          loopExits ++= gradientLoopState.unusedExits
      })
      // We need to include enter ops in the back-propagation too, for higher-order gradients.
      gradientLoopState.forwardContext.loopEnters
          .map(_.op)
          .filter(e => pendingCounts.getOrElse(e, 0) == 0)
          .foreach(e => pendingCounts(e) = 1)
    })
    loopExits.toSet
  }

  /** Performs postprocessing at the end of the `gradients()` function call.
    *
    * We have created the gradient graph at this point. So this function can be used to perform any postprocessing on
    * the gradient graph. We currently perform the following postprocessing:
    *
    *   1. Patch the gradient graph if the output of a loop variable doesn't depend on its input.
    */
  private[ops] def postProcess(): Unit = {
    map.values.foreach(gradientLoopState => {
      gradientLoopState.switchMap.values.flatMap({
        case o: Output => Seq(o)
        case o: OutputIndexedSlices =>
          if (o.denseShape == null)
            Seq(o.indices, o.values)
          else
            Seq(o.indices, o.values, o.denseShape)
        case o: SparseOutput =>
          if (o.denseShape == null)
            Seq(o.indices, o.values)
          else
            Seq(o.indices, o.values, o.denseShape)
      }).filter(m => m.op.inputs(0) == m.op.inputs(1)).foreach(merge => {
        // The value of this loop variable at iteration i+1 does not depend on its value at iteration i and so we use
        // zeros as the gradients for all iterations > 0.
        val dataType = merge.op.inputs(0).dataType
        val shape = merge.op.inputs(0).shape
        val nextGradientValue = {
          if (shape.isFullyDefined) {
            gradientLoopState.backwardContext.enter()
            // Create a zeros tensor and use it for iterations > 0.
            val gradientValue = Basic.zeros(dataType, shape)
            val nextGradientValue = ControlFlow.nextIteration(gradientValue)
            gradientLoopState.backwardContext.exit()
            nextGradientValue
          } else {
            // Create a zeros tensor in the outer gradient context.
            val outerGradientContext = gradientLoopState.backwardContext.outerContext
            outerGradientContext.foreach(_.enter())
            val enterGradient = merge.op.inputs(0).op.inputs(0)
            val gradientShape = Basic.shape(enterGradient, optimize = false)
            val gradientValue = Basic.zeros(dataType, gradientShape)
            outerGradientContext.foreach(_.exit())
            // Use the zeros for iterations > 0.
            gradientLoopState.backwardContext.enter()
            val nextGradientValue = ControlFlow.nextIteration(gradientValue)
            gradientLoopState.backwardContext.exit()
            nextGradientValue
          }
        }
        ControlFlow.updateInput(merge.op, 1, nextGradientValue)
      })
    })
  }
}

object GradientState {
  /** Creates the state for all the while loops involved in one `gradients()` call.
    *
    * This function creates a [[GradientState]] when there are while loops involved in the `gradients()` call. In
    * `gradients()`, the control flow logic is only invoked when the gradient state is not `None`.
    *
    * Note that this method modifies `between` and `betweenList`. */
  private[ops] def maybeCreate(
      between: mutable.Set[Op], betweenList: mutable.ListBuffer[Op],
      colocateGradientsWithOps: Boolean): Option[GradientState] = {
    var state: Option[GradientState] = None
    betweenList.filter(ControlFlow.isLoopExit).foreach(op => {
      if (state.isEmpty)
        state = Some(new GradientState())
      if (colocateGradientsWithOps)
        Op.colocateWith(Set(op))(state.foreach(_.addWhileLoopContext(op, between, betweenList)))
      else
        state.foreach(_.addWhileLoopContext(op, between, betweenList))
    })
    state
  }
}
