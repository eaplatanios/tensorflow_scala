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

import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.ops.{Basic, DataFlow, Op, Output, OutputLike}

import scala.collection.mutable

/** State used for constructing the gradient graph for a while loop.
  *
  * We create a [[GradientLoopState]] for each while loop in the forward pass and its corresponding while loop in the
  * backward pass. This gives us access to both the forward and the backward [[WhileLoopContext]]s.
  *
  * During the construction of the gradient graph, whenever we detect a forward value that is needed for the backward
  * pass, we create a history accumulator and add it to `historyMap`. Whenever we back-propagate a loop switch op, we
  * add the corresponding gradient merge op in `switchMap`.
  *
  * @param  forwardContext         While-loop context used for the forward pass.
  * @param  outerGradientLoopState The gradient loop state used for the outer loop.
  *
  * @author Emmanouil Antonios Platanios
  */
private[control_flow] case class GradientLoopState private[control_flow] (
    forwardContext: WhileLoopContext, outerGradientLoopState: Option[GradientLoopState]) {
  /** Map that records all tensors needed for back-propagation. */
  private[control_flow] val historyMap: mutable.Map[String, Output] = mutable.Map.empty[String, Output]

  /** Map that records all the switch ops needed for back-propagation. */
  private[control_flow] val switchMap: mutable.Map[Op, OutputLike] = mutable.Map.empty[Op, OutputLike]

  /** List containing all "unused" exits. */
  private[ops] val unusedExits: mutable.Set[Output] = mutable.Set.empty[Output]

  /** List containing all "deferred" exits. */
  private[ops] val deferredExits: mutable.Set[Output] = mutable.Set.empty[Output]

  /** List containing all forward loop exits. */
  private[control_flow] val forwardLoopExits: mutable.Set[Output] = mutable.Set(forwardContext.loopExits.toSeq: _*)

  /** Number of exits we expect to see during the backward pass, but have not seen yet. */
  private[ops] var pendingExitsCount: Int = forwardContext.loopExits.size

  /** `forwardIndex`:    Value of the loop counter for the next iteration added by `addForwardLoopCounter()`.
    * `backwardIndex`:   Value of the loop counter for the current iteration added by `addBackwardLoopCounter()`.
    * `backwardContext`: While loop context used for backpropagation. */
  private[control_flow] val (forwardIndex: Output, backwardIndex: Output, backwardContext: WhileLoopContext) = {
    val outerForwardContext = outerGradientLoopState.map(_.forwardContext).orElse(forwardContext.outerContext)
    // Add the forward loop counter.
    outerForwardContext.foreach(_.enter())
    val (count, forwardIndex) = forwardContext.addForwardLoopCounter(outerGradientLoopState)
    outerForwardContext.foreach(_.exit())

    // Add the backward while-loop context, and the backward loop counter.
    val (backwardIndex, backwardContext): (Output, WhileLoopContext) = outerGradientLoopState match {
      case Some(state) =>
        // This is a nested loop. Remember the iteration counts for each execution of this inner loop.
        outerForwardContext.foreach(_.values += count.name)
        val historyCount = state.addForwardAccumulator(count)
        state.backwardContext.enter()
        val backwardContext = WhileLoopContext(
          forwardContext.parallelIterations,
          forwardContext.enableBackPropagation,
          forwardContext.swapMemory,
          _gradientLoopState = Some(this),
          _name = forwardContext.name)
        val realCount = state.addBackwardAccumulatedValue(historyCount, count)
        val backwardIndex = backwardContext.addBackwardLoopCounter(realCount, outerGradientLoopState)
        state.backwardContext.exit()
        (backwardIndex, backwardContext)
      case None =>
        outerForwardContext.foreach(_.enter())
        val backwardContext = WhileLoopContext(
          forwardContext.parallelIterations,
          forwardContext.enableBackPropagation,
          forwardContext.swapMemory,
          _gradientLoopState = Some(this),
          _name = forwardContext.name)
        val backwardIndex = backwardContext.addBackwardLoopCounter(count, outerGradientLoopState)
        outerForwardContext.foreach(_.exit())
        (backwardIndex, backwardContext)
    }

    (forwardIndex, backwardIndex, backwardContext)
  }

  /** Control trigger node for synchronization in the forward loop. One main use is to keep the push ops of a stack
    * executed in the iteration order. */
  private[control_flow] lazy val forwardSync: Op = {
    val syncOp = Op.createWith(controlDependencies = Set.empty[Op])(ControlFlow.controlTrigger("ForwardSync"))
    syncOp.controlFlowContext = Some(forwardContext)
    ControlFlow.addControlInput(forwardIndex.op, syncOp)
    syncOp
  }

  /** Control trigger node for synchronization in the backward loop. One main use is to keep the pop ops of a stack
    * executed in the iteration order. */
  private[control_flow] lazy val backwardSync: Op = {
    val syncOp = Op.createWith(controlDependencies = Set.empty[Op])(ControlFlow.controlTrigger("BackwardSync"))
    syncOp.controlFlowContext = Some(backwardContext)
    ControlFlow.addControlInput(backwardIndex.op, syncOp)
    syncOp
  }

  /** Gets the real value of `value`.
    *
    * If back-propagation "uses" a value produced by the forward loop, an accumulator is added in the forward loop to
    * collect its values. We use the accumulated value. This method must be called for the backward loop context.
    * `value` must be in the forward loop and is needed for back-propagation. */
  private[control_flow] def getRealValue(value: Output): Output = {
    historyMap.getOrElseUpdate(value.name, {
      var realValue: Option[Output] = None
      var historyValue: Output = null
      var currentValue = value
      var currentGradientLoopState = this
      var loopCondition = true
      while (loopCondition) {
        ControlFlow.getLoopConstantEnter(currentValue) match {
          case Some(enterOp) =>
            // Special case: `currentValue` comes from a constant enter node.
            currentValue = enterOp.inputs(0)
            currentGradientLoopState.outerGradientLoopState match {
              case Some(outerState) => currentGradientLoopState = outerState
              case None =>
                // We are now outside all nested loops for this gradient and so `value` is a loop invariant and there is
                // no need to save its history. We just make `currentValue` enter the right control flow context.
                realValue = Some(backwardContext.add(currentValue))
                loopCondition = false
            }
          // case _ if currentValue.op.opType == "Const" =>
          //   realValue = Output.constantValue(currentValue).map(Basic.constant(_))
          //   loopCondition = false
          case _ =>
            // TODO: !!! [CONTROL_FLOW] Consider keeping constants outside the loop avoiding the accumulator for them.
            // Record the history of this value in the forward context.
            backwardContext.exit()
            historyValue = currentGradientLoopState.addForwardAccumulator(currentValue)
            backwardContext.enter()
            loopCondition = false
        }
      }
      realValue.getOrElse({
        // Add the stack pop op in the backward context.
        var realValue = currentGradientLoopState.addBackwardAccumulatedValue(historyValue, currentValue)
        if (currentGradientLoopState != this)
          realValue = backwardContext.add(realValue)
        realValue
      })
    })
  }

  /** Adds an accumulator for each forward tensor that is needed in the backward loop.
    *
    * This is added to the forward loop the first time when a tensor is used by the back-propagation gradient
    * computation loop. We create an accumulator that collects the value of the tensor at each iteration.
    *
    * The pseudocode is: `acc = newStack(); while (pivot) { acc = stackPush(acc, value); }`
    *
    * We make sure that the stack push op in one iteration is executed before next iteration. This is achieved by adding
    * a control edge from `forwardIndex.op.inputs(0).op` to the push op, and another control edge from the push op to
    * either `forwardIndex.op` or `forwardSync`.
    *
    * @param  value      Source tensor in the forward loop that is to be accumulated.
    * @param  deadBranch Set to `true`, if and only if `value` is on a dead branch of a conditional.
    * @return Resource handle to a stack that contains the accumulated history of the tensor.
    */
  private[control_flow] def addForwardAccumulator(value: Output, deadBranch: Boolean = false): Output = {
    val currentContext = Op.currentControlFlowContext
    Op.createWith(controlDependencies = Set.empty[Op]) {
      currentContext.foreach(_.enter())
      val accumulator = Op.colocateWith(Set(value.op)) {
        DataFlow.newStack(-1, value.dataType, name = "ForwardAccumulator")
      }
      currentContext.foreach(_.exit())
      // Make the `accumulator` available in the forward context.
      val enterAccumulator = forwardContext.add(accumulator)
      // Add the stack push op in the context of `value.op`.
      val stackPushOp = ControlFlow.getOutputContext(value.op) match {
        case Some(valueContext) if valueContext == forwardContext =>
          // `value` is not nested in the forward context.
          forwardContext.enter()
          val stackPushOp = DataFlow.stackPush(enterAccumulator, value, forwardContext.swapMemory).op
          forwardContext.exit()
          // Protect the stack push and order it before `forwardIndex`.
          ControlFlow.addControlInput(forwardIndex.op, stackPushOp)
          stackPushOp
        case Some(valueContext: CondContext) =>
          // `value` is in a conditional context within the forward context.
          val stackPushOp = {
            if (deadBranch) {
              // Special case for creating a zero tensor for a dead branch of a switch.
              valueContext.outerContext.foreach(_.enter())
              val stackPushOp = DataFlow.stackPush(enterAccumulator, value, forwardContext.swapMemory).op
              valueContext.outerContext.foreach(_.exit())
              stackPushOp.controlFlowContext = Some(valueContext)
              stackPushOp
            } else {
              valueContext.enter()
              val stackPushOp = DataFlow.stackPush(enterAccumulator, value, forwardContext.swapMemory).op
              valueContext.exit()
              stackPushOp
            }
          }
          // Protect the stack push and order it before `forwardSync`.
          ControlFlow.addControlInput(forwardSync, stackPushOp)
          stackPushOp
        case valueContext => throw InvalidArgumentException(s"'valueContext' is not a CondContext: $valueContext.")
      }
      // Order the stack push after the successor of `forwardIndex`.
      ControlFlow.addControlInput(stackPushOp, forwardIndex.op.inputs(0).op)
      accumulator
    }
  }

  /** Adds the getter for an accumulated value in the backward context.
    *
    * This is added to the back-propagation loop. It is called in the backward context to get the value of an
    * accumulated value. The stack pop op must be guarded by the predicate of the controlling conditional context.
    *
    * @param  historyValue Resource handle to stack containing the "history" of a value.
    * @param  value        Value that is pushed into the stack.
    * @param  deadBranch   Set to `true`, if and only if `value` is on a dead branch of a conditional.
    * @return Current value (popped from the top of the stack).
    */
  private[control_flow] def addBackwardAccumulatedValue(
      historyValue: Output, value: Output, deadBranch: Boolean = false): Output = {
    val historyContext = historyValue.op.controlFlowContext
    // Find the cond context that controls `historyValue`, if any.
    var condContext: Option[CondContext] = None
    var valueContext = value.op.controlFlowContext
    while (condContext.isEmpty && valueContext.isDefined && valueContext != historyContext) {
      valueContext match {
        case Some(context: CondContext) => condContext = Some(context)
        case _ => ()
      }
      valueContext = valueContext.get.outerContext
    }
    val stackPopOp = Op.createWith(controlDependencies = Set.empty[Op]) {
      backwardContext.enter()
      val stackHandle = condContext.map(c => {
        // Guard the stack pop op with a switch, if it is controlled by a conditional.
        var predicate: Option[Output] = None
        var gradientLoopState: Option[GradientLoopState] = Some(this)
        while (predicate.isEmpty && gradientLoopState.isDefined) {
          predicate = gradientLoopState.get.historyMap.get(c.predicate.name)
          gradientLoopState = gradientLoopState.flatMap(_.outerGradientLoopState)
        }
        if (predicate.isEmpty)
          predicate = Some(c.predicate)
        val switch = ControlFlow.colocatedSwitch(historyValue, predicate.get)
        c.branch match {
          case TrueBranch if deadBranch => switch._1
          case TrueBranch if !deadBranch => switch._2
          case FalseBranch if !deadBranch => switch._1
          case FalseBranch if deadBranch => switch._2
        }
      }).getOrElse(historyValue)
      val stackPopOp = DataFlow.stackPop(stackHandle, value.dataType)
      stackPopOp.setShape(value.shape)
      backwardContext.exit()
      stackPopOp
    }
    if (backwardContext.parallelIterations > 1) {
      // All stack pop ops are ordered after `pivotForBody` and before `backwardSync`.
      ControlFlow.addControlInput(backwardSync, stackPopOp.op)
    }
    stackPopOp
  }
}
