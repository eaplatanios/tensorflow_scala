/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{RESOURCE, TF}
import org.platanios.tensorflow.api.utilities.using
import org.platanios.tensorflow.jni.{TensorFlow => NativeLibrary}

import org.tensorflow.framework.AttrValue

/** Contains functions for constructing ops related to control flow.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] trait ControlFlow {
  /** Creates an op that produces the content of `input` only after all ops in `dependencies` have finished executing.
    *
    * In some cases, a user may want the output of an op to be consumed externally only after some other dependencies
    * have run first. This function ensures returns `input`, but only after all ops in `dependencies` have run. Note
    * that this means that there is no guarantee that `input` will be evaluated after any `dependencies` have run.
    *
    * @group ControlFlowOps
    * @param  dependencies Set of ops to be executed before `input`.
    * @param  input        Op output to be computed after all ops in `dependencies` have finished executing.
    * @param  name         Name for the created op (used mainly as a name scope).
    * @return Created op output.
    */
  private[api] def withControlDependencies[T: TF, OL[A] <: OutputLike[A]](
      dependencies: Set[UntypedOp],
      input: OL[T],
      name: String = "WithControlDependencies"
  ): OL[T] = {
    Op.nameScope(name) {
      Op.colocateWith(Set(input.op), ignoreExisting = true) {
        Op.createWith(controlDependencies = dependencies) {
          Basic.identity(input)
        }
      }
    }
  }

  /** $OpDocControlFlowGroup
    *
    * @group ControlFlowOps
    * @param  inputs Ops to group.
    * @param  name   Name for the created op (used mainly as a name scope).
    * @return Created op output, which in this case is the result of a `noOp`.
    */
  def group(
      inputs: Set[UntypedOp],
      name: String = "Group"
  ): Op[Unit, Unit] = {
    if (inputs.isEmpty) {
      noOp(name)
    } else {
      Op.createWith(graph = Op.getGraphFromInputs(inputs)) {
        val inputsByDevice = inputs.groupBy(_.device)
        if (inputsByDevice.size == 1) {
          // 1-level tree. The root node is the returned no-op node.
          val (device, ops) = inputsByDevice.head
          if (device != null)
            Op.createWith(device = device, controlDependencies = ops)(noOp(name))
          else
            Op.createWith(controlDependencies = ops)(noOp(name))
        } else {
          // 2-level tree. The root node is the returned no-op node.
          // `dependencies` contains 1 NoOp node for each device.
          val dependencies = inputsByDevice.toSeq.sortBy(_._1).map {
            case (device, ops) =>
              if (device != null)
                Op.createWith(device = device, controlDependencies = ops)(noOp(name).asUntyped)
              else
                Op.createWith(controlDependencies = ops)(noOp(name).asUntyped)
          }
          Op.createWith(controlDependencies = dependencies.toSet)(noOp(name))
        }
      }
    }
  }

  /** $OpDocControlFlowTuple
    *
    * @group ControlFlowOps
    * @param  inputs        Op outputs being grouped.
    * @param  controlInputs Set of additional ops that have to finish before this op finishes, but whose outputs are not
    *                       returned.
    * @param  name          Name for the created ops (used mainly as a name scope).
    * @return Created op outputs, which in this case are the values of `inputs`.
    */
  def tuple[T: TF, OL[A] <: OutputLike[A]](
      inputs: Seq[OL[T]],
      controlInputs: Set[UntypedOp] = Set.empty,
      name: String = "Tuple"
  ): Seq[OL[T]] = {
    val gatingOps = inputs.filter(_ != null).map(_.op).toSet
    if (gatingOps.isEmpty) {
      inputs
    } else {
      Op.nameScope(name) {
        val gate = group(gatingOps ++ controlInputs).asUntyped
        inputs.map(input => {
          if (input == null)
            input
          else
            withControlDependencies(Set(gate), input)
        })
      }
    }
  }

  /** $OpDocControlFlowNoOp
    *
    * @group ControlFlowOps
    * @param  name Name for the created op.
    * @return Created op output.
    */
  def noOp(name: String = "NoOp"): Op[Unit, Unit] = {
    Op.Builder[Unit, Unit](
      opType = "NoOp",
      name = name,
      input = ()
    ).build()
  }

  /** Creates an op that raises an exception to abort the process when called.
    *
    * @group ControlFlowOps
    * @param  errorMessage     Error message associated with the exception.
    * @param  exitWithoutError If `true`, the process will exit normally. Otherwise, it will exit with a `SIGABORT`
    *                          signal.
    * @param  name             Name for the created op.
    * @return Created op output.
    */
  private[api] def abort(
      errorMessage: String = "",
      exitWithoutError: Boolean = false,
      name: String = "Abort"
  ): Op[Unit, Unit] = {
    Op.Builder[Unit, Unit](
      opType = "Abort",
      name = name,
      input = ()
    ).setAttribute("error_message", errorMessage)
        .setAttribute("exit_without_error", exitWithoutError)
        .build()
  }

  /** $OpDocControlFlowCond
    *
    * @group ControlFlowOps
    * @param  predicate `BOOLEAN` scalar determining whether to return the result of `trueFn` or `falseFn`.
    * @param  trueFn    Function returning the computation to be performed if `predicate` is `true`.
    * @param  falseFn   Function returning the computation to be performed if `predicate` is `false`.
    * @param  name      Name prefix for the created ops.
    * @return Created op output structure, mirroring the return structure of `trueFn` and `falseFn`.
    * @throws InvalidDataTypeException If the data types of the tensors returned by `trueFn` and `falseFn` do not match.
    */
  @throws[InvalidDataTypeException]
  def cond[T, R](
      predicate: Output[Boolean],
      trueFn: () => T,
      falseFn: () => T,
      name: String = "Cond"
  )(implicit ev: CondOutput.Aux[T, R]): T = {
    Op.nameScope(name) {
      Output.constantValue(predicate) match {
        case Some(predicateValue) if predicateValue.scalar => trueFn()
        case Some(predicateValue) if !predicateValue.scalar => falseFn()
        case None =>
          // Add the switch to the graph.
          val (pFalse, pTrue) = ControlFlow.switch(predicate, predicate)
          val pivotTrue = Basic.identity(pTrue, "SwitchTrue")
          val pivotFalse = Basic.identity(pFalse, "SwitchFalse")
          val predicateId = Basic.identity(predicate, "PredicateIdentity")
          // Disable the fetching of tensors that are only on one branch of the cond.
          pTrue.op.graph.preventFetching(pTrue.op)
          pFalse.op.graph.preventFetching(pFalse.op)
          pivotTrue.op.graph.preventFetching(pivotTrue.op)
          pivotFalse.op.graph.preventFetching(pivotFalse.op)
          predicateId.op.graph.preventFetching(predicateId.op)

          // Build the graph for the true branch in a new context.
          val contextTrue = CondContext(predicateId, pivotTrue, TrueBranch)
          contextTrue.enter()
          val (originalResultTrue, resultTrue) = contextTrue.buildCondBranch(trueFn)
          contextTrue.exitResult(resultTrue)
          contextTrue.exit()

          // Build the graph for the false branch in a new context.
          val contextFalse = CondContext(predicateId, pivotFalse, FalseBranch)
          contextFalse.enter()
          val (_, resultFalse) = contextFalse.buildCondBranch(falseFn)
          contextFalse.exitResult(resultFalse)
          contextFalse.exit()

          // Check that the return values of the two branches have matching data types.
          resultTrue.zip(resultFalse).foreach(pair => {
            if (pair._1.dataType != pair._2.dataType)
              throw InvalidDataTypeException(
                s"The outputs of `trueFn` (dataType = ${pair._1.dataType}) and " +
                    s"`falseFn` (dataType = ${pair._2.dataType}) must have the same data type.")
          })

          // Add to collections.
          Op.currentGraph.addToCollection(contextTrue, CondContext.COND_CONTEXTS)
          Op.currentGraph.addToCollection(contextFalse, CondContext.COND_CONTEXTS)

          // Add the final merge to the graph.
          val merges = resultFalse.zip(resultTrue).map(p => {
            ControlFlow.merge(Seq(p._1, p._2))(TF.fromDataType(p._1.dataType))._1
          })
          ev.unflatten(originalResultTrue, merges)
      }
    }
  }

  /** $OpDocControlFlowCases
    *
    * @group ControlFlowOps
    * @param  predicateFnPairs Contains pairs of predicates and value functions for those predicates.
    * @param  default          Default return value function, in case all predicates evaluate to `false`.
    * @param  exclusive        If `true`, only one of the predicates is allowed to be `true` at the same time.
    * @param  name             Name prefix for the created ops.
    * @return Created op output structure, mirroring the return structure of the provided predicate functions.
    * @throws InvalidDataTypeException If the data types of the tensors returned by the provided predicate functions
    *                                  do not match.
    */
  @throws[InvalidDataTypeException]
  def cases[T, R](
      predicateFnPairs: Seq[(Output[Boolean], () => T)],
      default: () => T,
      exclusive: Boolean = false,
      name: String = "Cases"
  )(implicit
      ev: CondOutput.Aux[T, R]
  ): T = {
    Op.nameScope(name) {
      // To evaluate the conditions in the correct order, we create nested conditions in reverse.
      val fn = predicateFnPairs.reverse.foldLeft(default) {
        case (falseFn, predicateFnPair) => () =>
          cond(
            predicate = predicateFnPair._1,
            trueFn = predicateFnPair._2,
            falseFn = falseFn)
      }
      if (exclusive) {
        Op.createWith(controlDependencies = Set(Checks.assertAtMostNTrue(
          predicateFnPairs.map(_._1),
          n = 1,
          message = "'cases' was created with 'exclusive = true'.").asUntyped)
        ) {
          fn()
        }
      } else {
        fn()
      }
    }
  }

  /** $OpDocControlFlowWhileLoop
    *
    * @group ControlFlowOps
    * @param  predicateFn           Function returning the computation to be performed to determine whether to continue
    *                               looping or terminate.
    * @param  bodyFn                Function returning the computation to be performed in the loop body.
    * @param  loopVariables         Loop variables (possibly a structure over tensors).
    * @param  shapeInvariants       Shape invariants for the loop variables.
    * @param  parallelIterations    Number of iterations allowed to run in parallel.
    * @param  enableBackPropagation If `true`, back-propagation support is enabled for this while-loop context.
    * @param  swapMemory            If `true`, GPU-CPU memory swapping support is enabled for this while-loop context.
    * @param  maximumIterations     Optional `INT32` scalar specifying the maximum number of iterations to loop for. If
    *                               `null` (the default), no iteration limit is enforced.
    * @param  name                  Name prefix for the created ops.
    * @return Created op output structure containing the loop variables values after the loop finishes, mirroring the
    *         return structure of `bodyFn`.
    */
  def whileLoop[T, TS](
      predicateFn: T => Output[Boolean],
      bodyFn: T => T,
      loopVariables: T,
      shapeInvariants: Option[TS] = None,
      parallelIterations: Int = 10,
      enableBackPropagation: Boolean = true,
      swapMemory: Boolean = false,
      maximumIterations: Output[Int] = null,
      name: String = "WhileLoop"
  )(implicit ev: WhileLoopVariable.Aux[T, TS]): T = {
    require(parallelIterations > 0, "'parallelIterations' must be a positive integer.")
    Op.nameScope(name) {
      val loopContext = WhileLoopContext(
        Option(maximumIterations), parallelIterations, enableBackPropagation, swapMemory)
      Op.currentGraph.addToCollection(loopContext, WhileLoopContext.WHILE_LOOP_CONTEXTS)
      if (maximumIterations == null) {
        loopContext.buildLoop(predicateFn, bodyFn, loopVariables, shapeInvariants)
      } else {
        require(maximumIterations.rank == 0 || maximumIterations.rank == -1,
          s"'maximumIterations' must be a scalar, but has shape ${maximumIterations.shape}.")
        val zero = Basic.zeros[Int](Shape())
        val one = Basic.ones[Int](Shape())
        // Building a loop involves mutating ops and thus we need to lock on the graph.
        Op.currentGraph.synchronized {
          loopContext.buildLoop[(Output[Int], T), (Shape, TS)](
            (v: (Output[Int], T)) => Math.logicalAnd(v._1 < maximumIterations, predicateFn(v._2)),
            (v: (Output[Int], T)) => (v._1 + one, bodyFn(v._2)),
            (zero, loopVariables),
            shapeInvariants.map((Shape.scalar(), _)))._2
        }
      }
    }
  }
}

private[api] object ControlFlow extends ControlFlow {
  private[control_flow] trait Implicits {
    implicit class ControlFlowOps(val op: UntypedOp) {
      /** Returns `true` if the provided op is within a cond statement. */
      def isInCond: Boolean = {
        op.controlFlowContext.flatMap(_.condContext).isDefined
      }

      /** Returns `true` if the provided op is within a while loop statement. */
      def isInWhileLoop: Boolean = {
        op.controlFlowContext.flatMap(_.whileLoopContext()).isDefined
      }

      /** Returns `true` if the provided op is within an XLA control flow context. */
      def isInXLAContext: Boolean = {
        val xlaCompile = {
          try {
            op.booleanAttribute("_XlaCompile")
          } catch {
            case _: IllegalArgumentException => false
          }
        }
        xlaCompile || op.controlFlowContext.flatMap(_.xlaContext).isDefined
      }
    }
  }

  /** Returns `true` if and only if the provided op is a switch op. */
  private[ops] def isSwitch(op: Op[_, _]): Boolean = {
    op.opType == "Switch" || op.opType == "RefSwitch"
  }

  /** Returns `true` if and only if the provided op is a merge op. */
  private[ops] def isMerge(op: Op[_, _]): Boolean = {
    op.opType == "Merge" || op.opType == "RefMerge"
  }

  /** Returns `true` if and only if the provided op is a switch op for a conditional. */
  private[ops] def isCondSwitch(op: Op[_, _]): Boolean = {
    if (!isSwitch(op) || op.numOutputs == 0) {
      false
    } else {
      // Switch nodes are not part of the "cond" control flow context that they represent, and so we consider the
      // consumers of its outputs to determine if it is a "cond" switch or not. A switch is a "cond" switch if and only
      // if all its consumers are in "cond" contexts.
      op.outputsSeq.forall(_.consumers.forall(i => {
        var context = i.op.controlFlowContext
        if (isLoopEnter(i.op))
          context = context.flatMap(_.outerContext)
        context.isDefined && context.get.isInstanceOf[CondContext]
      }))
    }
  }

  /** Returns `true` if and only if the provided op is a merge op for a conditional. */
  private[ops] def isCondMerge(op: Op[_, _]): Boolean = {
    if (!isMerge(op) || op.numInputs == 0) {
      false
    } else {
      // Merge nodes are not part of the "cond" control flow context that they represent, and so we consider their
      // inputs to determine if they are "cond" merges or not. A merge is a "cond" merge if and only if all its inputs
      // are in "cond" contexts.
      op.inputsSeq.forall(i => {
        val context = getOutputContext(i.op)
        context.isDefined && context.get.isInstanceOf[CondContext]
      })
    }
  }

  /** Returns `true` if and only if the provided op is a loop invariant. */
  private[ops] def isLoopEnter(op: Op[_, _]): Boolean = {
    op.opType == "Enter" || op.opType == "RefEnter"
  }

  /** Returns `true` if and only if the provided op is a constant loop invariant. */
  private[ops] def isLoopConstantEnter(op: Op[_, _]): Boolean = {
    isLoopEnter(op) && op.booleanAttribute("is_constant")
  }

  /** Returns `true` if and only if the provided op is a loop exit op. */
  private[ops] def isLoopExit(op: Op[_, _]): Boolean = {
    op.opType == "Exit" || op.opType == "RefExit"
  }

  /** Returns `true` if and only if the provided op is a switch op for a while loop. */
  private[ops] def isLoopSwitch(op: Op[_, _]): Boolean = {
    isSwitch(op) &&
        op.controlFlowContext.isDefined &&
        op.controlFlowContext.get.isInstanceOf[WhileLoopContext] &&
        !isCondSwitch(op)
  }

  /** Returns `true` if and only if the provided op is a merge op for a while loop. */
  private[ops] def isLoopMerge(op: Op[_, _]): Boolean = {
    isMerge(op) &&
        op.controlFlowContext.isDefined &&
        op.controlFlowContext.get.isInstanceOf[WhileLoopContext] &&
        !isCondMerge(op)
  }

  /** Returns the enter op if we can infer `value` to be a loop invariant. Otherwise, returns [[None]]. */
  private[control_flow] def getLoopConstantEnter(value: Output[_]): Option[UntypedOp] = {
    val identityOpTypes = Set("Identity", "RefIdentity", "Switch", "RefSwitch")
    var op = value.op
    while (identityOpTypes.contains(op.opType))
      op = op.inputsSeq(0).op
    Some(op).filter(isLoopConstantEnter)
  }

  /** Returns the control flow context for the outputs of an op. */
  private[ops] def getOutputContext(op: Op[_, _]): Option[Context] = {
    val context = op.controlFlowContext
    if (isLoopExit(op))
      context.flatMap(_.outerContext)
    else
      context
  }

  /** Returns `true` if `maybeContainingContext` is or contains `context`. */
  private[ops] def isContainingContext(
      context: Context,
      maybeContainingContext: Option[Context]
  ): Boolean = {
    if (maybeContainingContext.isEmpty && context == null) {
      true
    } else {
      maybeContainingContext.exists(containingContext => {
        var currentContext = Option(context)
        while (currentContext.exists(_ != containingContext))
          currentContext = currentContext.flatMap(_.outerContext)
        currentContext.contains(containingContext)
      })
    }
  }

  /** Checks whether `inputOp` can be used from within the `op`'s context. Conceptually, only inputs from an op's while
    * loop context or any ancestor while loop context (including outside of any context) are valid. In practice, there
    * are many other edge cases as well. */
  @throws[InvalidArgumentException]
  private[ops] def checkInputFromValidContext(op: Op[_, _], inputOp: Op[_, _]): Unit = {
    val opContext = op.controlFlowContext
    val inputContext = getOutputContext(inputOp)
    val errorMessage = inputContext match {
      case None => null                            // `inputOp` is not in a control flow context.
      case Some(context) if context == opContext.orNull => null // `inputOp` is in the same control flow context.
      case Some(context) =>
        val whileContext = opContext.flatMap(_.whileLoopContext())
        val inputWhileContext = context.whileLoopContext()
        whileContext match {
          case None =>
            if (inputWhileContext.isEmpty) {
              // Neither `op` nor `inputOp` is in a while loop, but one or both are in conditionals. We allow this,
              // although execution will fail if the branch corresponding to the `inputOp`'s conditional context is not
              // taken.
              null
            } else if (isLoopEnter(op) || isSwitch(op)) {
              // The while loop building code clears the context for enter nodes, and the conditional context add value
              // code clears the context for switch nodes.
              null
            } else {
              s"Cannot use '${inputOp.name}' as input to '${op.name}' because '${inputOp.name}' is in a while loop."
            }
          case Some(whileLoopContext) if isContainingContext(whileLoopContext, inputWhileContext) =>
            // `inputOp` is in a while loop which contains `op`'s while loop (or not in a while loop at all).
            null
          case Some(whileLoopContext) if whileLoopContext.gradientLoopState.isDefined &&
              isContainingContext(whileLoopContext.gradientLoopState.get.forwardContext, inputWhileContext) =>
            // `op` is in a gradient context and `inputOp` is in the associated forward pass context or an ancestor
            // thereof. This case is needed to build while loop gradients. Note that we theoretically also need this
            // case for custom gradient functions that close over tensors from ancestor contexts, but this has not been
            // verified yet.
            null
          case Some(whileLoopContext) if whileLoopContext.gradientLoopState.isDefined &&
              whileLoopContext.gradientLoopState.get.forwardContext ==
                  inputWhileContext.flatMap(_.outerContext).orNull =>
            // `op` is in a gradient context and `inputOp` is in a child of the associated forward pass context. This
            // case is needed for the gradients of while loops with conditionals.
            null
          case Some(whileLoopContext) if inputWhileContext.flatMap(_.gradientLoopState).isDefined &&
              inputWhileContext.flatMap(_.gradientLoopState).get.forwardContext == whileLoopContext =>
            // `inputOp` is in the gradient context of `op`'s context. This case is needed when the gradient of a while
            // loop gradient is requested (this will eventually fail unless there is a `stopGradient` op or similar).
            null
          case Some(whileLoopContext) if inputWhileContext.flatMap(_.gradientLoopState).isDefined &&
              context.gradientLoopState.flatMap(_.forwardContext.gradientLoopState).isDefined &&
              context.gradientLoopState
                  .flatMap(_.forwardContext.gradientLoopState).get.forwardContext == whileLoopContext =>
            // `inputOp` is in the gradient gradient context of `op`'s context. This case is needed when the gradient of
            // a while loop gradient is requested (this will eventually fail unless there is a `stopGradient` op or
            // similar).
            null
          case _ =>
            s"Cannot use '${inputOp.name}' as input to '${op.name}' because they are in different while loops."
        }
    }
    if (errorMessage != null)
      throw InvalidArgumentException(errorMessage)
  }

  /** Calculates a maximum size for use by stack ops inside XLA while loops.
    *
    * @param  value            Value inside the while loop forward context. Used for printing error messages.
    * @param  whileLoopContext Forward context inside which value resides. This does not always match the value's
    *                          immediate context, as `value` may be inside e.g., a cond context, inside the while loop.
    * @return Tensor containing the `maxSize` to feed to a stack initializer.
    * @throws InvalidArgumentException If `value` is nested inside a while loop that either lacks a `maximumIterations`
    *                                  parameter, or whose `maximumIterations` parameter is inside a while loop that is
    *                                  a parent of the calling context, and cannot be evaluated at graph build time
    *                                  (i.e., statically) to a constant value.
    */
  @throws[InvalidArgumentException]
  private[control_flow] def getMaxSizeFromNestedMaximumIterations(
      value: Output[_],
      whileLoopContext: WhileLoopContext
  ): Output[Int] = {
    val valueName = value.name
    // `currentContext` is the context that `tf.gradients()` was called in.
    val currentContext = Op.currentControlFlowContext
    val currentContextName = currentContext.map(_.name).getOrElse("")

    // Loop through all containing while-loop contexts between the value and the current context, multiplying together
    // each context's `maxIterations`, in order to get the maximum stack size.
    var maxSize = Basic.constant(1)
    var currentWhileLoopContext: Option[WhileLoopContext] = Some(whileLoopContext)
    while (currentWhileLoopContext.isDefined) {
      currentWhileLoopContext.get.maximumIterations match {
        case None => throw InvalidArgumentException(
          s"Cannot create a gradient accumulator for tensor '$valueName', inside an XLA while loop, because " +
              "'maximumIterations' was not passed to the `tf.whileLoop()` call " +
              s"('${currentWhileLoopContext.get.name}').")
        case Some(maximumIterations) =>
          val maximumIterationsContext = maximumIterations.op.controlFlowContext
          // If `maximumIterationsContext` (non-strictly) contains `currentContext`, then it is ok to use.
          if (isContainingContext(currentContext.orNull, maximumIterationsContext)) {
            maxSize *= maximumIterations
          } else {
            // We cannot use `maximumIterations` because it is defined in a nested while-loop or cond context, and so
            // an error will be thrown if we try to use it as input to any ops in `currentContext` (e.g., `maxSize` or
            // the final accumulator stack). We attempt to get a constant value out to use instead.
            Output.constantValue(maximumIterations) match {
              case Some(constantMaximumIterations) => maxSize *= constantMaximumIterations
              case None => throw InvalidArgumentException(
                s"Cannot create a gradient accumulator for tensor '$valueName', inside an XLA while loop, because " +
                    s"the 'maximumIterations' tensor ('${maximumIterations.name}') for while-loop context " +
                    s"'${currentWhileLoopContext.get.name}' must be statically known (e.g., a constant value or " +
                    "known shape dimension), or must be defined at or outside the while-loop context " +
                    s"'$currentContextName' (currently defined in '${maximumIterationsContext.get.name}').")
            }
          }
      }
      // Find the next outer while-loop context, or stop if we have reached the `tf.gradients()` context.
      currentWhileLoopContext = currentWhileLoopContext
          .flatMap(_.outerContext.flatMap(_.whileLoopContext(currentContext)))
    }

    maxSize
  }

  /** Creates an op that forwards `input` to the output port determined by `predicate`, while making sure the new op is
    * colocated with `input`.
    *
    * If `predicate` is `true`, then `input` is forwarded to `outputTrue`. Otherwise, it goes to `outputFalse`.
    *
    * @param  input     Tensor to be forwarded to the appropriate output.
    * @param  predicate Scalar boolean tensor that specifies which output port will receive `input`.
    * @param  name      Name for the created op.
    * @return Tuple containing `outputFalse` and `outputTrue`, in that order.
    */
  private[control_flow] def colocatedSwitch[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      predicate: Output[Boolean],
      name: String = "Switch"
  ): (OL[T], OL[T]) = {
    // The device colocation below addresses the following scenario:
    //
    // Assume you execute Optimizer.applyGradients() in a branch of a cond() and:
    //   1. The update op is created inside a `Op.colocateWith(Set(var.op)) { }` block.
    //   2. Some tensor `data` is captured and a switch is created in a `Op.colocateWith(Set(data.op)) { }` block.
    //
    // Op.colocateWith(Set(var.op)) {
    //   Op.colocateWith(Set(data.op)) {
    //     op = ...
    //   }
    // }
    //
    // `var` and `data` may be pinned to different devices and so we want the ops created within the
    // `Op.colocateWith(Set(data.op)) { }` block to ignore the existing stack.
    Op.colocateWith(Set(input.op), ignoreExisting = true) {
      switch(input, predicate, name)
    }
  }

  /** Returns an `assert` op that checks that the provided predicates are exclusive (i.e., not more than one of them can
    * be `true` at the same time). */
  private[ControlFlow] def assertExclusive(
      predicates: Seq[Output[Boolean]]
  ): Op[Seq[Output[Any]], Unit] = {
    val stacked = Basic.stack(predicates, name = "StackedPredicates")
    val numTrue = Math.sum(stacked.castTo[Int], name = "NumTruePredicates")
    val atMostOneTrue = Math.less(numTrue, Basic.constant(2, name = "TwoTruePredicates"))
    val errorData =
      Seq(
        Basic.constant(Tensor(
          "More than one condition evaluated as 'true' but 'exclusive = true'. " +
              s"Conditions: (${predicates.map(_.name).mkString(", ")}), Values: ")),
        stacked)
    Checks.assert(atMostOneTrue, errorData, summarize = predicates.size)
  }

  //region Low Level Ops

  /** Creates an op that does nothing and serves as a control trigger for scheduling. The created op is only useful as
    * a placeholder for control edges.
    *
    * @param  name Name for the created op.
    * @return Created op output.
    */
  private[control_flow] def controlTrigger(
      name: String = "ControlTrigger"
  ): Op[Unit, Unit] = {
    Op.Builder[Unit, Unit](
      opType = "ControlTrigger",
      name = name,
      input = ()
    ).build()
  }

  /** Creates an op that forwards its input to the output.
    *
    * The op represents the loop termination condition used by the "pivot" switches of a loop.
    *
    * @param  input Boolean scalar tensor, representing the branch predicate of the switch op.
    * @param  name  Name for the created op.
    * @return Created op output, which has the same value as the input tensor.
    */
  private[control_flow] def loopCond(
      input: Output[Boolean],
      name: String = "LoopCond"
  ): Output[Boolean] = {
    Op.Builder[Output[Boolean], Output[Boolean]](
      opType = "LoopCond",
      name = name,
      input = input
    ).build().output
  }

  /** Creates an op that makes its input available to the next iteration.
    *
    * @param  input Tensor to make available to the next iteration.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[control_flow] def nextIteration[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "NextIteration"
  ): OL[T] = {
    val result = {
      input match {
        case o: Output[T] =>
          Op.Builder[Output[T], Output[T]](
            opType = "NextIteration",
            name = name,
            input = o
          ).setGradientFn(nextIterationGradient(_, _)(TF[T]))
              .build().output
        case o: OutputIndexedSlices[T] => Op.nameScope(name) {
          val values = nextIteration(o.values, "Values")
          val indices = nextIteration(o.indices, "Indices")
          val denseShape = {
            if (o.denseShape != null)
              nextIteration(o.denseShape, "DenseShape")
            else
              null
          }
          OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        }
        case o: SparseOutput[T] => Op.nameScope(name) {
          val values = nextIteration(o.values, "Values")
          val indices = nextIteration(o.indices, "Indices")
          val denseShape = nextIteration(o.denseShape, "DenseShape")
          SparseOutput(indices = indices, values = values, denseShape = denseShape)
        }
      }
    }
    result.asInstanceOf[OL[T]]
  }

  /** A forward next-iteration op is translated into a back-propagation identity op. Note that the back-propagation
    * next-iteration op is added in switch op gradient. */
  protected def nextIterationGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: OutputLike[T]
  ): OutputLike[T] = {
    outputGradient
  }

  /** Creates an op that creates or finds a child frame, and makes `input` available to that child frame.
    *
    * The op is used together with `exit` to create loops in the graph. The unique `frameName` is used by the `Executor`
    * to identify frames. If `isConstant` is `true`, then the output is a constant in the child frame. Otherwise, it may
    * be changed in the child frame. At most `parallelIterations` iterations are run in parallel in the child frame.
    *
    * @param  input              Tensor to be made available to the child frame.
    * @param  frameName          Name of the child frame.
    * @param  isConstant         If `true`, the output is constant within the child frame.
    * @param  parallelIterations Number of iterations allowed to run in parallel.
    * @param  useInputShape      If `true`, the output tensor's shape is manually set to the input tensor's shape.
    * @param  name               Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[control_flow] def enter[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      frameName: String,
      isConstant: Boolean = false,
      parallelIterations: Int = 10,
      useInputShape: Boolean = true,
      name: String = "Enter"
  ): OL[T] = {
    val result = {
      input match {
        case o: Output[T] =>
          val result = Op.Builder[Output[T], Output[T]](
            opType = "Enter",
            name = name,
            input = o
          ).setAttribute("frame_name", frameName)
              .setAttribute("is_constant", isConstant)
              .setAttribute("parallel_iterations", parallelIterations)
              .setGradientFn(enterGradient(_, _)(TF[T]))
              .build().output.toOutput
          if (useInputShape)
            result.setShape(o.shape)
          result
        case o: OutputIndexedSlices[T] => Op.nameScope(name) {
          val values = enter(o.values, frameName, isConstant, parallelIterations, useInputShape, "Values")
          val indices = enter(o.indices, frameName, isConstant, parallelIterations, useInputShape, "Indices")
          val denseShape = {
            if (o.denseShape != null)
              enter(o.denseShape, frameName, isConstant, parallelIterations, useInputShape, "DenseShape")
            else
              null
          }
          OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        }
        case o: SparseOutput[T] => Op.nameScope(name) {
          val values = enter(o.values, frameName, isConstant, parallelIterations, useInputShape, "Values")
          val indices = enter(o.indices, frameName, isConstant, parallelIterations, useInputShape, "Indices")
          val denseShape = enter(
            o.denseShape, frameName, isConstant, parallelIterations, useInputShape, "DenseShape")
          SparseOutput(indices = indices, values = values, denseShape = denseShape)
        }
      }
    }
    result.asInstanceOf[OL[T]]
  }

  /** Gradients for an enter op are calculated using an exit op. For loop variables, `outputGradients` is the gradient
    * and so we just add an exit op. For loop invariants, we need to add an accumulator loop. */
  protected def enterGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: OutputLike[T]
  ): OutputLike[T] = {
    Op.currentControlFlowContext.map(gradientContext => {
      if (!gradientContext.backPropagate) {
        // We skip gradient computation in this case.
        outputGradient
      } else if (gradientContext.gradientLoopState.isEmpty) {
        // We pass the gradient through if we are not in a gradient while-loop context.
        outputGradient
      } else if (op.booleanAttribute("is_constant")) {
        // We add a gradient accumulator for each while-loop invariant.
        gradientContext.asInstanceOf[WhileLoopContext].addBackwardAccumulator(op, outputGradient)
      } else {
        val gradientWhileLoopContext = gradientContext.asInstanceOf[WhileLoopContext]
        val result = exit(outputGradient)
        result match {
          case o: Output[T] =>
            gradientWhileLoopContext.loopExits += o
          case o: OutputIndexedSlices[T] =>
            gradientWhileLoopContext.loopExits += o.indices
            gradientWhileLoopContext.loopExits += o.values
            if (o.denseShape != null)
              gradientWhileLoopContext.loopExits += o.denseShape
          case o: SparseOutput[T] =>
            gradientWhileLoopContext.loopExits += o.indices
            gradientWhileLoopContext.loopExits += o.values
            if (o.denseShape != null)
              gradientWhileLoopContext.loopExits += o.denseShape
        }
        gradientContext.exitResult(Seq(result))
        result
      }
    }).get
  }

  /** Creates an op that exits from the current frame to its parent frame.
    *
    * The op makes `input` available to the parent frame.
    *
    * @param  input Tensor to be made available to the parent frame.
    * @param  name  Name for the created op.
    * @return Created op output, which is the same as `input`.
    */
  private[control_flow] def exit[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      name: String = "Exit"
  ): OL[T] = {
    val result = {
      input match {
        case o: Output[T] =>
          Op.Builder[Output[T], Output[T]](
            opType = "Exit",
            name = name,
            input = o
          ).setGradientFn(exitGradient(_, _)(TF[T]))
              .build().output
        case o: OutputIndexedSlices[T] => Op.nameScope(name) {
          val values = exit(o.values, "Values")
          val indices = exit(o.indices, "Indices")
          val denseShape = {
            if (o.denseShape != null)
              exit(o.denseShape, "DenseShape")
            else
              null
          }
          OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
        }
        case o: SparseOutput[T] => Op.nameScope(name) {
          val values = exit(o.values, "Values")
          val indices = exit(o.indices, "Indices")
          val denseShape = {
            if (o.denseShape != null)
              exit(o.denseShape, "DenseShape")
            else
              null
          }
          SparseOutput(indices = indices, values = values, denseShape = denseShape)
        }
      }
    }
    result.asInstanceOf[OL[T]]
  }

  /** Gradients for an exit op are calculated using an enter op. */
  @throws[UnimplementedException]
  protected def exitGradient[T: TF](
      op: Op[Output[T], Output[T]],
      outputGradient: OutputLike[T]
  ): OutputLike[T] = {
    Op.currentControlFlowContext.map(gradientContext => {
      if (!gradientContext.backPropagate) {
        // We skip gradient computation in this case.
        null
      } else if (op.controlFlowContext.flatMap(_.gradientLoopState).isDefined) {
        throw UnimplementedException("Second-order gradients are not supported for while loops.")
      } else {
        outputGradient match {
          case o: Output[T] =>
            gradientContext.values += o.name
          case o: OutputIndexedSlices[T] =>
            gradientContext.values += o.indices.name
            gradientContext.values += o.values.name
            if (o.denseShape != null)
              gradientContext.values += o.denseShape.name
          case o: SparseOutput[T] =>
            gradientContext.values += o.indices.name
            gradientContext.values += o.values.name
            if (o.denseShape != null)
              gradientContext.values += o.denseShape.name
        }
        val gradientWhileLoopContext = gradientContext.asInstanceOf[WhileLoopContext]
        gradientContext.enter()
        val result = enter(
          outputGradient,
          gradientContext.name,
          isConstant = false,
          parallelIterations = gradientWhileLoopContext.parallelIterations,
          name = "ExitGradient")
        result match {
          case o: Output[T] =>
            gradientWhileLoopContext.loopEnters += o
          case o: OutputIndexedSlices[T] =>
            gradientWhileLoopContext.loopEnters += o.indices
            gradientWhileLoopContext.loopEnters += o.values
            if (o.denseShape != null)
              gradientWhileLoopContext.loopEnters += o.denseShape
          case o: SparseOutput[T] =>
            gradientWhileLoopContext.loopEnters += o.indices
            gradientWhileLoopContext.loopEnters += o.values
            if (o.denseShape != null)
              gradientWhileLoopContext.loopEnters += o.denseShape
        }
        gradientContext.exit()
        result
      }
    }).get
  }

  /** Creates an op that forwards `input` to the output port determined by `predicate`.
    *
    * If `predicate` is `true`, then `input` is forwarded to `outputTrue`. Otherwise, it goes to `outputFalse`.
    *
    * @param  input     Tensor to be forwarded to the appropriate output.
    * @param  predicate Scalar boolean tensor that specifies which output port will receive `input`.
    * @param  name      Name for the created op.
    * @return Tuple containing `outputFalse` and `outputTrue`, in that order.
    */
  private[control_flow] def switch[T: TF, OL[A] <: OutputLike[A]](
      input: OL[T],
      predicate: Output[Boolean],
      name: String = "Switch"
  ): (OL[T], OL[T]) = {
    val result = {
      input match {
        case o: Output[T] =>
          Op.Builder[(Output[T], Output[Boolean]), (Output[T], Output[T])](
            opType = "Switch",
            name = name,
            input = (o, predicate)
          ).setGradientFn[(OutputLike[T], Output[Boolean]), (OutputLike[T], OutputLike[T])](switchGradient(_, _)(TF[T]))
              .build().output
        case o: OutputIndexedSlices[T] =>
          Op.nameScope(name) {
            val (valuesFalse, valuesTrue) = switch(o.values, predicate, "Values")
            val (indicesFalse, indicesTrue) = switch(o.indices, predicate, "Indices")
            val (denseShapeFalse, denseShapeTrue) = {
              if (o.denseShape != null)
                switch(o.denseShape, predicate, "DenseShape")
              else
                (null, null)
            }
            (OutputIndexedSlices(indices = indicesFalse, values = valuesFalse, denseShape = denseShapeFalse),
                OutputIndexedSlices(indices = indicesTrue, values = valuesTrue, denseShape = denseShapeTrue))
          }
        case o: SparseOutput[T] =>
          Op.nameScope(name) {
            val (valuesFalse, valuesTrue) = switch(o.values, predicate, "ValuesSwitch")
            val (indicesFalse, indicesTrue) = switch(o.indices, predicate, "IndicesSwitch")
            val (denseShapeFalse, denseShapeTrue) = {
              if (o.denseShape != null)
                switch(o.denseShape, predicate, "DenseShape")
              else
                (null, null)
            }
            (SparseOutput(indices = indicesFalse, values = valuesFalse, denseShape = denseShapeFalse),
                SparseOutput(indices = indicesTrue, values = valuesTrue, denseShape = denseShapeTrue))
          }
      }
    }
    result.asInstanceOf[(OL[T], OL[T])]
  }

  /** Gradients for a switch op are calculated using a merge op. If the switch is a loop switch, it will be visited
    * twice. We create the merge op on the first visit, and we update the second input of the merge on the second
    * visit. A next-iteration op is also added in the second visit. */
  protected def switchGradient[T: TF](
      op: Op[(Output[T], Output[Boolean]), (Output[T], Output[T])],
      outputGradient: (OutputLike[T], OutputLike[T])
  ): (OutputLike[T], Output[Boolean]) = {
    val gradientContext = Op.currentControlFlowContext
    op.controlFlowContext match {
      case Some(opContext: CondContext) =>
        val (gradientOtherBranch, gradientTakenBranch) = opContext.branch match {
          case TrueBranch => (outputGradient._1, outputGradient._2)
          case FalseBranch => (outputGradient._2, outputGradient._1)
        }

        if (gradientOtherBranch != null) {
          (merge(Seq(outputGradient._1, outputGradient._2), name = "CondGradient")._1, null)
        } else if (op.input._1.dataType == RESOURCE) {
          // At this point, we have created `zeroGradient` guarded by the right switch. Unfortunately, we may still
          // get `null` here for non-trainable data types or for some types of ops (e.g., `ResourceGather`) created
          // within only one branch.
          // TODO: !!! This may be inefficient. What if one branch of the switch is not differentiable?
          val zeros = gradientTakenBranch match {
            case o: Output[_] => Basic.zerosLike(o)(TF.fromDataType(o.dataType))
            case o: OutputIndexedSlices[_] =>
              OutputIndexedSlices(
                Basic.zeros[Long](Shape(1)),
                Basic.zeros(o.values.dataType, Shape(1, o.values.shape(1))),
                o.denseShape)
            case o: SparseOutput[_] =>
              SparseOutput(
                Basic.zeros[Long](Shape(1, o.indices.shape(1))),
                Basic.zeros(o.values.dataType, Shape(1)),
                o.denseShape)
          }
          val zeroGradient = opContext.branch.other.selectSwitchResult(
            ControlFlow.colocatedSwitch(zeros, opContext.predicate)(TF.fromDataType(zeros.dataType)))
          if (opContext.branch.value == 0) {
            (merge(Seq(
              gradientTakenBranch,
              zeroGradient.asInstanceOf[OutputLike[T]]
            ), name = "CondGradient")(TF.fromDataType(gradientTakenBranch.dataType))._1, null)
          } else {
            (merge(Seq(
              zeroGradient.asInstanceOf[OutputLike[T]],
              gradientTakenBranch
            ), name = "CondGradient")(TF.fromDataType(gradientTakenBranch.dataType))._1, null)
          }
        } else {
          (null, null)
        }
      case Some(_: WhileLoopContext) =>
        gradientContext.flatMap(_.gradientLoopState).flatMap(_.switchMap.get(op.asUntyped)) match {
          case Some(mergeGradient) =>
            // This is the second time this switch node is visited. It comes from the non-exit branch of the switch,
            // and so we update the second input to the merge node.
            if (outputGradient._2 != null) {
              WhileLoopContext.addNextIterationAndBackEdge(
                mergeGradient.asInstanceOf[OutputLike[T]],
                outputGradient._2,
                enforceShapeInvariant = false)
            }
            (null, null)
          case None if outputGradient._1 != null =>
            // This is the first time this switch node is visited. It comes from the exit branch of the switch, which
            // is `outputGradients(0)`. `outputGradients(1)` is empty at this point. We use `outputGradients(0)` for
            // both inputs to the merge for now, but we update the second input of the merge node when we visit this
            // switch node for a second time.
            val mergeGradient = merge(Seq(outputGradient._1, outputGradient._1), name = "SwitchGradient")._1
            gradientContext
                .flatMap(_.gradientLoopState)
                .map(_.switchMap)
                .foreach(_ += op.asUntyped -> mergeGradient)
            (mergeGradient, null)
          case _ =>
            // This is the first time this switch node is visited. It comes from the identity branch. Such a switch
            // has `null` gradient for the exit branch, meaning that the output is not differentiable.
            (null, null)
        }
      case _ =>
        val falseGradient = switch(outputGradient._1, op.input._2)._1
        val trueGradient = switch(outputGradient._2, op.input._2)._2
        (merge(Seq(falseGradient, trueGradient))._1, null)
    }
  }

  /** Creates an op that forwards the value of an available tensor from `inputs` to `output`.
    *
    * The op tests each of the tensors in `inputs` in turn to determine if any of them is available. If it finds an
    * available tensor, it returns it and its index, `outputIndex`, in `inputs`.
    *
    * No more than one tensor in `inputs` should be available. If no tensor in `inputs` is available, the returned
    * tensor and index are not set.
    *
    * This op is usually combined with `switch` to implement branching.
    *
    * IMPORTANT NOTE: The input tensors can either all be of type [[Output]] or [[SparseOutput]] or of mixed types that
    * extend [[OutputLike]]. If they are all of type [[Output]] or [[SparseOutput]], then that is also the return op
    * type. Otherwise, they will all be converted to [[OutputIndexedSlices]] first.
    *
    * @param  inputs Input tensors.
    * @param  name   Name for the created op.
    * @return Tuple containing `output` and `outputIndex`, in that order.
    */
  @throws[IllegalArgumentException]
  private[control_flow] def merge[T: TF, OL[A] <: OutputLike[A]](
      inputs: Seq[OL[T]],
      name: String = "Merge"
  ): (OL[T], Output[Int]) = {
    val result = {
      inputs match {
        case o if o.forall(_.isInstanceOf[Output[T]]) =>
          Op.Builder[Seq[Output[T]], (Output[T], Output[Int])](
            opType = "Merge",
            name = name,
            input = o.map(_.asInstanceOf[Output[T]])
          ).setGradientFn(mergeGradient(_, _)(TF[T]))
              .build().output
        case o if o.forall(_.isInstanceOf[SparseOutput[T]]) =>
          Op.nameScope(name) {
            val oo = o.map(_.asInstanceOf[SparseOutput[T]])
            val (indices, chosenIndex) = merge(oo.map(_.indices), "Indices")
            val (values, _) = merge(oo.map(_.values), "Values")
            val (denseShape, _) = if (oo.map(_.denseShape).exists(_ != null)) {
              if (oo.map(_.denseShape).contains(null))
                throw new IllegalArgumentException(
                  "Either all merged 'SparseOutput's must have a known dense shape, or none of them.")
              merge(oo.map(_.denseShape), "DenseShape")
            } else {
              null
            }
            (SparseOutput(indices = indices, values = values, denseShape = denseShape), chosenIndex)
          }
        case o =>
          Op.nameScope(name) {
            val oo = o.map(_.toOutputIndexedSlices(optimize = false))
            val (indices, chosenIndex) = merge(oo.map(_.indices), "Indices")
            val (values, _) = merge(oo.map(_.values), "Values")
            val (denseShape, _) = if (oo.map(_.denseShape).exists(_ != null)) {
              if (oo.map(_.denseShape).contains(null))
                throw new IllegalArgumentException(
                  "Either all merged 'OutputIndexedSlices' must have a known dense shape, or none of them.")
              merge(oo.map(_.denseShape), "DenseShape")
            } else {
              null
            }
            (OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape), chosenIndex)
          }
      }
    }
    result.asInstanceOf[(OL[T], Output[Int])]
  }

  /** Gradients for a merge op are calculated using a switch op. */
  protected def mergeGradient[T: TF](
      op: Op[Seq[Output[T]], (Output[T], Output[Int])],
      outputGradient: (Output[T], Output[Int])
  ): Seq[Output[T]] = {
    val gradientContext = Op.currentControlFlowContext
    ControlFlow.getOutputContext(op.input.head.op) match {
      case Some(opContext: CondContext) =>
        val predicate = gradientContext.flatMap(_.gradientLoopState).map(gradientLoopState => {
          // This merge node is part of a conditional structure within a loop. The back-propagation needs to have the
          // value of this predicate for every iteration and so, we must have its values accumulated in the forward,
          // and use the accumulated values as the predicate for this back-propagation switch.
          gradientLoopState.historyMap.getOrElse(opContext.predicate.name, {
            // We want to remember the value of the predicate for every iteration.
            gradientLoopState.backwardContext.exit()
            val historyPredicate = gradientLoopState.addForwardAccumulator(opContext.predicate)
            gradientLoopState.backwardContext.enter()
            // We now add the stack pop op. If `opContext.predicate.op` is in a (possibly outer) `CondContext`, then
            // the stack pop op will be guarded with a switch.
            val realPredicate = gradientLoopState.addBackwardAccumulatedValue(historyPredicate, opContext.predicate)
            gradientLoopState.historyMap += opContext.predicate.name -> realPredicate
            realPredicate
          })
        }).getOrElse(opContext.predicate).asInstanceOf[Output[Boolean]]
        val switch = colocatedSwitch(outputGradient._1, predicate)
        Seq(switch._1, switch._2)
      case Some(_: WhileLoopContext) =>
        val switch = colocatedSwitch(outputGradient._1, gradientContext.get.asInstanceOf[WhileLoopContext].pivot)
        Seq(switch._1, switch._2)
      case _ =>
        (0 until op.numInputs).map(i => {
          colocatedSwitch(outputGradient._1, Math.equal(op.output._2, i))._2
        })
    }
  }

  //endregion Low Level Ops

  //region Native Library Functions

  /** Replaces the `index`th input of `op` with `newInput`. */
  private[control_flow] def updateInput(op: Op[_, _], index: Int, newInput: Output[_]): Unit = {
    using(op.graph.reference)(r => {
      NativeLibrary.updateInput(r.nativeHandle, op.nativeHandle, index, newInput.op.nativeHandle, newInput.index)
    })
    op._reloadNumInputs()
    op._reloadInputs()
  }

  /** Adds `inputOp` as a control input of `op`. */
  private[control_flow] def addControlInput(op: Op[_, _], inputOp: Op[_, _]): Unit = {
    using(op.graph.reference)(r => {
      NativeLibrary.addControlInput(r.nativeHandle, op.nativeHandle, inputOp.nativeHandle)
    })
    op._reloadNumControlInputs()
    op._reloadControlInputs()
  }

  /** Clears the control inputs of `op` (i.e., removes all of them). */
  private[control_flow] def clearControlInputs(op: Op[_, _]): Unit = {
    using(op.graph.reference)(r => {
      NativeLibrary.clearControlInputs(r.nativeHandle, op.nativeHandle)
    })
    op._reloadNumControlInputs()
    op._reloadControlInputs()
  }

  /** Sets attribute `name` of `op` to the provided value. */
  private[control_flow] def setAttribute(op: Op[_, _], name: String, value: AttrValue): Unit = {
    using(op.graph.reference)(r => {
      NativeLibrary.setAttributeProto(r.nativeHandle, op.nativeHandle, name, value.toByteArray)
    })
  }

  //endregion Native Library Functions

  /** @define OpDocControlFlowGroup
    *   The `group` op groups multiple ops together.
    *
    *   When the op finishes, all ops in `inputs` have finished. The op has no output.
    *
    * @define OpDocControlFlowTuple
    *   The `tuple` op groups op outputs together.
    *
    *   The op creates a tuple of op outputs with the same values as `inputs`, except that the value of each output is
    *   only returned after the values of all outputs in `inputs` have been computed.
    *
    *   This op can be used as a "join" mechanism for parallel computations: all the argument tensors can be computed in
    *   parallel, but the values of any tensor returned by `tuple` are only available after all the parallel
    *   computations are done.
    *
    * @define OpDocControlFlowNoOp
    *   The `noOp` op does nothing. The created op is only useful as a placeholder for control edges.
    *
    * @define OpDocControlFlowCond
    *   The `cond` op returns `trueFn()` if the predicate `predicate` is true, else `falseFn()`.
    *
    *   `trueFn` and `falseFn` both return structures of tensors (e.g., lists of tensors). `trueFn` and `falseFn` must
    *   have the same non-zero number and type of outputs. Note that the conditional execution applies only to the ops
    *   defined in `trueFn` and `falseFn`.
    *
    *   For example, consider the following simple program:
    *   {{{
    *     val z = tf.multiply(a, b)
    *     val result = tf.cond(x < y, () => tf.add(x, z), () => tf.square(y))
    *   }}}
    *   If `x < y`, the `tf.add` operation will be executed and the `tf.square` operation will not be executed. Since
    *   `z` is needed for at least one branch of the `cond`, the `tf.multiply` operation is always executed,
    *   unconditionally. Although this behavior is consistent with the data-flow model of TensorFlow, it has
    *   occasionally surprised some users who expected lazier semantics.
    *
    *   Note that `cond` calls `trueFn` and `falseFn` *exactly once* (inside the call to `cond`, and not at all during
    *   `Session.run()`). `cond` stitches together the graph fragments created during the `trueFn` and `falseFn` calls
    *   with some additional graph nodes to ensure that the right branch gets executed depending on the value of
    *   `predicate`.
    *
    *   `cond` supports nested tensor structures, similar to `Session.run()`. Both `trueFn` and `falseFn` must return
    *   the same (possibly nested) value structure of sequences, tuples, and/or maps.
    *
    *   '''NOTE:''' If the predicate always evaluates to some constant value and that can be inferred statically, then
    *   only the corresponding branch is built and no control flow ops are added. In some cases, this can significantly
    *   improve performance.
    *
    * @define OpDocControlFlowCases
    *   The `cases` op creates a case operation.
    *
    *   The `predicateFnPairs` parameter is a sequence of pairs. Each pair contains a boolean scalar tensor and a
    *   function that takes no parameters and creates the tensors to be returned if the boolean evaluates to `true`.
    *   `default` is a function that returns the default value, used when all provided predicates evaluate to `false`.
    *
    *   All functions in `predicateFnPairs` as well as `default` (if provided) should return the same structure of
    *   tensors, and with matching data types. If `exclusive == true`, all predicates are evaluated, and an exception is
    *   thrown if more than one of the predicates evaluates to `true`. If `exclusive == false`, execution stops at the
    *   first predicate which evaluates to `true`, and the tensors generated by the corresponding function are returned
    *   immediately. If none of the predicates evaluate to `true`, the operation returns the tensors generated by
    *   `default`.
    *
    *   Example 1:
    *   {{{
    *     // r = if (x < y) 17 else 23.
    *     val r = tf.cases(
    *       Seq(x < y -> () => tf.constant(17)),
    *       default = () => tf.constant(23))
    *   }}}
    *
    *   Example 2:
    *   {{{
    *     // if (x < y && x > z) throw error.
    *     // r = if (x < y) 17 else if (x > z) 23 else -1.
    *     val r = tf.cases(
    *       Seq(x < y -> () => tf.constant(17), x > z -> tf.constant(23)),
    *       default = () => tf.constant(-1),
    *       exclusive = true)
    *   }}}
    *
    * @define OpDocControlFlowWhileLoop
    *   The `whileLoop` op repeats the result of `bodyFn` while the condition returned by `predicateFn` is `true`.
    *
    *   `predicateFn` is a function returning a `BOOLEAN` scalar tensor. `bodyFn` is a function returning a structure
    *   over tensors mirroring that of `loopVariables`. `loopVariables` is a structure over tensors that is passed to
    *   both `predicateFn` and `bodyFn`. `predicateFn` and `bodyFn` both take as many arguments as there are
    *   `loopVariables`.
    *
    *   In addition to regular tensors, indexed slices, or sparse tensors, the body function may accept and return
    *   tensor array objects. The flows of the tensor array objects will be appropriately forwarded between loops and
    *   during gradient calculations.
    *
    *   Note that `whileLoop()` calls `predicateFn` and `bodyFn` *exactly once* (inside the call to `whileLoop`, and not
    *   at all during `Session.run()`). `whileLoop()` stitches together the graph fragments created during the
    *   `predicateFn` and `bodyFn` calls with some additional graph nodes to create the graph flow that repeats `bodyFn`
    *   until `predicateFn` returns `false`.
    *
    *   For correctness, `whileLoop()` strictly enforces shape invariants for the loop variables. A shape invariant is a
    *   (possibly partial) shape that is unchanged across the iterations of the loop. An error will be raised if the
    *   shape of a loop variable after an iteration is determined to be more general than or incompatible with its shape
    *   invariant. For example, a shape of `[11, -1]` is more general than a shape of `[11, 17]`, and `[11, 21]` is not
    *   compatible with `[11, 17]`. By default, (if the argument `shapeInvariants` is not specified), it is assumed that
    *   the initial shape of each tensor in `loopVariables` is the same in every iteration. The `shapeInvariants`
    *   argument allows the caller to specify a less specific shape invariant for each loop variable, which is needed if
    *   the shape varies between iterations. The `Output.setShape()` function may also be used in the `bodyFn` function
    *   to indicate that the output loop variable has a particular shape. The shape invariants for indexed slices and
    *   sparse tensors are treated specially as follows:
    *
    *     a) If a loop variable is an indexed slices, the shape invariant must be a shape invariant of the values tensor
    *     of the indexed slices. This means that the shapes of the three tensors of the indexed slices are `[shape(0)]`,
    *     `shape`, and `[shape.rank]`.
    *
    *     b) If a loop variable is a sparse tensor, the shape invariant must be a shape `[r]`, where `r` is the rank of
    *     the dense tensor represented by the sparse tensor. This means that the shapes of the three tensors of the
    *     sparse tensor are `[-1, r]`, `[-1]`, and `[r]`. Note that the shape invariant here is the shape of the sparse
    *     tensor `denseShape` field. It must be the shape of a vector.
    *
    *   `whileLoop()` implements non-strict semantics, enabling multiple iterations to run in parallel. The maximum
    *   number of parallel iterations can be controlled by `parallelIterations`, which gives users some control over
    *   memory consumption and execution order. For correct programs, `whileLoop()` should return the same result for
    *   any value `parallelIterations > 0`.
    *
    *   For training, TensorFlow stores the tensors that are produced in the forward pass and are needed in
    *   back-propagation. These tensors are a main source of memory consumption and often cause out-of-memory errors
    *   when training on GPUs. When the flag `swapMemory` is set to `true`, we swap out these tensors from the GPU to
    *   the CPU. This, for example, allows us to train RNN models with very long sequences and large batch sizes.
    *
    *   For example:
    *   {{{
    *     val i = tf.constant(0)
    *     val p = (i: Output) => tf.less(i, 10)
    *     val b = (i: Output) => tf.add(i, 1)
    *     val r = tf.whileLoop(p, b, i)
    *   }}}
    *
    *   Or, using more involved tensor structures:
    *   {{{
    *     val ijk0 = (tf.constant(0), (tf.constant(1), tf.constant(2)))
    *     val p = (i: Output, (j: Output, k: Output)) => i < 10
    *     val b = (i: Output, (j: Output, k: Output)) => (i + 1, (j + k, j - k))
    *     val r = tf.whileLoop(p, b, ijk0)
    *   }}}
    *
    *   Also, using shape invariants:
    *   {{{
    *     val i0 = tf.constant(0)
    *     val m0 = tf.ones(Shape(2, 2))
    *     val p = (i: Output, m: Output) => i < 10
    *     val b = (i: Output, m: Output) => (i + 1, tf.concatenate(Seq(m, m), axis = 0))
    *     val r = tf.whileLoop(p, b, (i0, m0), (i0.shape, Shape(-1, 2)))
    *   }}}
    *
    *   Example which demonstrates non-strict semantics:
    *
    *   In the following example, the final value of the counter `i` does not depend on `x`. So, the `whileLoop` can
    *   increment the counter parallel to updates of `x`. However, because the loop counter at one loop iteration
    *   depends on the value at the previous iteration, the loop counter itself cannot be incremented in parallel.
    *   Hence, if we just want the final value of the counter, then `x` will never be incremented, but the counter will
    *   be updated on a single thread. Conversely, if we want the value of the output, then the counter may be
    *   incremented on its own thread, while `x` can be incremented in parallel on a separate thread.
    *   In the extreme case, it is conceivable that the thread incrementing the counter runs until completion before `x`
    *   is incremented even a single time. The only thing that can never happen is that the thread updating `x` can
    *   never get ahead of the counter thread because the thread incrementing `x` depends on the value of the counter.
    *   {{{
    *     val n = 10000
    *     val x = tf.constant(Tensor.zeros(INT32, Shape(n)))
    *     val p = (i: Output, x: Output) => i < n
    *     val b = (i: Output, x: Output) => (tf.print(i + 1, Seq(i)), tf.print(x + 1, Seq(x), "x: "))
    *     val r = tf.whileLoop(p, b, (0, x))
    *
    *     val session = tf.Session()
    *
    *     // The following line prints [0] to [9999]
    *
    *     // The following line may increment the counter and x in parallel. The counter thread may get ahead of the
    *     // other thread, but not the other way around. So you may see things like "[9996] x: [9987]", meaning that
    *     // the counter thread is on iteration 9996, while the other thread is on iteration 9987.
    *     session.run(r._2)
    *   }}}
    */
  private[ops] trait Documentation
}
