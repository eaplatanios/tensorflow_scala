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

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.ShapeMismatchException
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT32, RESOURCE}
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}
import org.platanios.tensorflow.api.utilities.Collections

import com.google.protobuf.{ByteString, GeneratedMessageV3}
import org.tensorflow.framework.{AttrValue, CollectionDef, WhileContextDef}
import org.tensorflow.framework.CollectionDef.BytesList
import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{SeqLike, mutable}
import scala.collection.JavaConverters._
import scala.collection.generic.CanBuildFrom
import scala.language.postfixOps
import scala.reflect.ClassTag

/** Control flow context for the while-loop construct.
  *
  * @param  maximumIterations     Optional `INT32` scalar specifying the maximum number of iterations to loop for. If
  *                               `null` (the default), no iteration limit is enforced.
  * @param  parallelIterations    Number of iterations allowed to run in parallel.
  * @param  enableBackPropagation If `true`, back-propagation support is enabled for this while-loop context.
  * @param  swapMemory            If `true`, GPU-CPU memory swapping support is enabled for this while-loop context.
  * @param  _gradientLoopState    Gradient loop state.
  * @param  pivot                 `BOOLEAN` tensor used for the loop termination condition. Used in code generation for
  *                               the gradient computation.
  * @param  pivotForPredicate     We use this node to control constants created by the predicate function.
  * @param  pivotForBody          We use this node to control constants created by the body function.
  * @param  loopEnters            Enter tensors for loop variables.
  * @param  loopExits             Exit tensors for loop variables.
  * @param  _name                 Name prefix for this while-loop context.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] case class WhileLoopContext private[control_flow] (
    maximumIterations: Option[Output] = None,
    parallelIterations: Int = 10,
    enableBackPropagation: Boolean = true,
    swapMemory: Boolean = false,
    private[control_flow] val _gradientLoopState: Option[GradientLoopState] = None,
    private[ops] var pivot: Output = null,
    private var pivotForPredicate: Op = null,
    private var pivotForBody: Op = null,
    private[control_flow] val loopEnters: mutable.Set[Output] = mutable.Set.empty[Output],
    private[control_flow] val loopExits: mutable.Set[Output] = mutable.Set.empty[Output],
    private val _name: String = "WhileLoopContext"
) extends Context() with ProtoSerializable {
  require(parallelIterations > 0, "'parallelIterations' must be a positive integer.")

  override val name: String = Op.currentGraph.uniqueName(_name)

  override def controlPivot: Option[Op] = Option(pivotForBody).orElse(Option(pivotForPredicate))

  override def whileLoopContext(stopContext: Option[Context] = None): Option[WhileLoopContext] = Some(this)

  override def add(op: Op): Unit = {
    // For a reduction op, if the op is in a gradient context and its input is from its forward context, moving the op
    // to the forward context means we would store the tensor after the reduction as opposed to the tensor before the
    // reduction, and therefore we could significantly reduce memory consumption. For now, we do this only for a few
    // ops.
    var added = false
    if (Set("Shape", "Size", "Rank").contains(op.opType)) {
      val gradientContext = Op.currentControlFlowContext
      if (gradientContext.isDefined) {
        gradientContext.flatMap(_.whileLoopContext().flatMap(_.gradientLoopState)).foreach(gradientLoopState => {
          WhileLoopContext.getWhileLoopContext(op.inputs(0).op).foreach(opInputForwardContext => {
            if (opInputForwardContext == gradientLoopState.forwardContext) {
              val opInputContext = op.inputs(0).op.controlFlowContext
              op.controlFlowContext = opInputContext
              opInputContext.foreach(_.addInternal(op))
              added = true
            }
          })
        })
      }
    }
    if (!added)
      addInternal(op)
  }

//  override private[control_flow] def addInternal(op: Op): Unit = {
//    if (op.numInputs == 0) {
//      // Remove any external control dependencies on this op.
//      val controlInputs = removeExternalControlEdges(op)
//      // Add a control edge from the control pivot to this op.
//      if (controlInputs.isEmpty)
//        controlPivot.foreach(ControlFlow.addControlInput(op, _))
//      op.outputs.foreach(values += _.name)
//    } else {
//      op.inputs.zipWithIndex.foreach({
//        case (input, index) =>
//          val realInput = add(input)
//          if (realInput != input)
//            ControlFlow.updateInput(op, index, realInput)
//      })
//      // Remove any external control dependencies on this op.
//      removeExternalControlEdges(op)
//      // Add a control dependency to prevent loop invariants from enabling ops that should not be executed.
//      if (op.controlInputs.isEmpty &&
//          ((op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient") ||
//              op.inputs.forall(o => ControlFlow.isLoopConstantEnter(o.op))))
//        controlPivot.foreach(ControlFlow.addControlInput(op, _))
//      op.outputs.foreach(values += _.name)
//    }
//    if (outerContext.isDefined || !ControlFlow.isLoopExit(op)) {
//      op.graph.preventFetching(op)
//      op.outputs.foreach(op.graph.preventFeeding)
//    }
//    outerContext.foreach(_.addInnerOp(op))
//  }

  override def add(output: Output): Output = {
    if (values.contains(output.name)) {
      // Use the real value if it comes from an outer context. This is needed in particular for nested conditionals.
      externalValues.getOrElse(output.name, output)
    } else {
      var value: Output = null
      values += output.name
      // If we are in a gradient context and `output` is from its forward context, we use `getRealValue()`, which adds
      // the logic to save the history of `output` in the forward pass.
      Op.currentControlFlowContext.foreach(gradientContext => {
        gradientContext.whileLoopContext().flatMap(_.gradientLoopState).foreach(gradientLoopState => {
          WhileLoopContext.getWhileLoopContext(output.op).flatMap(forwardContext => {
            if (ControlFlow.isLoopExit(output.op))
              forwardContext.outerContext.flatMap(_.whileLoopContext())
            else
              Some(forwardContext)
          }).foreach(forwardContext => {
            if (forwardContext == gradientLoopState.forwardContext) {
              val realValue = gradientLoopState.getRealValue(output)
              externalValues += output.name -> realValue
              value = realValue
            }
          })
        })
      })
      if (value != null) {
        value
      } else {
        val result = outerContext.map(_.add(output)).getOrElse(output)
        // Create an enter op to make `result` known to this loop context.
        val enter = Op.createWith(controlDependencies = Set.empty[Op]) {
          val enter = ControlFlow.enter(result, name, isConstant = true, parallelIterations)
          enter.graph.preventFeeding(enter)
          enter
        }
        // Fix the control inputs and control flow context of these enter ops.
        fixControlInputsAndContext(Seq(enter))
        // Add `enter` in this context.
        values += enter.name
        externalValues += output.name -> enter
        enter
      }
    }
  }

  def backPropagate: Boolean = enableBackPropagation

  def gradientLoopState: Option[GradientLoopState] = _gradientLoopState

  private[this] def isInOuterContext(op: Op): Boolean = {
    val opContext = ControlFlow.getOutputContext(op)
    var outerContext = this.outerContext
    while (outerContext != opContext && outerContext.isDefined)
      outerContext = outerContext.flatMap(_.outerContext)
    outerContext == opContext
  }

  /** Adds the loop termination condition and the loop body to the graph. */
  private[control_flow] def buildLoop[T, TS](
      predicateFn: T => Output, bodyFn: T => T, loopVariables: T, shapeInvariants: Option[TS]
  )(implicit ev: WhileLoopVariable.Aux[T, TS]): T = {
    try {
      // Enter the frame for this loop.
      enter()

      val flattenedLoopVariables = ev.outputs(loopVariables)
      val flattenedShapeInvariants = shapeInvariants.map(ev.shapes)

      // Let the context know the loop variables so the loop variables would be added to the outer contexts properly.
      initializeValues(flattenedLoopVariables)
      val realVariables = outerContext.map(c => flattenedLoopVariables.map(c.add)).getOrElse(flattenedLoopVariables)
      val enterVariables = Op.createWith(controlDependencies = Set.empty[Op]) {
        val enterVariables = realVariables.map(v => ControlFlow.enter(
          v, name, isConstant = false, parallelIterations, useInputShape = flattenedShapeInvariants.isEmpty))
        enterVariables.foreach(v => v.graph.preventFeeding(v))
        enterVariables
      }

      // Find the closest enclosing non-None control pivot.
      var outerCtx: Option[Context] = outerContext
      var controlPivot: Option[Op] = None
      while (outerCtx.isDefined && controlPivot.isEmpty) {
        controlPivot = outerContext.get.controlPivot
        outerCtx = outerCtx.get.outerContext
      }

      controlPivot.foreach(p => {
        enterVariables.filter(v => ControlFlow.isLoopConstantEnter(v.op.inputs(0).op)).foreach(v => {
          ControlFlow.addControlInput(v.op, p)
        })
      })
      flattenedShapeInvariants.foreach(WhileLoopContext.setShapeInvariants(realVariables, enterVariables, _))

      // Fix the control inputs and control flow context of these enter ops.
      fixControlInputsAndContext(enterVariables)
      initializeValues(enterVariables)
      loopEnters.clear()
      loopEnters ++= enterVariables

      val mergeVariables = enterVariables.map(v => ControlFlow.merge(Seq(v, v))._1)
      pivotForPredicate = mergeVariables(0).op

      // Build the graph for the predicate.
      val packedPredicateVariables = ev.fromOutputs(loopVariables, mergeVariables)
      val predicateResult = predicateFn(packedPredicateVariables)
      pivot = ControlFlow.loopCond(predicateResult, name = "LoopCond")
      val switchVariables = mergeVariables.map(v => ControlFlow.colocatedSwitch(v, pivot))

      // Build the graph for the body.
      val bodyVariables = switchVariables.map(v => Basic.identity(v._2))
      pivotForBody = bodyVariables(0).op
      val packedBodyVariables = ev.fromOutputs(loopVariables, bodyVariables)
      val bodyResult = bodyFn(packedBodyVariables)

      // Convert the tensor arrays returned by the body function into their flow variables.
      val flattenedBodyResult = ev.outputs(bodyResult)

      // Add the `NextIteration` op and the back edges to complete the loop.
      mergeVariables.zip(flattenedBodyResult).map(p => {
        WhileLoopContext.addNextIterationAndBackEdge(p._1, p._2)
      })

      // Add the exit ops.
      val exitVariables = switchVariables.map(v => ControlFlow.exit(v._1))
      loopExits.clear()
      loopExits ++= exitVariables

      // Exit the loop.
      exitResult(exitVariables)

      // Convert any tensor array flow variables outside the context back into their associated tensor arrays for
      // returning to the caller.
      ev.fromOutputs(bodyResult, exitVariables)
    } catch {
      case t: Throwable =>
        exit()
        throw t
    }
  }

  private[this] def fixControlInputsAndContext(values: Seq[OutputLike]): Unit = {
    values.foreach(value => {
      val outputs = value match {
        case o: Output => Set(o)
        case o: OutputIndexedSlices =>
          if (o.denseShape != null)
            Set(o.indices, o.values, o.denseShape)
          else
            Set(o.indices, o.values)
        case o: SparseOutput =>
          if (o.denseShape != null)
            Set(o.indices, o.values, o.denseShape)
          else
            Set(o.indices, o.values)
      }
      outputs.foreach(output => {
        val input = output.op.inputs(0)
        val outerControlInputs = Op.controlDependencies(Set(input)).filter(isInOuterContext)
        output.op.controlFlowContext = Some(this)
        outerControlInputs.foreach(i => ControlFlow.addControlInput(output.op, i))
      })
    })
  }

  /** Makes the provided values known to this context. */
  private[this] def initializeValues(providedValues: Seq[OutputLike]): Unit = {
    values.clear()
    providedValues.foreach {
      case v: Output => values += v.name
      case v: OutputIndexedSlices =>
        values += v.indices.name
        values += v.values.name
        if (v.denseShape != null)
          values += v.denseShape.name
      case v: SparseOutput =>
        values += v.indices.name
        values += v.values.name
        if (v.denseShape != null)
          values += v.denseShape.name
    }
  }

  /** Adds a loop that counts the number of iterations.
    *
    * This is added to the forward loop at the time when we start to create the loop for the back-propagation gradient
    * computation. It is called in the outer context of this forward context.
    *
    * The pseudocode is: `n = 0; while (pivot) { n++; }`
    *
    * Note that a control dependency is added to `n` to ensure the correct execution order of stack push ops.
    *
    * @param  outerGradientLoopState Outer gradient loop state (`None` if not nested).
    * @return Tuple containing the number of iterations taken by the forward loop and the loop index.
    */
  private[control_flow] def addForwardLoopCounter(
      outerGradientLoopState: Option[GradientLoopState]): (Output, Output) = {
    val n = Basic.constant(0, name = "ForwardLoopCounter")
    outerGradientLoopState.foreach(state => {
      // Force the stack pushes of the i-th execution of an inner loop to be ordered before the pushes of the (i+1)-th
      // execution of the same inner loop.
      ControlFlow.addControlInput(n.op, state.forwardIndex.op.inputs(0).op)
    })
    enter()
    values += n.name
    val enterN = ControlFlow.enter(n, name, isConstant = false, parallelIterations, name = "ForwardLoopCounter")
    loopEnters += enterN
    val mergeN = ControlFlow.merge(Seq(enterN, enterN))._1
    val switchN = ControlFlow.switch(mergeN, pivot)
    val index = Math.add(switchN._2, 1)
    val nextN = ControlFlow.nextIteration(index)
    ControlFlow.updateInput(mergeN.op, 1, nextN)
    val exitN = ControlFlow.exit(switchN._1, name = "ForwardLoopCounter").toOutput
    loopExits += exitN
    exitResult(Seq(exitN))
    exit()
    (exitN, nextN)
  }

  /** Adds the back-propagation loop that counts the number of iterations.
    *
    * This is added to the back-propagation loop. It is used to control the loop termination of the back-propagation
    * loop. It is called in the outer context of this gradient context.
    *
    * The pseudocode is: `n = count; while (n >= 1) { n--; }`
    *
    * Note that a control dependency is added to the final exit op to ensure the correct execution order of stack pop
    * ops.
    *
    * @param  count              Number of iterations for the back-propagation loop.
    * @param  outerGradientLoopState Outer gradient loop state (`None` if not nested).
    * @return Loop index.
    */
  private[control_flow] def addBackwardLoopCounter(
      count: Output, outerGradientLoopState: Option[GradientLoopState]): Output = {
    val one = Basic.constant(1, name = "BackwardLoopCounter")
    enter()
    values += count.name
    val enterC = ControlFlow.enter(count, name, isConstant = false, parallelIterations, name = "BackwardLoopCounter")
    loopEnters += enterC
    val mergeC = ControlFlow.merge(Seq(enterC, enterC))._1
    pivotForPredicate = mergeC
    pivot = ControlFlow.loopCond(Math.greaterEqual(mergeC, one), name = "BackwardLoopCounter")
    val switchC = ControlFlow.switch(mergeC, pivot)
    val indexC = Math.subtract(switchC._2, one)
    pivotForBody = indexC
    val nextC = ControlFlow.nextIteration(indexC)
    ControlFlow.updateInput(mergeC.op, 1, nextC)
    val exitC = ControlFlow.exit(switchC._1, name = "BackwardLoopCounter")
    loopExits += exitC
    outerGradientLoopState.foreach(state => {
      // Force the stack pops of the i-th execution of an inner loop to be ordered before the pops of the (i+1)-th
      // execution of the same inner loop.
      ControlFlow.addControlInput(state.backwardSync, exitC.op)
    })
    exitResult(Seq(exitC))
    exit()
    nextC
  }

  /** Adds an accumulation loop for every loop invariant.
    *
    * This is added to the back-propagation loop. It is used to accumulate partial gradients within each loop iteration.
    * It is called when in the gradient while context.
    *
    * The pseudocode is: `acc = 0.0; while (pivot) { acc += grad; }`
    *
    * @param  op       Enter op for a loop invariant.
    * @param  gradient Partial gradient of an iteration for a loop invariant.
    * @return Gradient for a loop invariant.
    */
  private[control_flow] def addBackwardAccumulator[T <: OutputLike](op: Op, gradient: T): T = {
    val result = gradient match {
      case g: Output =>
        exit()
        // We create a zeros tensor with the right shape for the accumulator. If we don't know the full shape
        // statically, we will have to get the shape dynamically from the forward inference. Getting the shape right for
        // the zeros is only needed for the base case when the loop exits without running any iterations.
        val shape = g.shape
        val acc: Output = {
          if (shape.isFullyDefined) {
            outerContext.foreach(_.enter())
            val acc = Basic.zerosLike(g, name = "BackwardAccumulator")
            outerContext.foreach(_.exit())
            acc
          } else {
            val value = op.inputs(0)
            // TODO: !!! [CONTROL_FLOW] Is this even necessary for obtaining the shape?
            outerContext match {
              case Some(context: WhileLoopContext) if context.gradientLoopState.isDefined =>
                // We are in a nested while loop.
                val forwardContext = context.gradientLoopState.get.forwardContext
                forwardContext.outerContext.foreach(_.enter())
                val zerosShape = resourceSafeShape(value)
                forwardContext.outerContext.foreach(_.exit())
                val outerGradientLoopState = context.gradientLoopState.get.outerGradientLoopState.get
                val historyZerosShape = outerGradientLoopState.addForwardAccumulator(zerosShape)
                context.enter()
                val realShape = outerGradientLoopState.addBackwardAccumulatedValue(historyZerosShape, zerosShape)
                val acc = Basic.zeros(g.dataType, realShape)
                context.exit()
                acc.setShape(g.shape)
                acc
              case _ =>
                outerContext.foreach(_.enter())
                val zerosShape = resourceSafeShape(value)
                val acc = Basic.zeros(g.dataType, zerosShape)
                outerContext.foreach(_.exit())
                // TODO: [CONTROL_FLOW] Figure out if this is necessary.
                // acc.setShape(g.shape)
                acc
            }
          }
        }
        Op.createWithNameScope("BackwardAccumulator") {
          enter()
          values += acc.name
          val enterAcc = ControlFlow.enter(acc, name, isConstant = false, parallelIterations)
          loopEnters += enterAcc
          val mergeAcc = ControlFlow.merge(Seq(enterAcc, enterAcc))._1.toOutput
          val switchAcc = ControlFlow.switch(mergeAcc, pivot)
          val addAcc = Math.add(switchAcc._2, g)
          val nextAcc = ControlFlow.nextIteration(addAcc)
          ControlFlow.updateInput(mergeAcc.op, 1, nextAcc)
          val exitAcc = ControlFlow.exit(switchAcc._1)
          loopExits += exitAcc
          exitResult(Seq(exitAcc))
          exitAcc
        }
      case g: OutputIndexedSlices => Op.createWithNameScope("BackwardAccumulator") {
        exit()
        // We create a zeros tensor with the right shape for the accumulator. If we don't know the full shape
        // statically, we will have to get the shape dynamically from the forward inference. Getting the shape right for
        // the zeros is only needed for the base case when the loop exits without running any iterations.
        outerContext.foreach(_.enter())
        val indicesAcc: Output = Basic.zeros(g.indices.dataType, Shape(1))
        val valuesAcc: Output = {
          if (g.values.shape.isFullyDefined) {
            val zerosShape = Shape(1 +: g.values.shape.asArray.tail: _*)
            Basic.zeros(g.values.dataType, zerosShape)
          } else {
            val value = op.inputs(0)
            // TODO: !!! [CONTROL_FLOW] Is this even necessary for obtaining the shape?
            outerContext match {
              case Some(context: WhileLoopContext) if context.gradientLoopState.isDefined =>
                // We are in a nested while loop.
                val forwardContext = context.gradientLoopState.get.forwardContext
                forwardContext.outerContext.foreach(_.enter())
                val zerosShape = Basic.concatenate(
                  Seq(Tensor(1L), resourceSafeShape(value).slice(1 ::)), axis = 0)
                forwardContext.outerContext.foreach(_.exit())
                val outerGradientLoopState = context.gradientLoopState.get.outerGradientLoopState.get
                val historyZerosShape = outerGradientLoopState.addForwardAccumulator(zerosShape)
                context.enter()
                val realShape = outerGradientLoopState.addBackwardAccumulatedValue(historyZerosShape, zerosShape)
                val acc = Basic.zeros(g.values.dataType, realShape)
                context.exit()
                // TODO: [CONTROL_FLOW] Figure out if this is necessary.
                // acc.setShape(g.values.shape)
                acc
              case _ =>
                val zerosShape = Basic.concatenate(
                  Seq(Tensor(1L), resourceSafeShape(op.inputs(0)).slice(1 ::)), axis = 0)
                Basic.zeros(g.values.dataType, zerosShape)
            }
          }
        }
        val denseShapeAcc: Option[Output] = Option(g.denseShape).map(shape => {
          if (shape.shape.isFullyDefined) {
            Basic.zeros(shape.dataType, shape.shape)
          } else {
            Basic.zerosLike(Basic.shape(op.inputs(0), optimize = false), optimize = false)
          }
        })
        outerContext.foreach(_.exit())
        enter()
        values += indicesAcc.name
        values += valuesAcc.name
        denseShapeAcc.foreach(values += _.name)
        val initAcc = Seq(indicesAcc, valuesAcc) ++ denseShapeAcc.map(Seq(_)).getOrElse(Seq.empty)
        // Set `useInputShape` to `false` since the accumulator tensors will grow in size. If `useInputShape` is `true`,
        // the `updateInput` call below will result in incompatible shapes.
        val enterAcc = initAcc.map(a => {
          ControlFlow.enter(a, name, isConstant = false, parallelIterations, useInputShape = false)
        })
        // Manually set appropriate partial shapes.
        enterAcc.head.setShape(Shape(-1))
        if (valuesAcc.rank != -1 && valuesAcc.rank > 1)
          enterAcc(1).setShape(Shape(-1) ++ valuesAcc.shape(1 ::))
        else if (valuesAcc.rank != -1)
          enterAcc(1).setShape(Shape(-1))
        loopEnters ++= enterAcc
        val mergeAcc = enterAcc.map(a => ControlFlow.merge(Seq(a, a))._1)
        val switchAcc = mergeAcc.map(a => ControlFlow.switch(a, pivot))

        // The actual accumulation.
        var addAcc = mutable.ListBuffer(
          Basic.concatenate(Seq(switchAcc(0)._2, g.indices), 0),
          Basic.concatenate(Seq(switchAcc(1)._2, g.values), 0))
        denseShapeAcc.foreach(_ => {
          // For the shape we just keep the maximum.
          addAcc += Math.maximum(g.denseShape, switchAcc(2)._2)
        })
        val nextAcc = addAcc.map(ControlFlow.nextIteration(_))
        mergeAcc.zip(nextAcc).foreach(a => ControlFlow.updateInput(a._1.op, 1, a._2))
        val exitAcc = switchAcc.map(a => ControlFlow.exit(a._1))
        loopExits ++= exitAcc
        exitResult(exitAcc)
        OutputIndexedSlices(exitAcc(0), exitAcc(1), denseShapeAcc.map(_ => exitAcc(2)).orNull)
      }
      case g: SparseOutput => ???
    }
    result.asInstanceOf[T]
  }

  /** Returns the shape of `value` of the shape of the variable it points to. */
  private[this] def resourceSafeShape(value: Output): Output = {
    if (value.dataType == RESOURCE) {
      var v = value
      while (v.op.inputs.nonEmpty)
        v = v.op.inputs(0)
      v.op.shapeAttribute("shape")
    } else {
      Basic.shape(value, optimize = false)
    }
  }

  override def toProto: GeneratedMessageV3 = toProto(null)

  /** Alias for `toWhileContextDef`. */
  override def toProto(exportScope: String = null): GeneratedMessageV3 = toWhileContextDef(exportScope)

  /** Constructs and returns a [[WhileContextDef]] object that represents this while-loop context.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[WhileContextDef]].
    */
  def toWhileContextDef(exportScope: String = null): WhileContextDef = {
    if (exportScope == null || name.startsWith(exportScope)) {
      // TODO: !!! Add `maximumIterations` when possible.
      WhileContextDef.newBuilder()
          .setContextName(Op.stripNameScope(exportScope, name))
          .setParallelIterations(parallelIterations)
          .setBackProp(enableBackPropagation)
          .setSwapMemory(swapMemory)
          .setPivotName(Op.stripNameScope(exportScope, pivot.name))
          .setPivotForPredName(Op.stripNameScope(exportScope, pivotForPredicate.name))
          .setPivotForBodyName(Op.stripNameScope(exportScope, pivotForBody.name))
          .addAllLoopEnterNames(loopEnters.map(e => Op.stripNameScope(exportScope, e.name)).asJava)
          .addAllLoopExitNames(loopExits.map(e => Op.stripNameScope(exportScope, e.name)).asJava)
          .setValuesDef(super.toValuesDef(exportScope))
          .build()
    } else {
      null
    }
  }
}

object WhileLoopContext {
  /** Returns the while-loop context to which `op` belongs. */
  private[control_flow] def getWhileLoopContext(op: Op): Option[WhileLoopContext] = {
    op.controlFlowContext.flatMap(_.whileLoopContext())
  }

  /** Creates a next iteration op for `v` and adds a back edge from `v` to `m`. */
  @throws[IllegalArgumentException]
  private[ops] def addNextIterationAndBackEdge[T <: OutputLike](
      m: T, v: T, enforceShapeInvariant: Boolean = true): T = {
    val result = (m, v) match {
      case (mm: Output, vv: Output) =>
        val nextVV = ControlFlow.nextIteration(vv)
        if (enforceShapeInvariant) {
          // Make sure the shapes of the loop outputs are correct. We do this before calling `updateInput`, which will
          // raise a less helpful error message if the types do not match.
          // TODO: Apply the same checks for the other cases, below.
          WhileLoopContext.enforceShapeInvariant(mm, nextVV)
        }
        ControlFlow.updateInput(mm.op, 1, nextVV)
        nextVV
      case (mm: OutputIndexedSlices, vv: OutputIndexedSlices) =>
        val nextVV = ControlFlow.nextIteration(vv)
        ControlFlow.updateInput(mm.values.op, 1, nextVV.values)
        ControlFlow.updateInput(mm.indices.op, 1, nextVV.indices)
        if (mm.denseShape != null) {
          if (nextVV.denseShape == null)
            throw new IllegalArgumentException(s"Output indexed slices '$nextVV' must have dense shape information.")
          else
            ControlFlow.updateInput(mm.denseShape.op, 1, nextVV.denseShape)
        }
        nextVV
      case (mm: SparseOutput, vv: SparseOutput) =>
        val nextVV = ControlFlow.nextIteration(vv)
        ControlFlow.updateInput(mm.values.op, 1, nextVV.values)
        ControlFlow.updateInput(mm.indices.op, 1, nextVV.indices)
        if (mm.denseShape != null) {
          if (nextVV.denseShape == null)
            throw new IllegalArgumentException(s"Sparse output '$nextVV' must have dense shape information.")
          else
            ControlFlow.updateInput(mm.denseShape.op, 1, nextVV.denseShape)
        }
        nextVV
      case (_, _) =>
        throw new IllegalArgumentException(
          "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the tensor types must match.")
    }
    result.asInstanceOf[T]
  }

  //region Shape Invariants

  /** Returns `true` if `shape2` is a less strict shape than `shape1`, while being compatible with `shape1`. */
  private[this] def shapeLessThenOrEqual(shape1: Shape, shape2: Shape): Boolean = {
    shape2.rank == -1 ||
        shape1.rank == shape2.rank ||
        shape1.asArray.zip(shape2.asArray).forall(pair => pair._2 == -1 || pair._1 == pair._2)
  }

  /** Sets the shapes of the tensors in `enterTensors` to `shapes` and makes sure that the shape invariants apply.
    *
    * @param  inputTensors Tensors that are inputs to `enterTensors`.
    * @param  enterTensors Tensors whose shapes will be set.
    * @param  shapes       Shapes to use for `enterTensors`.
    * @throws ShapeMismatchException   If any tensor in `inputTensors` has a less specific shape than its corresponding
    *                                  shape in `shapes`.
    * @throws IllegalArgumentException If the types of the input tensors do not match the types of the enter tensors or
    *                                  if the type of either is not supported.
    */
  @throws[ShapeMismatchException]
  @throws[IllegalArgumentException]
  private[WhileLoopContext] def setShapeInvariants(
      inputTensors: Seq[OutputLike], enterTensors: Seq[OutputLike], shapes: Seq[Shape]): Unit = {
    // Check that the shapes of the inputs are less than the shape invariants, and set the shapes of the enter tensors
    // to the shape invariants.
    for ((input, enter, shape) <- (inputTensors, enterTensors, shapes).zipped) {
      (input, enter) match {
        case (i: Output, e: Output) =>
          if (!shapeLessThenOrEqual(i.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.name}' is not compatible with the initial shape of the " +
                  s"loop variable. It enters the loop with shape '${i.shape}', but the specified shape invariant " +
                  s"is '$shape'.")
          e.setShape(shape)
        case (i: OutputIndexedSlices, e: OutputIndexedSlices) =>
          if (!shapeLessThenOrEqual(i.values.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.values.name}' is not compatible the initial shape of the " +
                  s"values tensor of these indexed slices. It enters the loop with shape '${i.values.shape}', but " +
                  s"the specified shape invariant is '$shape'.")
          e.values.setShape(shape)
          e.indices.setShape(Shape(shape(0)))
          if (e.denseShape != null)
            e.denseShape.setShape(Shape(shape.rank))
        case (i: SparseOutput, e: SparseOutput) =>
          if (!shapeLessThenOrEqual(i.denseShape.shape, shape))
            throw ShapeMismatchException(
              s"The shape invariant specified for '${i.denseShape.name}' is not compatible the initial shape of the " +
                  s"dense shape tensor of this sparse tensor. It enters the loop with shape '${i.denseShape.shape}', " +
                  s" but the specified shape invariant is '$shape'.")
          e.values.setShape(Shape(-1))
          e.indices.setShape(Shape(-1, shape.rank))
          e.denseShape.setShape(shape)
        case (_, _) =>
          throw new IllegalArgumentException(
            "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the input tensor " +
                "and the enter tensor types must match.")
      }
    }
  }

  /** Checks if the shapes of a loop variable satisfy the shape invariants.
    *
    * @param  mergeTensor Tensor representing the initial value of the loop variable.
    * @param  nextTensor  Tensor representing the value of the loop variable after one loop iteration.
    * @throws ShapeMismatchException   If `mergeTensor` has a less specific shape than its corresponding shape in
    *                                  `nextTensor`.
    * @throws IllegalArgumentException If the type of the merge tensor does not match the type of the next tensor or if
    *                                  the type of either is not supported.
    */
  @throws[ShapeMismatchException]
  @throws[IllegalArgumentException]
  private[WhileLoopContext] def enforceShapeInvariant(mergeTensor: OutputLike, nextTensor: OutputLike): Unit = {
    (mergeTensor, nextTensor) match {
      case (merge: Output, next: Output) =>
        if (!shapeLessThenOrEqual(next.shape, merge.shape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'${merge.shape}', but has shape '${next.shape}' after one iteration. Please provide shape " +
                s"invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' method of " +
                s"the loop variables.")
      case (merge: OutputIndexedSlices, next: OutputIndexedSlices) =>
        val mergeValuesShape = merge.values.shape
        val mergeIndicesShape = merge.indices.shape
        val mergeDenseShapeShape = if (merge.denseShape != null) merge.denseShape.shape else Shape.unknown()
        val nextValuesShape = next.values.shape
        val nextIndicesShape = next.indices.shape
        val nextDenseShapeShape = if (next.denseShape != null) next.denseShape.shape else Shape.unknown()
        if (!shapeLessThenOrEqual(nextValuesShape, mergeValuesShape) ||
            !shapeLessThenOrEqual(nextIndicesShape, mergeIndicesShape) ||
            !shapeLessThenOrEqual(nextDenseShapeShape, mergeDenseShapeShape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'($mergeValuesShape, $mergeIndicesShape, $mergeDenseShapeShape)', but has shape " +
                s"'($nextValuesShape, $nextIndicesShape, $nextDenseShapeShape)' after one iteration. Please provide " +
                s"shape invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' " +
                s"method of the loop variables.")
      case (merge: SparseOutput, next: SparseOutput) =>
        val mergeValuesShape = merge.values.shape
        val mergeIndicesShape = merge.indices.shape
        val mergeDenseShapeShape = merge.denseShape.shape
        val nextValuesShape = next.values.shape
        val nextIndicesShape = next.indices.shape
        val nextDenseShapeShape = next.denseShape.shape
        if (!shapeLessThenOrEqual(nextValuesShape, mergeValuesShape) ||
            !shapeLessThenOrEqual(nextIndicesShape, mergeIndicesShape) ||
            !shapeLessThenOrEqual(nextDenseShapeShape, mergeDenseShapeShape))
          throw ShapeMismatchException(
            s"The shape for '${merge.name}' is not an invariant for the loop. The tensor enters the loop with shape " +
                s"'($mergeValuesShape, $mergeIndicesShape, $mergeDenseShapeShape)', but has shape " +
                s"'($nextValuesShape, $nextIndicesShape, $nextDenseShapeShape)' after one iteration. Please provide " +
                s"shape invariants using either the 'shapeInvariants' argument of 'whileLoop' or the 'setShape' " +
                s"method of the loop variables.")
      case (_, _) =>
        throw new IllegalArgumentException(
          "Only 'Output', 'OutputIndexedSlices', and 'SparseOutput' are supported. Also, the merge tensor " +
              "and the next tensor types must match>")
    }
  }

  //endregion Shape Invariants

  /** Creates a [[WhileLoopContext]] from the provided [[WhileContextDef]] object.
    *
    * @param  whileContextDef Serialized while-loop context object.
    * @param  importScope     Optional prefix that will be prepended to all op names in the cond context that is being
    *                         loaded from the provided [[WhileContextDef]].
    * @return Constructed [[WhileLoopContext]].
    */
  def fromWhileContextDef(whileContextDef: WhileContextDef, importScope: String = null): WhileLoopContext = {
    // TODO: !!! Add `maximumIterations` when possible.
    val graph = Op.currentGraph
    val name = Op.prependNameScope(importScope, whileContextDef.getContextName)
    val parallelIterations = whileContextDef.getParallelIterations
    val enableBackPropagation = whileContextDef.getBackProp
    val swapMemory = whileContextDef.getSwapMemory
    val pivot = graph.getOutputByName(Op.prependNameScope(importScope, whileContextDef.getPivotName))
    val pivotForPredicate = graph.getOpByName(Op.prependNameScope(importScope, whileContextDef.getPivotForPredName))
    val pivotForBody = graph.getOpByName(Op.prependNameScope(importScope, whileContextDef.getPivotForBodyName))
    val loopEnters = mutable.Set(whileContextDef.getLoopEnterNamesList.asScala.map(name => {
      graph.getOutputByName(Op.prependNameScope(importScope, name))
    }): _*)
    val loopExits = mutable.Set(whileContextDef.getLoopExitNamesList.asScala.map(name => {
      graph.getOutputByName(Op.prependNameScope(importScope, name))
    }): _*)
    val (values, externalValues) = Context.fromValuesDef(whileContextDef.getValuesDef, importScope)
    val whileLoopContext = WhileLoopContext(
      None, parallelIterations, enableBackPropagation, swapMemory, None, pivot, pivotForPredicate, pivotForBody,
      loopEnters, loopExits, name)
    whileLoopContext.values ++= values
    whileLoopContext.externalValues ++= externalValues
    if (importScope != null) {
      val frameName = AttrValue.newBuilder().setS(ByteString.copyFromUtf8(whileLoopContext.name)).build()
      values.map(Op.currentGraph.findOp).filter(_.exists(ControlFlow.isLoopEnter)).foreach(_.foreach(op => {
        ControlFlow.setAttribute(op, "frame_name", frameName)
      }))
    }
    whileLoopContext
  }

  /** Key for collections of [[WhileLoopContext]]s. */
  trait CollectionKey extends Graph.Key[WhileLoopContext] {
    override def createCollectionDef(values: Set[WhileLoopContext], exportScope: String = null): CollectionDef = {
      val bytesListBuilder = BytesList.newBuilder()
      values
          .map(_.toProto(exportScope))
          .filter(_ != null)
          .foreach(s => bytesListBuilder.addValue(s.toByteString))
      CollectionDef.newBuilder().setBytesList(bytesListBuilder.build()).build()
    }

    override def parseCollectionDef(collectionDef: CollectionDef, graph: Graph, importScope: String): Unit = {
      val kind = collectionDef.getKindCase
      if (kind != CollectionDef.KindCase.BYTES_LIST)
        throw new IllegalArgumentException(s"The '$name' collection should be stored as a byte list.")
      collectionDef.getBytesList.getValueList.asScala
          .foreach(s => graph.addToCollection(
            WhileLoopContext.fromWhileContextDef(WhileContextDef.parseFrom(s), importScope), this))
    }
  }

  /** Key to collect the [[WhileLoopContext]]s that have been created in the graph. */
  object WHILE_LOOP_CONTEXTS extends CollectionKey {
    override def name: String = "while_context"
  }
}

/** Type trait used for representing supported while-loop construct loop variable types. */
trait WhileLoopVariable[T] {
  type ShapeType

  /** Helper used by the RNN construction code to create default states. */
  def zero(batchSize: Output, dataType: DataType, shape: ShapeType, name: String = "Zero"): T

  def size(output: T): Int
  def outputs(output: T): Seq[Output]
  def shapes(shape: ShapeType): Seq[Shape]
  def fromOutputs(output: T, values: Seq[Output]): T = segmentOutputs(output, values)._1
  def segmentOutputs(output: T, values: Seq[Output]): (T, Seq[Output])
  def fromShapes(output: T, values: Seq[Shape]): ShapeType = segmentShapes(output, values)._1
  def segmentShapes(output: T, values: Seq[Shape]): (ShapeType, Seq[Shape])

  // TODO: These "map" function involve some runtime checking for the "Symbol" type that would be good to work around.
  def map(value: T, mapFn: OutputConvertible => OutputConvertible): T
  def mapWithShape(value: T, shape: ShapeType, mapFn: (OutputConvertible, Shape) => OutputConvertible): T
}

object WhileLoopVariable {
  type Aux[T, TS] = WhileLoopVariable[T] {
    type ShapeType = TS
  }

  implicit val outputWhileLoopVariable: Aux[Output, Shape] = new WhileLoopVariable[Output] {
    override type ShapeType = Shape

    override def zero(batchSize: Output, dataType: DataType, shape: Shape, name: String = "Zero"): Output = {
      val staticBatchSize = Output.constantValue(batchSize).map(_.scalar.asInstanceOf[Int]).getOrElse(-1)
      Op.createWithNameScope(name, Set(batchSize.op)) {
        val fullShape = Basic.concatenate(
          Seq(batchSize.expandDims(0), shape.toOutput(batchSize.dataType)), axis = 0)
        val zero = Basic.zeros(dataType, fullShape)
        zero.setShape(Shape(staticBatchSize) ++ shape)
        zero
      }
    }

    override def size(output: Output): Int = 1
    override def outputs(output: Output): Seq[Output] = Seq(output)
    override def shapes(shape: Shape): Seq[Shape] = Seq(shape)

    override def segmentOutputs(output: Output, values: Seq[Output]): (Output, Seq[Output]) = {
      (values.head, values.tail)
    }

    override def segmentShapes(output: Output, values: Seq[Shape]): (Shape, Seq[Shape]) = {
      (values.head, values.tail)
    }

    override def map(value: Output, mapFn: OutputConvertible => OutputConvertible): Output = {
      mapFn(value).asInstanceOf[Output]
    }

    override def mapWithShape(
        value: Output,
        shape: Shape,
        mapFn: (OutputConvertible, Shape) => OutputConvertible
    ): Output = {
      mapFn(value, shape).asInstanceOf[Output]
    }
  }

  implicit val outputIndexedSlicesWhileLoopVariable: Aux[OutputIndexedSlices, Shape] = {
    new WhileLoopVariable[OutputIndexedSlices] {
      override type ShapeType = Shape

      override def zero(
          batchSize: Output,
          dataType: DataType,
          shape: Shape,
          name: String = "Zero"
      ): OutputIndexedSlices = {
        Op.createWithNameScope(name, Set(batchSize.op)) {
          val fullShape = Basic.concatenate(
            Seq(batchSize.expandDims(0), shape.toOutput(batchSize.dataType)), axis = 0)
          OutputIndexedSlices(
            Basic.zeros(INT32, Shape(0)),
            Basic.zeros(dataType, Shape.fromSeq(0 +: shape.asArray)),
            fullShape)
        }
      }

      override def size(output: OutputIndexedSlices): Int = 3

      override def outputs(output: OutputIndexedSlices): Seq[Output] = {
        Seq(output.indices, output.values, output.denseShape)
      }

      override def shapes(shape: Shape): Seq[Shape] = Seq(shape)

      override def segmentOutputs(
          output: OutputIndexedSlices,
          values: Seq[Output]
      ): (OutputIndexedSlices, Seq[Output]) = {
        (OutputIndexedSlices(values(0), values(1), values(2)), values.drop(3))
      }

      override def segmentShapes(output: OutputIndexedSlices, values: Seq[Shape]): (Shape, Seq[Shape]) = {
        (values.head, values.tail)
      }

      override def map(
          value: OutputIndexedSlices,
          mapFn: OutputConvertible => OutputConvertible
      ): OutputIndexedSlices = {
        mapFn(value).asInstanceOf[OutputIndexedSlices]
      }

      override def mapWithShape(
          value: OutputIndexedSlices,
          shape: Shape,
          mapFn: (OutputConvertible, Shape) => OutputConvertible
      ): OutputIndexedSlices = {
        mapFn(value, shape).asInstanceOf[OutputIndexedSlices]
      }
    }
  }

  implicit val sparseOutputWhileLoopVariable: Aux[SparseOutput, Shape] = {
    new WhileLoopVariable[SparseOutput] {
      override type ShapeType = Shape

      override def zero(
          batchSize: Output,
          dataType: DataType,
          shape: Shape,
          name: String = "Zero"
      ): SparseOutput = {
        Op.createWithNameScope(name, Set(batchSize.op)) {
          val fullShape = Basic.concatenate(
            Seq(batchSize.expandDims(0), shape.toOutput(batchSize.dataType)), axis = 0)
          SparseOutput(
            Basic.zeros(INT32, Shape(0, fullShape.rank)),
            Basic.zeros(dataType, Shape(0)),
            fullShape)
        }
      }

      override def size(output: SparseOutput): Int = 3
      override def outputs(output: SparseOutput): Seq[Output] = Seq(output.indices, output.values, output.denseShape)
      override def shapes(shape: Shape): Seq[Shape] = Seq(shape)

      override def segmentOutputs(output: SparseOutput, values: Seq[Output]): (SparseOutput, Seq[Output]) = {
        (SparseOutput(values(0), values(1), values(2)), values.drop(3))
      }

      override def segmentShapes(output: SparseOutput, values: Seq[Shape]): (Shape, Seq[Shape]) = {
        (values.head, values.tail)
      }

      override def map(value: SparseOutput, mapFn: OutputConvertible => OutputConvertible): SparseOutput = {
        mapFn(value).asInstanceOf[SparseOutput]
      }

      override def mapWithShape(
          value: SparseOutput,
          shape: Shape,
          mapFn: (OutputConvertible, Shape) => OutputConvertible
      ): SparseOutput = {
        mapFn(value, shape).asInstanceOf[SparseOutput]
      }
    }
  }

  implicit val tensorArrayWhileLoopVariable: Aux[TensorArray, Shape] = new WhileLoopVariable[TensorArray] {
    override type ShapeType = Shape

    override def zero(batchSize: Output, dataType: DataType, shape: Shape, name: String = "Zero"): TensorArray = ???

    override def size(output: TensorArray): Int = 1
    override def outputs(output: TensorArray): Seq[Output] = Seq(output.flow)
    override def shapes(shape: Shape): Seq[Shape] = Seq(shape)

    override def segmentOutputs(output: TensorArray, values: Seq[Output]): (TensorArray, Seq[Output]) = {
      val newTensorArray = output.copy(flow = values.head)
      // TODO: !!! [TENSOR_ARRAY] What about colocate with?
      (newTensorArray, values.tail)
    }

    override def segmentShapes(output: TensorArray, values: Seq[Shape]): (Shape, Seq[Shape]) = {
      (values.head, values.tail)
    }

    override def map(value: TensorArray, mapFn: OutputConvertible => OutputConvertible): TensorArray = {
      mapFn(value).asInstanceOf[TensorArray]
    }

    override def mapWithShape(
        value: TensorArray,
        shape: Shape,
        mapFn: (OutputConvertible, Shape) => OutputConvertible
    ): TensorArray = {
      mapFn(value, shape).asInstanceOf[TensorArray]
    }
  }

  implicit def whileLoopVariableArray[T: ClassTag, TS: ClassTag](implicit ev: Aux[T, TS]): Aux[Array[T], Array[TS]] = {
    new WhileLoopVariable[Array[T]] {
      override type ShapeType = Array[TS]

      override def zero(batchSize: Output, dataType: DataType, shape: Array[TS], name: String): Array[T] = {
        Op.createWithNameScope(name) {
          shape.map(ev.zero(batchSize, dataType, _))
        }
      }

      override def size(output: Array[T]): Int = output.map(ev.size).sum
      override def outputs(output: Array[T]): Seq[Output] = output.toSeq.flatMap(ev.outputs)
      override def shapes(shape: Array[TS]): Seq[Shape] = shape.toSeq.flatMap(ev.shapes)

      override def segmentOutputs(output: Array[T], values: Seq[Output]): (Array[T], Seq[Output]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromOutputs(f._1, f._2)), values.drop(n))
      }

      override def segmentShapes(output: Array[T], values: Seq[Shape]): (Array[TS], Seq[Shape]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromShapes(f._1, f._2)), values.drop(n))
      }

      override def map(value: Array[T], mapFn: OutputConvertible => OutputConvertible): Array[T] = {
        value.map(ev.map(_, mapFn))
      }

      override def mapWithShape(
          value: Array[T],
          shape: Array[TS],
          mapFn: (OutputConvertible, Shape) => OutputConvertible
      ): Array[T] = {
        value.zip(shape).map(p => ev.mapWithShape(p._1, p._2, mapFn))
      }
    }
  }

  implicit def whileLoopVariableSeq[T, TS, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, TS],
      cbfTST: CanBuildFrom[CC[TS], T, CC[T]],
      cbfTT: CanBuildFrom[CC[T], T, CC[T]],
      cbfTTS: CanBuildFrom[CC[T], TS, CC[TS]]
  ): Aux[CC[T], CC[TS]] = {
    new WhileLoopVariable[CC[T]] {
      override type ShapeType = CC[TS]

      override def zero(batchSize: Output, dataType: DataType, shape: CC[TS], name: String): CC[T] = {
        Op.createWithNameScope(name) {
          shape.map(ev.zero(batchSize, dataType, _))(cbfTST)
        }
      }

      override def size(output: CC[T]): Int = output.map(ev.size).sum
      override def outputs(output: CC[T]): Seq[Output] = output.flatMap(ev.outputs).toSeq
      override def shapes(shape: CC[TS]): Seq[Shape] = shape.flatMap(ev.shapes).toSeq

      override def segmentOutputs(output: CC[T], values: Seq[Output]): (CC[T], Seq[Output]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromOutputs(f._1, f._2)).to[CC](cbfTT), values.drop(n))
      }

      override def segmentShapes(output: CC[T], values: Seq[Shape]): (CC[TS], Seq[Shape]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.fromShapes(f._1, f._2)).to[CC](cbfTTS), values.drop(n))
      }

      override def map(value: CC[T], mapFn: OutputConvertible => OutputConvertible): CC[T] = {
        value.map(ev.map(_, mapFn))
      }

      override def mapWithShape(
          value: CC[T],
          shape: CC[TS],
          mapFn: (OutputConvertible, Shape) => OutputConvertible
      ): CC[T] = {
        value.zip(shape.toSeq).map(p => ev.mapWithShape(p._1, p._2, mapFn)).to[CC](cbfTT)
      }
    }
  }

  implicit def whileLoopVariableMap[T, TS, MK](implicit
      ev: Aux[T, TS]
  ): Aux[Map[MK, T], Map[MK, TS]] = {
    new WhileLoopVariable[Map[MK, T]] {
      override type ShapeType = Map[MK, TS]

      override def zero(batchSize: Output, dataType: DataType, shape: Map[MK, TS], name: String): Map[MK, T] = {
        Op.createWithNameScope(name) {
          shape.mapValues(ev.zero(batchSize, dataType, _))
        }
      }

      override def size(output: Map[MK, T]): Int = output.values.map(ev.size).sum
      override def outputs(output: Map[MK, T]): Seq[Output] = output.values.flatMap(ev.outputs).toSeq
      override def shapes(shape: Map[MK, TS]): Seq[Shape] = shape.values.flatMap(ev.shapes).toSeq

      override def segmentOutputs(output: Map[MK, T], values: Seq[Output]): (Map[MK, T], Seq[Output]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(values.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.fromOutputs(f._1, f._2))).toMap, values.drop(n))
      }

      override def segmentShapes(output: Map[MK, T], values: Seq[Shape]): (Map[MK, TS], Seq[Shape]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(values.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.fromShapes(f._1, f._2))).toMap, values.drop(n))
      }

      override def map(value: Map[MK, T], mapFn: OutputConvertible => OutputConvertible): Map[MK, T] = {
        value.mapValues(ev.map(_, mapFn))
      }

      override def mapWithShape(
          value: Map[MK, T],
          shape: Map[MK, TS],
          mapFn: (OutputConvertible, Shape) => OutputConvertible
      ): Map[MK, T] = {
        value.map(p => p._1 -> ev.mapWithShape(p._2, shape(p._1), mapFn))
      }
    }
  }

  implicit val hnil: Aux[HNil, HNil] = new WhileLoopVariable[HNil] {
    override type ShapeType = HNil

    override def zero(batchSize: Output, dataType: DataType, shape: HNil, name: String = "Zero"): HNil = HNil
    override def size(output: HNil): Int = 0
    override def outputs(output: HNil): Seq[Output] = Seq.empty[Output]
    override def shapes(shape: HNil): Seq[Shape] = Seq.empty[Shape]
    override def segmentOutputs(output: HNil, values: Seq[Output]): (HNil, Seq[Output]) = (HNil, values)
    override def segmentShapes(output: HNil, values: Seq[Shape]): (HNil, Seq[Shape]) = (HNil, values)

    override def map(value: HNil, mapFn: OutputConvertible => OutputConvertible): HNil = HNil

    override def mapWithShape(
        value: HNil,
        shape: HNil,
        mapFn: (OutputConvertible, Shape) => OutputConvertible
    ): HNil = HNil
  }

  implicit def recursiveConstructor[H, HS, T <: HList, TS <: HList](implicit
      evHead: Lazy[Aux[H, HS]],
      evTail: Aux[T, TS]
  ): Aux[H :: T, HS :: TS] = new WhileLoopVariable[H :: T] {
    override type ShapeType = HS :: TS

    override def zero(batchSize: Output, dataType: DataType, shape: HS :: TS, name: String = "Zero"): H :: T = {
      Op.createWithNameScope(name) {
        evHead.value.zero(batchSize, dataType, shape.head) :: evTail.zero(batchSize, dataType, shape.tail)
      }
    }

    override def size(output: H :: T): Int = evHead.value.size(output.head) + evTail.size(output.tail)

    override def outputs(output: H :: T): Seq[Output] = {
      evHead.value.outputs(output.head) ++ evTail.outputs(output.tail)
    }

    override def shapes(shape: HS :: TS): Seq[Shape] = {
      evHead.value.shapes(shape.head) ++ evTail.shapes(shape.tail)
    }

    override def segmentOutputs(output: H :: T, values: Seq[Output]): (H :: T, Seq[Output]) = {
      val (headOut, headRemaining) = evHead.value.segmentOutputs(output.head, values)
      val (tailOut, tailRemaining) = evTail.segmentOutputs(output.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def segmentShapes(output: H :: T, values: Seq[Shape]): (HS :: TS, Seq[Shape]) = {
      val (headOut, headRemaining) = evHead.value.segmentShapes(output.head, values)
      val (tailOut, tailRemaining) = evTail.segmentShapes(output.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }

    override def map(value: H :: T, mapFn: OutputConvertible => OutputConvertible): H :: T = {
      evHead.value.map(value.head, mapFn) :: evTail.map(value.tail, mapFn)
    }

    override def mapWithShape(
        value: H :: T,
        shape: HS :: TS,
        mapFn: (OutputConvertible, Shape) => OutputConvertible
    ): H :: T = {
      evHead.value.mapWithShape(value.head, shape.head, mapFn) :: evTail.mapWithShape(value.tail, shape.tail, mapFn)
    }
  }

  implicit def productConstructor[P, PS, L <: HList, LS <: HList](implicit
      genP: Generic.Aux[P, L],
      evL: Aux[L, LS],
      tuplerS: Tupler.Aux[LS, PS],
      tuplerP: Tupler.Aux[L, P],
      genS: Generic.Aux[PS, LS]
  ): Aux[P, PS] = new WhileLoopVariable[P] {
    override type ShapeType = PS

    override def zero(batchSize: Output, dataType: DataType, shape: PS, name: String = "Zero"): P = {
      tuplerP(evL.zero(batchSize, dataType, genS.to(shape), name))
    }

    override def size(output: P): Int = evL.size(genP.to(output))
    override def outputs(output: P): Seq[Output] = evL.outputs(genP.to(output))
    override def shapes(shape: PS): Seq[Shape] = evL.shapes(genS.to(shape))

    override def segmentOutputs(output: P, values: Seq[Output]): (P, Seq[Output]) = {
      val (out, remaining) = evL.segmentOutputs(genP.to(output), values)
      (tuplerP(out), remaining)
    }

    override def segmentShapes(output: P, values: Seq[Shape]): (PS, Seq[Shape]) = {
      val (out, remaining) = evL.segmentShapes(genP.to(output), values)
      (tuplerS(out), remaining)
    }

    override def map(value: P, mapFn: OutputConvertible => OutputConvertible): P = {
      tuplerP(evL.map(genP.to(value), mapFn))
    }

    override def mapWithShape(value: P, shape: PS, mapFn: (OutputConvertible, Shape) => OutputConvertible): P = {
      tuplerP(evL.mapWithShape(genP.to(value), genS.to(shape), mapFn))
    }
  }
}
