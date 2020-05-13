/* Copyright 2017-19, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.ProtoSerializable
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.basic.Basic
import org.platanios.tensorflow.proto.ValuesDef

import com.google.protobuf.GeneratedMessageV3

import scala.collection.mutable
import scala.jdk.CollectionConverters._

/** Base class for control flow contexts.
  *
  * The usage pattern is a sequence of (Enter, Exit) ops followed by a final ExitResult op.
  *
  * We maintain the following state for control flow contexts during graph construction:
  *   1. [[GraphConstructionScope]] has a `controlFlowContext` field, which represents the current control flow context
  *      used to construct new ops. It can be changed by `context.enter()` and `context.exit()`.
  *   2. Each op has a `controlFlowContext` field, which is the control flow context to which the op belongs. It is set
  *      at the time the op is created and it is immutable.
  *   3. Each [[Context]] has an `outerContext` field, which is the control flow context in which this context is
  *      created. It is set at the time a context is created and it is immutable.
  *   4. Each [[Context]] has a `contextStack` field, which contains the control flow contexts pushed and popped by
  *      `context.enter()` and `context.exit()`.
  *
  * @param  values         Set of values that have already been seen in this context.
  * @param  externalValues Set of values referenced by but external to this context.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class Context protected (
    private[control_flow] val values: mutable.Set[String] = mutable.Set.empty,
    private[control_flow] val externalValues: mutable.Map[String, Output[Any]] = mutable.Map.empty
) extends ProtoSerializable {
  values.diff(externalValues.keySet)
      .map(_.split(":")(0))
      .map(Op.currentGraph.getOpByName(_))
      .foreach(_.controlFlowContext = Some(this))

  /** Name of this control flow context. */
  val name: String

  /** Contains the stack of control flow contexts that have been entered so far. */
  private[this] val contextStack: mutable.ListBuffer[Option[Context]] = {
    mutable.ListBuffer.empty
  }

  /** Control flow context containing this context. */
  val outerContext: Option[Context] = {
    Op.currentControlFlowContext
  }

  /** Returns the control pivot op output for this context. */
  def controlPivot: Option[UntypedOp] = {
    None
  }

  /** Returns the cond context containing this context. */
  def condContext: Option[CondContext] = {
    outerContext.flatMap(_.condContext)
  }

  /** Returns the while context containing this context. */
  def whileLoopContext(
      stopContext: Option[Context] = None
  ): Option[WhileLoopContext] = {
    stopContext match {
      case Some(context) if context == this => Some(context.asInstanceOf[WhileLoopContext])
      case None => outerContext.flatMap(_.whileLoopContext(stopContext))
    }
  }

  /** Returns the XLA context containing this context. */
  def xlaContext: Option[XLAControlFlowContext] = {
    outerContext.flatMap(_.xlaContext)
  }

  /** Adds `op` to the current context. */
  def add(op: UntypedOp): Unit = {
    addInternal(op)
  }

  /** Adds `op` to the current context. We move any external control dependencies of the op to the control flow pivot,
    * to ensure they get executed. */
  private[control_flow] def addInternal(op: UntypedOp): Unit = {
    val externalInputs = {
      if (op.numInputs == 0) {
        // Remove any external control dependencies on this op.
        val (controlInputs, externalInputs) = removeExternalControlEdges(op)
        // Add a control edge from the control pivot to this op.
        if (controlInputs.isEmpty)
          controlPivot.foreach(ControlFlow.addControlInput(op, _))
        op.outputsSeq.foreach(values += _.name)
        externalInputs
      } else {
        op.inputsSeq.zipWithIndex.foreach({
          case (input, index) =>
            val realInput = add(input)
            if (realInput != input)
              ControlFlow.updateInput(op, index, realInput)
        })
        // Remove any external control dependencies on this op.
        val (_, externalInputs) = removeExternalControlEdges(op)
        // Add a control dependency to the op if it only depends on loop invariants. That is to prevent loop invariants
        // from enabling ops that should not be executed.
        if (op.controlInputs.isEmpty &&
            ((op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient") ||
                op.inputsSeq.forall(o => ControlFlow.isLoopConstantEnter(o.op))))
          controlPivot.foreach(ControlFlow.addControlInput(op, _))
        op.outputsSeq.foreach(values += _.name)
        externalInputs
      }
    }
    // TODO: [CONTROL_FLOW] Stop ignoring ops with no outputs.
    // Use an identity to pull control inputs as data inputs. Note that we ignore ops which do not have any outputs.
    Op.createWith(controlDependencies = Set.empty) {
      enter()
      externalInputs
          .filter(_.outputsSeq.nonEmpty)
          .map(op => {
            val output = op.outputsSeq(0)
            Basic.identity(output)(TF.fromDataType(output.dataType)).op
          })
          .foreach(ControlFlow.addControlInput(op, _))
      exit()
    }
    if (outerContext.isDefined || !ControlFlow.isLoopExit(op)) {
      op.graph.preventFetching(op)
      op.outputsSeq.foreach(op.graph.preventFeeding)
    }
  }

  /** Adds `output` to the current context and its outer context recursively. */
  def add[T](output: Output[T]): Output[T]

  /** Returns `true` if back-propagation is supported for this control flow context. */
  def backPropagate: Boolean

  /** Gradient loop state for this context, used for back-propagation. */
  def gradientLoopState: Option[GradientLoopState]

  /** Enters this control flow context. */
  def enter(): Unit = {
    contextStack.append(graphConstructionScope.value.controlFlowContext)
    graphConstructionScope.value = graphConstructionScope.value.copy(
      controlFlowContext = Some(this),
      outerContext = Some(graphConstructionScope.value))
  }

  /** Exits this control flow context. */
  def exit(): Unit = {
    graphConstructionScope.value = graphConstructionScope.value.copy(
      controlFlowContext = contextStack.remove(contextStack.size - 1),
      outerContext = Some(graphConstructionScope.value))
  }

  /** Makes a sequence of tensors available in the outer context. */
  def exitResult(result: Seq[OutputLike[Any]]): Unit = {
    outerContext.foreach(c => result.foreach(r => c.values += r.name))
  }

  /** Enters a control flow context for building a gradient colocated with `colocationOps`. */
  def enterGradientColocation(
      colocationOps: Set[UntypedOp],
      gradientUID: String
  ): Unit = {
    outerContext.foreach(_.enterGradientColocation(colocationOps, gradientUID))
  }

  /** Exits a control flow context for building a gradient colocated with `colocationOps`. */
  def exitGradientColocation(
      colocationOps: Set[UntypedOp],
      gradientUID: String
  ): Unit = {
    outerContext.foreach(_.exitGradientColocation(colocationOps, gradientUID))
  }

  /** Removes any external control dependency on this op and returns the remaining internal control inputs and any
    * external control inputs that were removed. */
  private[control_flow] def removeExternalControlEdges(
      op: UntypedOp
  ): (Set[UntypedOp], Set[UntypedOp]) = {
    // A control input of 'op' is internal if it is in the same while loop context as the enclosing while loop context
    // of this context.
    val internalControlInputs = this.whileLoopContext() match {
      case None => op.controlInputs
      case Some(wC) =>
        op.controlInputs.filter(i => ControlFlow.getOutputContext(i).exists(_.whileLoopContext().contains(wC)))
    }
    val externalControlInputs = {
      if (internalControlInputs.size != op.controlInputs.size) {
        val externalControlInputs = op.controlInputs -- internalControlInputs
        ControlFlow.clearControlInputs(op)
        internalControlInputs.foreach(ControlFlow.addControlInput(op, _))
        externalControlInputs
      } else {
        Set.empty[UntypedOp]
      }
    }
    (internalControlInputs, externalControlInputs)
  }

  override def toProto: GeneratedMessageV3 = {
    toProto(null)
  }

  /** Alias for `toValuesDef`. */
  def toProto(exportScope: String): GeneratedMessageV3 = {
    toValuesDef(exportScope)
  }

  /** Constructs and returns a [[ValuesDef]] object that represents this control flow context.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[ValuesDef]].
    */
  def toValuesDef(exportScope: String = null): ValuesDef = {
    Context.toValuesDef(values, externalValues, exportScope)
  }
}

object Context {
  /** Constructs and returns a [[ValuesDef]] object that represents the provided values.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[ValuesDef]].
    */
  def toValuesDef(
      values: mutable.Set[String],
      externalValues: mutable.Map[String, Output[Any]],
      exportScope: String = null
  ): ValuesDef = {
    val valuesDefBuilder = ValuesDef.newBuilder()
    values.foreach(v => valuesDefBuilder.addValues(Op.stripNameScope(exportScope, v)))
    externalValues.foreach(p => valuesDefBuilder.putExternalValues(
      Op.stripNameScope(exportScope, p._1), Op.stripNameScope(exportScope, p._2.name)))
    valuesDefBuilder.build()
  }

  /** Returns a set of values and a map of external values loaded from the provided [[ValuesDef]] object.
    *
    * @param  valuesDef   Serialized control flow context object.
    * @param  importScope Optional prefix that will be prepended to all op names in the objects that are being loaded
    *                     from the provided [[ValuesDef]].
    * @return Tuple containing the loaded values and external values that can be used to create control flow contexts.
    */
  def fromValuesDef(
      valuesDef: ValuesDef,
      importScope: String = null
  ): (mutable.Set[String], mutable.Map[String, Output[Any]]) = {
    val values = valuesDef.getValuesList.asScala.toSet[String].map(v => Op.prependNameScope(importScope, v))
    val externalValues = valuesDef.getExternalValuesMap.asScala.map {
      case (k, v) =>
        val key = Op.prependNameScope(importScope, k)
        val value = Op.currentGraph.getOutputByName(Op.prependNameScope(importScope, v))
        (key, value)
    }
    (mutable.Set(values.toSeq: _*), externalValues)
  }

  /** Create a `zerosLike` op for the specified op output, while taking into account control flow contexts. */
  private[ops] def zerosLikeOutsideLoop(
      op: UntypedOp,
      index: Int
  ): Output[Any] = {
    val dataType = op.outputsSeq(index).dataType
    if (ControlFlow.isSwitch(op)) {
      op.controlFlowContext.filter(_.isInstanceOf[CondContext]).map(c => {
        val condContext = c.asInstanceOf[CondContext]
        // We are in a conditional context and so we use a switch to create zeros only when needed.
        val switch = condContext.branch.other.selectSwitchResult(
          ControlFlow.switch(op.inputsSeq(0), condContext.predicate)(TF.fromDataType(dataType)))
        Basic.zeros(dataType, Basic.shape(switch, optimize = false)(TF.fromDataType(dataType)))
      }).getOrElse(Basic.zerosLike(op.outputsSeq(index), optimize = false))
    } else {
      Basic.zerosLike(op.outputsSeq(index), optimize = false)
    }
  }
}

/** Base class for XLA and TPU control flow contexts.
  *
  * @param  values         Set of values that have already been seen in this context.
  * @param  externalValues Set of values referenced by but external to this context.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class XLAControlFlowContext protected (
    override private[control_flow] val values: mutable.Set[String] = mutable.Set.empty,
    override private[control_flow] val externalValues: mutable.Map[String, Output[Any]] = mutable.Map.empty
) extends Context(values, externalValues) {
  override def xlaContext: Option[XLAControlFlowContext] = {
    Some(this)
  }
}
