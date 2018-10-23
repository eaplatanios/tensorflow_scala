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

import org.platanios.tensorflow.api.core.Graph
import org.platanios.tensorflow.api.core.types.TF
import org.platanios.tensorflow.api.implicits.helpers.OutputStructure
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import com.google.protobuf.GeneratedMessageV3
import org.tensorflow.framework.{CollectionDef, CondContextDef}
import org.tensorflow.framework.CollectionDef.BytesList

import scala.collection.JavaConverters._
import scala.language.higherKinds

/** Control flow context for the conditional construct.
  *
  * @param  predicate Scalar tensor for the conditional predicate.
  * @param  pivot     Predicate tensor for the current branch.
  * @param  branch    Current branch (i.e., `TrueBranch` or `FalseBranch`).
  * @param  _name     Name prefix for this conditional context.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] case class CondContext private[control_flow] (
    predicate: Output[Boolean],
    pivot: Output[Boolean],
    branch: CondBranch,
    private val _name: String = "CondContext"
) extends Context() with ProtoSerializable {
  values += predicate.name
  values += pivot.name

  pivot.op.controlFlowContext = Some(this)

  override val name: String = {
    Op.currentGraph.uniqueName(_name)
  }

  override def controlPivot: Option[UntypedOp] = {
    Some(pivot.op)
  }

  override def condContext: Option[CondContext] = {
    Some(this)
  }

  override def add(op: UntypedOp): Unit = {
    addInternal(op)
  }

  override private[control_flow] def addInternal(op: UntypedOp): Unit = {
    if (op.numInputs == 0) {
      // Remove any external control dependencies on this op.
      removeExternalControlEdges(op)
      controlPivot.foreach(ControlFlow.addControlInput(op, _))
    } else {
      op.inputsSeq.zipWithIndex.foreach({
        case (input, index) =>
          val realInput = add(input)
          if (realInput != input)
            ControlFlow.updateInput(op, index, realInput)
      })
      // Remove any external control dependencies on this op.
      removeExternalControlEdges(op)
      if (op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient")
        controlPivot.foreach(ControlFlow.addControlInput(op, _))
    }
    val outputNames = op.outputsSeq.map(_.name)
    var context: Option[Context] = Some(this)
    while (context.isDefined) {
      context.foreach(_.values ++= outputNames)
      context = context.flatMap(_.outerContext)
    }
    if (outerContext.isDefined || !ControlFlow.isLoopExit(op))
      op.graph.preventFetching(op)
  }

  override def add[T](output: Output[T]): Output[T] = {
    if (values.contains(output.name)) {
      // Use the real value if it comes from an outer context. This is needed in particular for nested conditionals.
      externalValues.getOrElse(output.name, output).asInstanceOf[Output[T]]
    } else {
      values += output.name
      val switchInput = outerContext.map(c => {
        val i = c.add(output)
        values += i.name
        i
      }).getOrElse(output)
      val result = Op.createWith(controlDependencies = Set.empty) {
        branch.selectSwitchResult(
          ControlFlow.colocatedSwitch(switchInput, predicate)(TF.fromDataType(switchInput.dataType)))
      }
      result.graph.preventFetching(result.op)
      result.op.controlFlowContext = Some(this)
      values += result.name
      externalValues += output.name -> result
      result
    }
  }

  override def backPropagate: Boolean = {
    whileLoopContext().exists(_.backPropagate)
  }

  override def gradientLoopState: Option[GradientLoopState] = {
    whileLoopContext().flatMap(_.gradientLoopState)
  }

  /** Processes an op used in a conditional branch. */
  private[control_flow] def processOp(op: UntypedOp): Output[Boolean] = {
    // Use pivot as the proxy for this op.
    ControlFlow.withControlDependencies(Set(op), pivot)
  }

  /** Processes an op output used in a conditional branch. */
  private[control_flow] def processOutput[T](
      output: Output[T]
  ): Output[T] = {
    if (!values.contains(output.name)) {
      // Handle the special case of () => x.
      values += output.name
      val switchInput = outerContext.map(c => {
        val i = c.add(output)
        values += i.name
        i
      }).getOrElse(output)
      val realValue = branch.selectSwitchResult(
        ControlFlow.colocatedSwitch(switchInput, predicate)(TF.fromDataType(switchInput.dataType)))
      externalValues += output.name -> realValue
      realValue.asInstanceOf[Output[T]]
    } else {
      externalValues.getOrElse(output.name, output).asInstanceOf[Output[T]]
    }
  }

  /** Adds the sub-graph constructed by `function` to the current graph. */
  @throws[IllegalArgumentException]
  private[control_flow] def buildCondBranch[T](function: () => T)(implicit
      evCondArgT: CondArg[T]
  ): (T, Seq[Output[Any]]) = {
    val originalResult = function()
    if (originalResult == null)
      throw new IllegalArgumentException("The provide cond branch functions must have return values other than 'null'.")
    (originalResult, evCondArgT.outputs(originalResult, context = this))
  }

  override def toProto: GeneratedMessageV3 = {
    toProto(null)
  }

  /** Alias for `toCondContextDef`. */
  override def toProto(exportScope: String = null): GeneratedMessageV3 = {
    toCondContextDef(exportScope)
  }

  /** Constructs and returns a [[CondContextDef]] object that represents this cond context.
    *
    * @param  exportScope Optional string specifying the name scope to remove. Only the ops within this name scope will
    *                     be included in the resulting ProtoBuf object and the export scope will be stripped from their
    *                     names to allow for easy import into new name scopes.
    * @return Constructed [[CondContextDef]].
    */
  def toCondContextDef(exportScope: String = null): CondContextDef = {
    if (exportScope == null || name.startsWith(exportScope)) {
      CondContextDef.newBuilder()
          .setContextName(Op.stripNameScope(exportScope, name))
          .setPredName(Op.stripNameScope(exportScope, predicate.name))
          .setPivotName(Op.stripNameScope(exportScope, pivot.name))
          .setBranch(branch.value)
          .setValuesDef(super.toValuesDef(exportScope))
          .build()
    } else {
      null
    }
  }
}

object CondContext {
  /** Creates a [[CondContext]] from the provided [[CondContextDef]] object.
    *
    * @param  condContextDef Serialized cond context object.
    * @param  importScope    Optional prefix that will be prepended to all op names in the cond context that is being
    *                        loaded from the provided [[CondContextDef]].
    * @return Constructed [[CondContext]].
    */
  def fromCondContextDef(condContextDef: CondContextDef, importScope: String = null): CondContext = {
    val graph = Op.currentGraph
    val name = Op.prependNameScope(importScope, condContextDef.getContextName)
    val predicate = graph.getOutputByName(Op.prependNameScope(importScope, condContextDef.getPredName))
        .asInstanceOf[Output[Boolean]]
    val pivot = graph.getOutputByName(Op.prependNameScope(importScope, condContextDef.getPivotName))
        .asInstanceOf[Output[Boolean]]
    val branch = CondBranch.fromValue(condContextDef.getBranch)
    val (values, externalValues) = Context.fromValuesDef(condContextDef.getValuesDef, importScope)
    val condContext = CondContext(predicate, pivot, branch, name)
    condContext.values ++= values
    condContext.externalValues ++= externalValues
    condContext
  }

  /** Key for collections of [[CondContext]]s. */
  trait CollectionKey extends Graph.Key[CondContext] {
    override def createCollectionDef(values: Set[CondContext], exportScope: String = null): CollectionDef = {
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
          .foreach(s => graph.addToCollection(this)(
            CondContext.fromCondContextDef(CondContextDef.parseFrom(s), importScope)))
    }
  }

  /** Key to collect the [[CondContext]]s that have been created in the graph. */
  object COND_CONTEXTS extends CollectionKey {
    override def name: String = "cond_context"
  }
}

/** Represents a branch of a conditional control flow construct (i.e., true/false branch). */
sealed trait CondBranch {
  private[control_flow] val value: Int
  private[control_flow] val other: CondBranch
  private[control_flow] def selectSwitchResult[T, OL[A] <: OutputLike[A]](switchResult: (OL[T], OL[T])): OL[T]
}

case object TrueBranch extends CondBranch {
  override private[control_flow] val value: Int = 1
  override private[control_flow] val other: CondBranch = FalseBranch
  override private[control_flow] def selectSwitchResult[T, OL[A] <: OutputLike[A]](
      switchResult: (OL[T], OL[T])
  ): OL[T] = {
    switchResult._2
  }
}

case object FalseBranch extends CondBranch {
  override private[control_flow] val value: Int        = 0
  override private[control_flow] val other: CondBranch = TrueBranch
  override private[control_flow] def selectSwitchResult[T, OL[A] <: OutputLike[A]](
      switchResult: (OL[T], OL[T])
  ): OL[T] = {
    switchResult._1
  }
}

object CondBranch {
  private[control_flow] def fromValue(value: Int): CondBranch = value match {
    case TrueBranch.value => TrueBranch
    case FalseBranch.value => FalseBranch
    case _ => throw new IllegalArgumentException(s"Cond branch value was $value. Supported values are only 0 and 1.")
  }
}

/** Type trait used for representing supported conditional construct branch function return types. */
trait CondArg[T] {
  def outputs(output: T, context: CondContext): Seq[Output[Any]]
  def decodeOutputFromOutput(output: T, outputs: Seq[Output[Any]]): (T, Seq[Output[Any]])
}

object CondArg extends CondArgLowPriority {
  implicit val evCondArgString: CondArg[Output[String]] = {
    fromNestedStructure[Output[String]]
  }

  implicit val evCondArgLong: CondArg[Output[Long]] = {
    fromNestedStructure[Output[Long]]
  }

  implicit val evCondArgFloat: CondArg[Output[Float]] = {
    fromNestedStructure[Output[Float]]
  }

  implicit val evCondArgUntyped: CondArg[Output[Any]] = {
    fromNestedStructure[Output[Any]]
  }

  implicit val evCondArgSeqUntyped: CondArg[Seq[Output[Any]]] = {
    fromNestedStructure[Seq[Output[Any]]]
  }

  implicit val evCondArgOptionSeqUntyped: CondArg[Option[Seq[Output[Any]]]] = {
    fromNestedStructure[Option[Seq[Output[Any]]]]
  }

  def apply[T: CondArg]: CondArg[T] = {
    implicitly[CondArg[T]]
  }
}

trait CondArgLowPriority {
  implicit def fromNestedStructure[T](implicit ev: OutputStructure[T]): CondArg[T] = {
    new CondArg[T] {
      override def outputs(
          output: T,
          context: CondContext
      ): Seq[Output[Any]] = {
        ev.outputs(output).map(context.processOutput)
      }

      override def decodeOutputFromOutput(
          output: T,
          outputs: Seq[Output[Any]]
      ): (T, Seq[Output[Any]]) = {
        ev.decodeOutput(output, outputs)
      }
    }
  }

  // TODO: [IMPLICITS] !!! What about ops appearing in nested sequences?

  implicit def fromOp[I, O]: CondArg[Op[I, O]] = {
    new CondArg[Op[I, O]] {
      override def outputs(
          output: Op[I, O],
          context: CondContext
      ): Seq[Output[Any]] = {
        Seq(context.processOp(output))
      }

      override def decodeOutputFromOutput(
          output: Op[I, O],
          outputs: Seq[Output[Any]]
      ): (Op[I, O], Seq[Output[Any]]) = {
        (outputs.head.op.asInstanceOf[Op[I, O]], outputs.tail)
      }
    }
  }
}
