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
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.utilities.Collections
import org.platanios.tensorflow.api.utilities.Proto.{Serializable => ProtoSerializable}

import com.google.protobuf.GeneratedMessageV3
import org.tensorflow.framework.{CollectionDef, CondContextDef}
import org.tensorflow.framework.CollectionDef.BytesList
import shapeless._
import shapeless.ops.hlist.Tupler

import scala.collection.{MapLike, SeqLike}
import scala.collection.generic.CanBuildFrom
import scala.collection.JavaConverters._
import scala.language.higherKinds
import scala.reflect.ClassTag

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
          val realInput = add(input)(TF.fromDataType(input.dataType))
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

  override def add[T: TF](output: Output[T]): Output[T] = {
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
        branch.selectSwitchResult(ControlFlow.colocatedSwitch(switchInput, predicate))
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
  private[control_flow] def processOutput[T: TF](
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
      val realValue = branch.selectSwitchResult(ControlFlow.colocatedSwitch(switchInput, predicate))
      externalValues += output.name -> realValue
      realValue.asInstanceOf[Output[T]]
    } else {
      externalValues.getOrElse(output.name, output).asInstanceOf[Output[T]]
    }
  }

  /** Adds the sub-graph constructed by `function` to the current graph. */
  @throws[IllegalArgumentException]
  private[control_flow] def buildCondBranch[T, R](function: () => T)(implicit
      ev: CondOutput.Aux[T, R]
  ): (T, Seq[OutputLike[Any]]) = {
    val originalResult = function()
    if (originalResult == null)
      throw new IllegalArgumentException("The provide cond branch functions must have return values other than 'null'.")
    (originalResult, ev.flatten(ev.processOutput(originalResult, this)))
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
trait CondOutput[T] {
  type ResultType
  def size(output: T): Int
  def processOutput(output: T, context: CondContext): ResultType
  def flatten(processedOutput: ResultType): Seq[OutputLike[Any]]
  def unflatten(output: T, values: Seq[OutputLike[Any]]): T = segment(output, values)._1
  def segment(output: T, values: Seq[OutputLike[Any]]): (T, Seq[OutputLike[Any]])
}

object CondOutput {
  type Aux[T, R] = CondOutput[T] {type ResultType = R}

  implicit def opCondOutput[I, O]: Aux[Op[I, O], Output[Any]] = {
    new CondOutput[Op[I, O]] {
      override type ResultType = Output[Any]

      override def size(output: Op[I, O]): Int = {
        1
      }

      override def processOutput(
          output: Op[I, O],
          context: CondContext
      ): Output[Any] = {
        context.processOp(output)
      }

      override def flatten(
          processedOutput: Output[Any]
      ): Seq[OutputLike[Any]] = {
        Seq(processedOutput)
      }

      override def segment(
          output: Op[I, O],
          values: Seq[OutputLike[Any]]
      ): (Op[I, O], Seq[OutputLike[Any]]) = {
        (values.head.op.asInstanceOf[Op[I, O]], values.tail)
      }
    }
  }

  implicit def outputCondOutput[T]: Aux[Output[T], Output[T]] = {
    new CondOutput[Output[T]] {
      override type ResultType = Output[T]

      override def size(output: Output[T]): Int = {
        1
      }

      override def processOutput(
          output: Output[T],
          context: CondContext
      ): Output[T] = {
        context.processOutput(output)(TF.fromDataType(output.dataType))
      }
      override def flatten(
          processedOutput: Output[T]
      ): Seq[OutputLike[Any]] = {
        Seq(processedOutput)
      }

      override def segment(
          output: Output[T],
          values: Seq[OutputLike[Any]]
      ): (Output[T], Seq[OutputLike[Any]]) = {
        (values.head.asInstanceOf[Output[T]], values.tail)
      }
    }
  }

  implicit def outputIndexedSlicesCondOutput[T]: Aux[OutputIndexedSlices[T], OutputIndexedSlices[T]] = {
    new CondOutput[OutputIndexedSlices[T]] {
      override type ResultType = OutputIndexedSlices[T]

      override def size(output: OutputIndexedSlices[T]): Int = {
        1
      }

      override def processOutput(
          output: OutputIndexedSlices[T],
          context: CondContext
      ): OutputIndexedSlices[T] = {
        val indices = context.processOutput(output.indices)
        val values = context.processOutput(output.values)(TF.fromDataType(output.values.dataType))
        val denseShape = {
          if (output.denseShape != null)
            context.processOutput(output.denseShape)
          else
            null
        }
        OutputIndexedSlices(
          indices = indices,
          values = values,
          denseShape = denseShape)
      }

      override def flatten(
          processedOutput: OutputIndexedSlices[T]
      ): Seq[OutputLike[Any]] = {
        Seq(processedOutput)
      }

      override def segment(
          output: OutputIndexedSlices[T],
          values: Seq[OutputLike[Any]]
      ): (OutputIndexedSlices[T], Seq[OutputLike[Any]]) = {
        (values.head.asInstanceOf[OutputIndexedSlices[T]], values.tail)
      }
    }
  }

  implicit def sparseOutputCondOutput[T]: Aux[SparseOutput[T], SparseOutput[T]] = {
    new CondOutput[SparseOutput[T]] {
      override type ResultType = SparseOutput[T]

      override def size(output: SparseOutput[T]): Int = {
        1
      }

      override def processOutput(
          output: SparseOutput[T],
          context: CondContext
      ): SparseOutput[T] = {
        val indices = context.processOutput(output.indices)
        val values = context.processOutput(output.values)(TF.fromDataType(output.values.dataType))
        val denseShape = {
          if (output.denseShape != null)
            context.processOutput(output.denseShape)
          else
            null
        }
        SparseOutput(
          indices = indices,
          values = values,
          denseShape = denseShape)
      }

      override def flatten(
          processedOutput: SparseOutput[T]
      ): Seq[OutputLike[Any]] = {
        Seq(processedOutput)
      }

      override def segment(
          output: SparseOutput[T],
          values: Seq[OutputLike[Any]]
      ): (SparseOutput[T], Seq[OutputLike[Any]]) = {
        (values.head.asInstanceOf[SparseOutput[T]], values.tail)
      }
    }
  }

  implicit def tensorArrayCondOutput[T]: Aux[TensorArray[T], Output[Float]] = {
    new CondOutput[TensorArray[T]] {
      override type ResultType = Output[Float]

      override def size(output: TensorArray[T]): Int = {
        1
      }

      override def processOutput(
          output: TensorArray[T],
          context: CondContext
      ): Output[Float] = {
        context.processOutput(output.flow)
      }

      override def flatten(
          processedOutput: Output[Float]
      ): Seq[OutputLike[Any]] = {
        Seq(processedOutput)
      }

      override def segment(
          output: TensorArray[T],
          values: Seq[OutputLike[Any]]
      ): (TensorArray[T], Seq[OutputLike[Any]]) = {
        val newTensorArray = output.copy(
          flow = values.head.asInstanceOf[Output[Float]])
        // TODO: !!! [TENSOR_ARRAY] What about colocate with?
        (newTensorArray, values.tail)
      }
    }
  }

  implicit def condOutputArray[T: ClassTag, R: ClassTag](implicit ev: Aux[T, R]): Aux[Array[T], Array[R]] = {
    new CondOutput[Array[T]] {
      override type ResultType = Array[R]

      override def size(output: Array[T]): Int = {
        output.map(ev.size).sum
      }

      override def processOutput(
          output: Array[T],
          context: CondContext
      ): Array[R] = {
        output.map(ev.processOutput(_, context))
      }

      override def flatten(
          processedOutput: Array[R]
      ): Seq[OutputLike[Any]] = {
        processedOutput.toSeq.flatMap(ev.flatten)
      }

      override def segment(
          output: Array[T],
          values: Seq[OutputLike[Any]]
      ): (Array[T], Seq[OutputLike[Any]]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.unflatten(f._1, f._2)), values.drop(n))
      }
    }
  }

  implicit def condOutputSeq[T, R, CC[A] <: SeqLike[A, CC[A]]](implicit
      ev: Aux[T, R],
      cbfTT: CanBuildFrom[CC[T], T, CC[T]],
      cbfTR: CanBuildFrom[CC[T], R, CC[R]]
  ): Aux[CC[T], CC[R]] = {
    new CondOutput[CC[T]] {
      override type ResultType = CC[R]

      override def size(output: CC[T]): Int = {
        output.map(ev.size).sum
      }

      override def processOutput(
          output: CC[T],
          context: CondContext
      ): CC[R] = {
        output.map(ev.processOutput(_, context))(cbfTR)
      }

      override def flatten(
          processedOutput: CC[R]
      ): Seq[OutputLike[Any]] = {
        processedOutput.flatMap(ev.flatten).toSeq
      }

      override def segment(
          output: CC[T],
          values: Seq[OutputLike[Any]]
      ): (CC[T], Seq[OutputLike[Any]]) = {
        val n = size(output)
        (output.zip(Collections.segment(values.take(n), output.map(ev.size).toSeq))
            .map(f => ev.unflatten(f._1, f._2)).to[CC](cbfTT), values.drop(n))
      }
    }
  }

  implicit def condOutputMap[T, R, MK, CC[K, V] <: MapLike[K, V, CC[K, V]] with Map[K, V]](implicit
      ev: Aux[T, R]
  ): Aux[Map[MK, T], Map[MK, R]] = {
    new CondOutput[Map[MK, T]] {
      override type ResultType = Map[MK, R]

      override def size(output: Map[MK, T]): Int = {
        output.values.map(ev.size).sum
      }

      override def processOutput(
          output: Map[MK, T],
          context: CondContext
      ): Map[MK, R] = {
        output.map({ case (k, v) => k -> ev.processOutput(v, context) })
      }

      override def flatten(
          processedOutput: Map[MK, R]
      ): Seq[OutputLike[Any]] = {
        processedOutput.values.flatMap(ev.flatten).toSeq
      }

      override def segment(
          output: Map[MK, T],
          values: Seq[OutputLike[Any]]
      ): (Map[MK, T], Seq[OutputLike[Any]]) = {
        val n = size(output)
        (output.keys.zip(
          output.values
              .zip(Collections.segment(values.take(n), output.values.map(ev.size).toSeq))
              .map(f => ev.unflatten(f._1, f._2))).toMap, values.drop(n))
      }
    }
  }

  implicit val hnil: Aux[HNil, HNil] = new CondOutput[HNil] {
    override type ResultType = HNil

    override def size(output: HNil): Int = {
      0
    }

    override def processOutput(
        output: HNil,
        context: CondContext
    ): HNil = {
      HNil
    }

    override def flatten(
        processedOutput: HNil
    ): Seq[OutputLike[Any]] = {
      Seq.empty[OutputLike[Any]]
    }

    override def segment(
        output: HNil,
        values: Seq[OutputLike[Any]]
    ): (HNil, Seq[OutputLike[Any]]) = {
      (HNil, values)
    }
  }

  implicit def recursiveConstructor[H, HR, T <: HList, TR <: HList](implicit
      evHead: Lazy[Aux[H, HR]],
      evTail: Aux[T, TR]
  ): Aux[H :: T, HR :: TR] = new CondOutput[H :: T] {
    override type ResultType = HR :: TR

    override def size(output: H :: T): Int = {
      evHead.value.size(output.head) + evTail.size(output.tail)
    }

    override def processOutput(
        output: H :: T,
        context: CondContext
    ): HR :: TR = {
      evHead.value.processOutput(output.head, context) ::
          evTail.processOutput(output.tail, context)
    }

    override def flatten(
        processedOutput: HR :: TR
    ): Seq[OutputLike[Any]] = {
      evHead.value.flatten(processedOutput.head) ++
          evTail.flatten(processedOutput.tail)
    }

    override def segment(
        output: H :: T,
        values: Seq[OutputLike[Any]]
    ): (H :: T, Seq[OutputLike[Any]]) = {
      val (headOut, headRemaining) = evHead.value.segment(output.head, values)
      val (tailOut, tailRemaining) = evTail.segment(output.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }
  }

  implicit def productConstructor[P, R, L <: HList, LR <: HList](implicit
      genP: Generic.Aux[P, L],
      evL: Aux[L, LR],
      tuplerR: Tupler.Aux[LR, R],
      tuplerP: Tupler.Aux[L, P],
      genR: Generic.Aux[R, LR]
  ): Aux[P, R] = new CondOutput[P] {
    override type ResultType = R

    override def size(output: P): Int = {
      evL.size(genP.to(output))
    }

    override def processOutput(
        output: P,
        context: CondContext
    ): R = {
      tuplerR(evL.processOutput(genP.to(output), context))
    }

    override def flatten(
        processedOutput: R
    ): Seq[OutputLike[Any]] = {
      evL.flatten(genR.to(processedOutput))
    }

    override def segment(
        output: P,
        values: Seq[OutputLike[Any]]
    ): (P, Seq[OutputLike[Any]]) = {
      val (out, remaining) = evL.segment(genP.to(output), values)
      (tuplerP(out), remaining)
    }
  }
}
