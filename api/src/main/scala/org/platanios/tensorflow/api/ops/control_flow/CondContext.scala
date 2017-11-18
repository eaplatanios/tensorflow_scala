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

import org.platanios.tensorflow.api.core.Graph
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
  * @param  predicate `BOOLEAN` scalar tensor for the conditional predicate.
  * @param  pivot     Predicate tensor for the current branch.
  * @param  branch    Current branch (i.e., `TrueBranch` or `FalseBranch`).
  * @param  _name     Name prefix for this conditional context.
  *
  * @author Emmanouil Antonios Platanios
  */
private[api] case class CondContext private[control_flow] (
    predicate: Output, pivot: Output, branch: CondBranch, private val _name: String = "CondContext"
) extends Context() with ProtoSerializable {
  values += predicate.name
  values += pivot.name

  override val name: String = Op.currentGraph.uniqueName(_name)

  override def controlPivot: Option[Op] = Some(pivot.op)

//  override def add(op: Op): Unit = addInternal(op)
//
//  override private[control_flow] def addInternal(op: Op): Unit = {
//    if (op.numInputs == 0) {
//      // Remove any external control dependencies on this op.
//      removeExternalControlEdges(op)
//      controlPivot.foreach(ControlFlow.addControlInput(op, _))
//    } else {
//      op.inputs.zipWithIndex.foreach({
//        case (input, index) =>
//          val realInput = add(input)
//          if (realInput != input)
//            ControlFlow.updateInput(op, index, realInput)
//      })
//      // Remove any external control dependencies on this op.
//      removeExternalControlEdges(op)
//      if (op.graph.isFunction(op.opType) || op.opType == "SymbolicGradient")
//        controlPivot.foreach(ControlFlow.addControlInput(op, _))
//    }
//    op.outputs.foreach(values += _.name)
//    if (outerContext.isDefined || !ControlFlow.isLoopExit(op))
//      op.graph.preventFetching(op)
//  }

  override def add(output: Output): Output = {
    if (values.contains(output.name)) {
      // Use the real value if it comes from an outer context. This is needed in particular for nested conditionals.
      externalValues.getOrElse(output.name, output)
    } else {
      values += output.name
      var result = output
      outerContext.foreach(c => {
        result = c.add(output)
        values += result.name
      })
      Op.createWith(controlDependencies = Set.empty[Op]) {
        result = branch.selectSwitchResult(ControlFlow.colocatedSwitch(result, predicate))
      }
      result.graph.preventFetching(result.op)
      result.op.controlFlowContext = Some(this)
      values += result.name
      externalValues += output.name -> result
      result
    }
  }

  override def backPropagate: Boolean = whileLoopContext.exists(_.backPropagate)

  override def gradientLoopState: Option[GradientLoopState] = whileLoopContext.flatMap(_.gradientLoopState)

  /** Processes an op used in a conditional branch. */
  private[control_flow] def processOp(op: Op): Output = {
    // Use pivot as the proxy for this op.
    ControlFlow.withControlDependencies(Set(op), pivot)
  }

  /** Processes an op output used in a conditional branch. */
  private[control_flow] def processOutput(output: Output): Output = {
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
      realValue
    } else {
      externalValues.getOrElse(output.name, output)
    }
  }

  /** Adds the sub-graph constructed by `function` to the current graph. */
  @throws[IllegalArgumentException]
  private[control_flow] def buildCondBranch[T, R](function: () => T)(implicit
      ev: CondOutput.Aux[T, R]
  ): (T, Seq[OutputLike]) = {
    val originalResult = function()
    if (originalResult == null)
      throw new IllegalArgumentException("The provide cond branch functions must have return values other than 'null'.")
    (originalResult, ev.flatten(ev.processOutput(originalResult, this)))
  }

  override def toProto: GeneratedMessageV3 = toProto(null)

  /** Alias for `toCondContextDef`. */
  override def toProto(exportScope: String = null): GeneratedMessageV3 = toCondContextDef(exportScope)

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
    val pivot = graph.getOutputByName(Op.prependNameScope(importScope, condContextDef.getPivotName))
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
          .foreach(s => graph.addToCollection(
            CondContext.fromCondContextDef(CondContextDef.parseFrom(s), importScope), this))
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
  private[control_flow] def selectSwitchResult[T <: OutputLike](switchResult: (T, T)): T
}

case object TrueBranch extends CondBranch {
  override private[control_flow] val value: Int = 1
  override private[control_flow] val other: CondBranch = FalseBranch
  override private[control_flow] def selectSwitchResult[T <: OutputLike](switchResult: (T, T)): T = switchResult._2
}

case object FalseBranch extends CondBranch {
  override private[control_flow] val value: Int = 0
  override private[control_flow] val other: CondBranch = TrueBranch
  override private[control_flow] def selectSwitchResult[T <: OutputLike](switchResult: (T, T)): T = switchResult._1
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
  def flatten(processedOutput: ResultType): Seq[OutputLike]
  def unflatten(output: T, values: Seq[OutputLike]): T = segment(output, values)._1
  def segment(output: T, values: Seq[OutputLike]): (T, Seq[OutputLike])
}

object CondOutput {
  type Aux[T, R] = CondOutput[T] {type ResultType = R}

  implicit val opCondOutput: Aux[Op, Output] = new CondOutput[Op] {
    override type ResultType = Output
    override def size(output: Op): Int = 1
    override def processOutput(output: Op, context: CondContext): Output = context.processOp(output)
    override def flatten(processedOutput: Output): Seq[OutputLike] = Seq(processedOutput)
    override def segment(output: Op, values: Seq[OutputLike]): (Op, Seq[OutputLike]) = (values.head.op, values.tail)
  }

  implicit val outputCondOutput: Aux[Output, Output] = new CondOutput[Output] {
    override type ResultType = Output
    override def size(output: Output): Int = 1
    override def processOutput(output: Output, context: CondContext): Output = context.processOutput(output)
    override def flatten(processedOutput: Output): Seq[OutputLike] = Seq(processedOutput)
    override def segment(output: Output, values: Seq[OutputLike]): (Output, Seq[OutputLike]) = {
      (values.head.asInstanceOf[Output], values.tail)
    }
  }

  implicit val outputIndexedSlicesCondOutput: Aux[OutputIndexedSlices, OutputIndexedSlices] = {
    new CondOutput[OutputIndexedSlices] {
      override type ResultType = OutputIndexedSlices

      override def size(output: OutputIndexedSlices): Int = 1

      override def processOutput(output: OutputIndexedSlices, context: CondContext): OutputIndexedSlices = {
        val indices = context.processOutput(output.indices)
        val values = context.processOutput(output.values)
        val denseShape = {
          if (output.denseShape != null)
            context.processOutput(output.denseShape)
          else
            null
        }
        OutputIndexedSlices(indices = indices, values = values, denseShape = denseShape)
      }

      override def flatten(processedOutput: OutputIndexedSlices): Seq[OutputLike] = Seq(processedOutput)

      override def segment(
          output: OutputIndexedSlices, values: Seq[OutputLike]): (OutputIndexedSlices, Seq[OutputLike]) = {
        (values.head.asInstanceOf[OutputIndexedSlices], values.tail)
      }
    }
  }

  implicit val sparseOutputCondOutput: Aux[SparseOutput, SparseOutput] = new CondOutput[SparseOutput] {
    override type ResultType = SparseOutput

    override def size(output: SparseOutput): Int = 1

    override def processOutput(output: SparseOutput, context: CondContext): SparseOutput = {
      val indices = context.processOutput(output.indices)
      val values = context.processOutput(output.values)
      val denseShape = {
        if (output.denseShape != null)
          context.processOutput(output.denseShape)
        else
          null
      }
      SparseOutput(indices = indices, values = values, denseShape = denseShape)
    }

    override def flatten(processedOutput: SparseOutput): Seq[OutputLike] = Seq(processedOutput)

    override def segment(output: SparseOutput, values: Seq[OutputLike]): (SparseOutput, Seq[OutputLike]) = {
      (values.head.asInstanceOf[SparseOutput], values.tail)
    }
  }

  implicit val tensorArrayCondOutput: Aux[TensorArray, Output] = new CondOutput[TensorArray] {
    override type ResultType = Output
    override def size(output: TensorArray): Int = 1
    override def processOutput(output: TensorArray, context: CondContext): Output = context.processOutput(output.flow)
    override def flatten(processedOutput: Output): Seq[OutputLike] = Seq(processedOutput)
    override def segment(output: TensorArray, values: Seq[OutputLike]): (TensorArray, Seq[OutputLike]) = {
      val newTensorArray = output.copy(flow = values.head.asInstanceOf[Output])
      // TODO: !!! [TENSOR_ARRAY] What about colocate with?
      (newTensorArray, values.tail)
    }
  }

  implicit def condOutputArray[T: ClassTag, R: ClassTag](implicit ev: Aux[T, R]): Aux[Array[T], Array[R]] = {
    new CondOutput[Array[T]] {
      override type ResultType = Array[R]

      override def size(output: Array[T]): Int = output.map(ev.size).sum

      override def processOutput(output: Array[T], context: CondContext): Array[R] = {
        output.map(ev.processOutput(_, context))
      }

      override def flatten(processedOutput: Array[R]): Seq[OutputLike] = processedOutput.toSeq.flatMap(ev.flatten)

      override def segment(output: Array[T], values: Seq[OutputLike]): (Array[T], Seq[OutputLike]) = {
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

      override def size(output: CC[T]): Int = output.map(ev.size).sum

      override def processOutput(output: CC[T], context: CondContext): CC[R] = {
        output.map(ev.processOutput(_, context))(cbfTR)
      }

      override def flatten(processedOutput: CC[R]): Seq[OutputLike] = processedOutput.flatMap(ev.flatten).toSeq

      override def segment(output: CC[T], values: Seq[OutputLike]): (CC[T], Seq[OutputLike]) = {
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

      override def size(output: Map[MK, T]): Int = output.values.map(ev.size).sum

      override def processOutput(output: Map[MK, T], context: CondContext): Map[MK, R] = {
        output.map({ case (k, v) => k -> ev.processOutput(v, context) })
      }

      override def flatten(processedOutput: Map[MK, R]): Seq[OutputLike] = {
        processedOutput.values.flatMap(ev.flatten).toSeq
      }

      override def segment(output: Map[MK, T], values: Seq[OutputLike]): (Map[MK, T], Seq[OutputLike]) = {
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
    override def size(output: HNil): Int = 0
    override def processOutput(output: HNil, context: CondContext): HNil = HNil
    override def flatten(processedOutput: HNil): Seq[OutputLike] = Seq.empty[OutputLike]
    override def segment(output: HNil, values: Seq[OutputLike]): (HNil, Seq[OutputLike]) = (HNil, values)
  }

  implicit def recursiveConstructor[H, HR, T <: HList, TR <: HList](implicit
      evHead: Lazy[Aux[H, HR]],
      evTail: Aux[T, TR]
  ): Aux[H :: T, HR :: TR] = new CondOutput[H :: T] {
    override type ResultType = HR :: TR

    override def size(output: H :: T): Int = {
      evHead.value.size(output.head) + evTail.size(output.tail)
    }

    override def processOutput(output: H :: T, context: CondContext): HR :: TR = {
      evHead.value.processOutput(output.head, context) :: evTail.processOutput(output.tail, context)
    }

    override def flatten(processedOutput: HR :: TR): Seq[OutputLike] = {
      evHead.value.flatten(processedOutput.head) ++ evTail.flatten(processedOutput.tail)
    }

    override def segment(output: H :: T, values: Seq[OutputLike]): (H :: T, Seq[OutputLike]) = {
      val (headOut, headRemaining) = evHead.value.segment(output.head, values)
      val (tailOut, tailRemaining) = evTail.segment(output.tail, headRemaining)
      (headOut :: tailOut, tailRemaining)
    }
  }

  implicit def productConstructor[P <: Product, PR <: Product, L <: HList, LR <: HList](implicit
      genP: Generic.Aux[P, L],
      evL: Aux[L, LR],
      tuplerR: Tupler.Aux[LR, PR],
      tuplerP: Tupler.Aux[L, P],
      genR: Generic.Aux[PR, LR]
  ): Aux[P, PR] = new CondOutput[P] {
    override type ResultType = PR

    override def size(output: P): Int = evL.size(genP.to(output))

    override def processOutput(output: P, context: CondContext): PR = {
      tuplerR(evL.processOutput(genP.to(output), context))
    }

    override def flatten(processedOutput: PR): Seq[OutputLike] = evL.flatten(genR.to(processedOutput))

    override def segment(output: P, values: Seq[OutputLike]): (P, Seq[OutputLike]) = {
      val (out, remaining) = evL.segment(genP.to(output), values)
      (tuplerP(out), remaining)
    }
  }
}
