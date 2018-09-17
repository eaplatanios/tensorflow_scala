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

package org.platanios.tensorflow.api.ops.training.optimizers

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer._
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, Initializer, Variable, VariableScope}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, FLOAT32, FLOAT64, RESOURCE}

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
trait Optimizer {
  /** Name of this optimizer. This name is used for the accumulators created for this optimizer. */
  val name: String

  /** Boolean value indicating whether to apply use locks to prevent concurrent updates to variables. */
  val useLocking: Boolean

  /** Boolean value indicating whether to ignore duplicate indices during sparse updates. */
  val ignoreDuplicateSparseIndices: Boolean

  // TODO: [OPTIMIZER] Slot variables make re-using optimizer objects a bit dirty.

  /** Some [[Optimizer]] subclasses use additional variables. For example, `MomentumOptimizer` and `AdaGradOptimizer`
    * use variables to accumulate updates. This map is where these variables are stored. */
  protected final val slots = mutable.Map.empty[String, mutable.Map[Variable, Variable]]

  /** Returns the names of all slots used by this optimizer. */
  protected final def slotNames: Set[String] = slots.keySet.toSet

  /** Contains variables used by some optimizers that require no slots to be stored. */
  protected final val nonSlotVariables = mutable.Map.empty[(String, Option[Graph]), Variable]

  /** Creates an op that makes a step towards minimizing `loss` by updating the values of the variables in `variables`.
    *
    * This method simply combines calls [[computeGradients]] and [[applyGradients]]. If you want to process the
    * gradients before applying them call [[computeGradients]] and [[applyGradients]] explicitly instead of using this
    * method.
    *
    * @param  loss                       Loss value whose gradients will be computed.
    * @param  lossGradients              Optional gradients to back-propagate for `loss`.
    * @param  variables                  Optional list of variables for which to compute the gradients. Defaults to the
    *                                    set of trainable variables in the graph where `loss` is defined.
    * @param  gradientsGatingMethod      Gating method for the gradients computation.
    * @param  gradientsAggregationMethod Aggregation method used to combine gradient terms.
    * @param  colocateGradientsWithOps   Boolean value indicating whether to colocate the gradient ops with the original
    *                                    ops.
    * @param  iteration                  Optional `Variable` to increment by one after the variables have been updated.
    * @param  name                       Name for the created op.
    * @return Created op.
    */
  final def minimize(
      loss: Output,
      lossGradients: Seq[OutputLike] = null,
      variables: Set[Variable] = null,
      gradientsGatingMethod: Gradients.GatingMethod = Gradients.OpGating,
      gradientsAggregationMethod: Gradients.AggregationMethod = Gradients.AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false,
      iteration: Option[Variable] = None,
      name: String = "Minimize"
  ): Op = {
    val gradientsAndVariables = computeGradients(
      loss, lossGradients, variables, gradientsGatingMethod, gradientsAggregationMethod, colocateGradientsWithOps)
    applyGradients(gradientsAndVariables, iteration, name)
  }

  /** Computes the gradients of `loss` with respect to the variables in `variables`, if provided, otherwise with respect
    * to all the trainable variables in the graph where `loss` is defined.
    *
    * @param  loss                       Loss value whose gradients will be computed.
    * @param  lossGradients              Optional gradients to back-propagate for `loss`.
    * @param  variables                  Optional list of variables for which to compute the gradients. Defaults to the
    *                                    set of trainable variables in the graph where `loss` is defined.
    * @param  gradientsGatingMethod      Gating method for the gradients computation.
    * @param  gradientsAggregationMethod Aggregation method used to combine gradient terms.
    * @param  colocateGradientsWithOps   Boolean value indicating whether to colocate the gradient ops with the original
    *                                    ops.
    * @return Sequence of gradient-variable pairs.
    */
  def computeGradients(
      loss: Output,
      lossGradients: Seq[OutputLike] = null,
      variables: Set[Variable] = null,
      gradientsGatingMethod: Gradients.GatingMethod = Gradients.OpGating,
      gradientsAggregationMethod: Gradients.AggregationMethod = Gradients.AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false
  ): Seq[(OutputLike, Variable)] = {
    assertSupportedDataTypes(Iterable[OutputLike](loss))
    if (lossGradients != null)
      assertSupportedDataTypes(lossGradients)
    // TODO: [VARIABLES] Settle on what keys to use for variables.
    val collectedVariables: Seq[Variable] = {
      {
        if (variables == null) loss.graph.trainableVariables else variables
      } ++ loss.graph.getCollection(Graph.Keys.STREAMING_MODEL_PORTS)
    }.toSeq
    if (collectedVariables.isEmpty)
      throw new IllegalArgumentException("There are no variables to optimize.")
    val variableProcessors: Seq[VariableProcessor] = collectedVariables.map(getVariableProcessor)
    val variableTargets: Seq[Output] = variableProcessors.map(_.target)
    val gradients: Seq[OutputLike] = {
      val gradients = Gradients.gradients(
        ys = Seq[Output](loss),
        xs = variableTargets,
        dys = lossGradients,
        gateGradients = gradientsGatingMethod == Gradients.OpGating,
        aggregationMethod = gradientsAggregationMethod,
        colocateGradientsWithOps = colocateGradientsWithOps)
      if (gradientsGatingMethod == Gradients.GraphGating)
        ControlFlow.tuple(gradients.toArray).toSeq
      else
        gradients
    }
    val gradientsAndVariables: Seq[(OutputLike, Variable)] = gradients.zip(collectedVariables)
    assertSupportedDataTypes(
      gradientsAndVariables.filter(p => (p._1 != null) && p._2.dataType != RESOURCE).map(_._2.value))
    gradientsAndVariables
  }

  /** Creates an op that applies the provided gradients to the provided variables.
    *
    * @param  gradientsAndVariables Sequence with gradient-variable pairs.
    * @param  iteration             Optional `Variable` to increment by one after the variables have been updated.
    * @param  name                  Name for the created op.
    * @return Created op.
    */
  def applyGradients(
      gradientsAndVariables: Seq[(OutputLike, Variable)],
      iteration: Option[Variable] = None,
      name: String = this.name
  ): Op = {
    // This is a default implementation of `applyGradients` that is shared by most optimizers. It relies on the subclass
    // implementing the following methods: `createSlots`, `prepare`, `finish`, `applyDense`, and `applySparse`.
    val variables: Seq[Variable] = gradientsAndVariables.filter(_._1 != null).map(_._2)
    if (variables.isEmpty)
      throw new IllegalArgumentException(
        s"No gradients were provided for any of the variables: ${gradientsAndVariables.map(_._2).mkString(", ")}.")

    Op.createWithNameScope(name) {
      // Create the slots needed by the variables.
      Op.initialization {
        VariableScope.scope(name) {
          createSlots(variables)
        }
      }

      prepare(iteration)

      // Collect the update ops for all variables.
      val updateOps = mutable.Set.empty[Op]
      for ((g, v, p) <- gradientsAndVariables.map(p => (p._1, p._2, getVariableProcessor(p._2))).filter(_._1 != null)) {
        // We colocate all ops created for variable application on the same device as the variable.
        Op.createWith(nameScope = s"Update/${v.op.name}") {
          Op.colocateWith(Set(v.op), ignoreExisting = true) {
            updateOps.add(p.updateOp(this, g, iteration))
          }
        }
      }

      // Create the op that applies the gradient updates to all variables.
      val applyUpdates = {
        iteration match {
          case Some(i) => Op.createWith(controlDependencies = Set(finish(updateOps.toSet, "Finish"))) {
            Op.colocateWith(Set(i.op), ignoreExisting = true) {
              // The implicit read in the default assign add operation in `Variable` is slow and so we avoid that here.
              Variable.assignAdd(i.handle, Basic.constant(1, dataType = i.dataType), name)
            }
          }
          case None => finish(updateOps.toSet, "Finish")
        }
      }

      // Add the created op to the graph train ops collection.
      updateOps.head.graph.addToCollection(applyUpdates, Graph.Keys.TRAIN_OP)

      applyUpdates
    }
  }

  /** Supported data types for the loss function, the variables, and the gradients. Subclasses should override this
    * field allow other float types. */
  val supportedDataTypes: Set[DataType] = Set[DataType](FLOAT32, FLOAT64)

  /** Asserts that the provided `outputs` all have data types that are supported by this optimizer.
    *
    * @param  outputs Outputs whose data types to check.
    * @throws InvalidDataTypeException If any of the provided outputs has an unsupported data type.
    */
  @throws[InvalidDataTypeException]
  private[this] def assertSupportedDataTypes(outputs: Iterable[OutputLike]): Unit = {
    outputs.foreach(output => {
      if (!supportedDataTypes.contains(output.dataType))
        throw InvalidDataTypeException(s"Data type '${output.dataType}' is not supported by this optimizer.")
    })
  }

  /** Create all slots needed by this optimizer. */
  def createSlots(variables: Seq[Variable]): Unit = {
    // No slots are created by default.
  }

  /** Creates all necessary tensors before applying the gradients. This function is called from within an op creation
    * context that uses as its name scope the name that users have chosen for the application of gradients. */
  def prepare(iteration: Option[Variable]): Unit = {}

  /** Creates an op that finishes the gradients application. This function is called from within an op creation context
    * that uses as its name scope the name that users have chosen for the application of gradients.
    *
    * @param  updateOps Set of ops needed to apply the gradients and update the variable values.
    * @param  nameScope Name scope to use for all the ops created by this function.
    * @return Created op output.
    */
  def finish(updateOps: Set[Op], nameScope: String): Op = {
    ControlFlow.group(updateOps, nameScope)
  }

  /** Applies the updates corresponding to the provided gradient, to the provided variable.
    *
    * @param  gradient  Gradient tensor.
    * @param  variable  Variable.
    * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  def applyDense(gradient: Output, variable: Variable, iteration: Option[Variable]): Op

  /** Applies the updates corresponding to the provided gradient, to the provided variable.
    *
    * The [[OutputIndexedSlices]] object specified by `gradient` in this function is by default pre-processed in
    * `applySparseDuplicateIndices` to remove duplicate indices (refer to that function's documentation for details).
    * Optimizers which can tolerate or have correct special cases for duplicate sparse indices may override
    * `applySparseDuplicateIndices` instead of this function, avoiding that overhead.
    *
    * @param  gradient  Gradient tensor.
    * @param  variable  Variable.
    * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  def applySparse(gradient: OutputIndexedSlices, variable: Variable, iteration: Option[Variable]): Op

  /** Applies the updates corresponding to the provided gradient (with potentially duplicate indices), to the provided
    * variable.
    *
    * Optimizers which override this method must deal with [[OutputIndexedSlices]] objects such as the following:
    * `OutputIndexedSlices(indices=[0, 0], values=[1, 1], denseShape=[1])`, which contain duplicate indices. The
    * correct interpretation in that case should be: `OutputIndexedSlices(values=[2], indices=[0], denseShape=[1])`.
    *
    * Many optimizers deal incorrectly with repeated indices when updating based on sparse gradients (e.g. summing
    * squares rather than squaring the sum, or applying momentum terms multiple times). Adding first is always the
    * correct behavior, so this is enforced here by reconstructing the [[OutputIndexedSlices]] to have only unique
    * indices, and then calling [[applySparse]].
    *
    * Optimizers which deal correctly with repeated indices may instead override this method to avoid the induced
    * overhead.
    *
    * @param  gradient  Gradient tensor.
    * @param  variable  Variable.
    * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  def applySparseDuplicateIndices(
      gradient: OutputIndexedSlices,
      variable: Variable,
      iteration: Option[Variable]
  ): Op = {
    if (ignoreDuplicateSparseIndices)
      applySparse(gradient, variable, iteration)
    else
      applySparse(deDuplicateOutputIndexedSlices(gradient), variable, iteration)
  }

  /** Gets the map used for caching slots created under the provided name. If the map does not exist, then a new empty
    * map is created and returned.
    *
    * @param  name Slot name.
    * @return Map used for caching slots created under the provided name.
    */
  private[this] def slotMap(name: String): mutable.Map[Variable, Variable] = {
    slots.getOrElseUpdate(name, mutable.Map.empty[Variable, Variable])
  }

  /** Gets an existing slot or creates a new one if none exists, for the provided arguments.
    *
    * @param  name          Slot name.
    * @param  variable      Slot primary variable.
    * @param  initializer   Slot variable initializer.
    * @param  shape         Slot variable shape.
    * @param  dataType      Slot variable data type.
    * @param  variableScope Name to use when scoping the variable that needs to be created for the slot.
    * @return Requested slot variable.
    */
  protected final def getSlot(
      name: String,
      variable: Variable,
      initializer: Initializer,
      shape: Shape,
      dataType: DataType,
      variableScope: String
  ): Variable = {
    Op.colocateWith(Set(variable.op), ignoreExisting = true) {
      slotMap(name).getOrElseUpdate(variable, Slot.create(variable, initializer, variableScope, dataType, shape))
    }
  }

  /** Gets an existing slot.
    *
    * @param  name     Slot name.
    * @param  variable Slot primary variable.
    * @return Requested slot variable, or `null` if it cannot be found.
    */
  protected final def getSlot(name: String, variable: Variable): Variable = {
    slots.getOrElse(name, Map.empty[Variable, Variable]).getOrElse(variable, null)
  }

  /** Gets an existing slot or creates a new one using an initial value of zeros, if none exists.
    *
    * @param  name          Slot name.
    * @param  variable      Slot primary variable.
    * @param  variableScope Name to use when scoping the variable that needs to be created for the slot.
    * @return Requested slot variable.
    */
  protected final def zerosSlot(name: String, variable: Variable, variableScope: String): Variable = {
    Op.colocateWith(Set(variable.op), ignoreExisting = true) {
      slotMap(name).getOrElseUpdate(variable, Slot.zeros(variable, s"$variableScope/$name"))
    }
  }

  /** Gets or creates (and adds to this optimizer) a non-slot variable.
    *
    * @param  name          Variable name.
    * @param  initialValue  Variable initial value.
    * @param  colocationOps Set of colocation ops for the non-slot variable.
    * @return Created non-slot variable.
    */
  protected final def getOrCreateNonSlotVariable(
      name: String,
      initialValue: Tensor[_ <: DataType],
      colocationOps: Set[Op] = Set.empty,
      ignoreExisting: Boolean = false
  ): Variable = {
    nonSlotVariables.getOrElseUpdate(
      (name, colocationOps.map(_.graph).headOption),
      Op.colocateWith(colocationOps, ignoreExisting) {
        Variable.getVariable(name, initializer = ConstantInitializer(initialValue), trainable = false)
      })
  }

  /** Gets a non-slot variable that has been added to this optimizer (or throws an error if no such non-slot variable
    * could be found in this optimizer).
    *
    * @param  name  Variable name.
    * @param  graph Graph in which the variable is defined.
    * @return Obtained non-slot variable.
    */
  protected final def getNonSlotVariable(name: String, graph: Graph = null): Variable = {
    nonSlotVariables((name, Option(graph)))
  }

  /** Gets all the non-slot variables that have been added to this optimizer. */
  protected final def getNonSlotVariables: Iterable[Variable] = nonSlotVariables.values

  /** Returns a sequence of variables which encode the current state of this optimizer. The returned variables include
    * both slot variables and non-slot global variables created by this optimizer, in the current graph. */
  final def variables: Seq[Variable] = {
    (getNonSlotVariables.filter(_.graph == Op.currentGraph) ++ slots.values.flatMap(_.values))
        .toSeq.sortBy(_.name)
  }
}

private[optimizers] object Optimizer {
  /** Gets the appropriate variable processor to use for `variable`. */
  private[optimizers] def getVariableProcessor(variable: Variable): VariableProcessor = variable match {
    // TODO: [VARIABLES] This is dummy for now.
    case v if v.op.opType == "VarHandleOp" => ResourceVariableProcessor(v)
    case v if v.op.opType == "SubmodelPort" => StreamingModelPortProcessor(v)
    case _ => throw new IllegalArgumentException(s"Unsupported variable op type '${variable.op.opType}'.")
  }

  /** Trait for abstracting over variables in the optimizers. */
  private[Optimizer] sealed trait VariableProcessor {
    /** Returns the optimization target for this variable. */
    def target: Output

    /** Returns the update ops for updating this variable using the gradient provided by `gradient`. */
    def updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Option[Variable]): Op
  }

  /** Variable processor for resource-based variables. */
  private[Optimizer] case class ResourceVariableProcessor(variable: Variable) extends VariableProcessor {
    override def target: Output = variable.handle

    override def updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Option[Variable]): Op = {
      gradient match {
        case g: Output => optimizer.applyDense(g, variable, iteration)
        case g: OutputIndexedSlices => optimizer.applySparseDuplicateIndices(g, variable, iteration)
        case _ => throw new IllegalArgumentException(
          "Unsupported gradient type. Currently only 'Output' and 'OutputIndexedSlices' are supported.")
      }
    }
  }

  /** Variable processor for streaming model ports. */
  private[Optimizer] case class StreamingModelPortProcessor(variable: Variable) extends VariableProcessor {
    // TODO: [VARIABLES] This is probably wrong.
    override def target: Output = variable.handle

    override def updateOp(optimizer: Optimizer, gradient: OutputLike, iteration: Option[Variable]): Op = gradient.op
  }

  /** Sums the values of the provided indexed slices associated with any non-unique indices and returns the resulting
    * de-duplicated version of the provided indexed slices.
    *
    * @param  input Indexed slices with potentially duplicate indices.
    * @return Indexed slices with de-duplicated indices and summed values slices associated with each unique index.
    */
  private[Optimizer] def deDuplicateOutputIndexedSlices(input: OutputIndexedSlices): OutputIndexedSlices = {
    val (uniqueIndices, newIndexPositions) = Basic.unique(input.indices, Tensor(0))
    val summedValues = Math.unsortedSegmentSum(input.values, newIndexPositions, Basic.shape(uniqueIndices)(0))
    OutputIndexedSlices(indices = uniqueIndices, values = summedValues, denseShape = input.denseShape)
  }
}
