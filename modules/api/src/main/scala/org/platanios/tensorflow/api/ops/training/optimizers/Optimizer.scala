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
import org.platanios.tensorflow.api.ops._
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.training.optimizers.Optimizer._
import org.platanios.tensorflow.api.ops.variables.{ConstantInitializer, Initializer, Variable, VariableScope}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types._

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

  /** Some optimizer subclasses use additional variables. For example, `MomentumOptimizer` and `AdaGradOptimizer`
    * use variables to accumulate updates. This map is where these variables are stored. */
  protected final val slots: mutable.Map[String, mutable.Map[Variable[Any], Variable[Any]]] = {
    mutable.Map.empty
  }

  /** Returns the names of all slots used by this optimizer. */
  protected final def slotNames: Set[String] = {
    slots.keySet.toSet
  }

  /** Contains variables used by some optimizers that require no slots to be stored. */
  protected final val nonSlotVariables: mutable.Map[(String, Option[Graph]), Variable[Any]] = {
    mutable.Map.empty
  }

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
  final def minimize[T: IsFloat32OrFloat64, I: IsInt32OrInt64](
      loss: Output[T],
      lossGradients: Seq[OutputLike[Any]] = null,
      variables: Set[Variable[Any]] = null,
      gradientsGatingMethod: Gradients.GatingMethod = Gradients.OpGating,
      gradientsAggregationMethod: Gradients.AggregationMethod = Gradients.AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false,
      iteration: Option[Variable[I]] = None,
      name: String = "Minimize"
  ): UntypedOp = {
    val gradientsAndVariables = computeGradients(
      loss, lossGradients, variables, gradientsGatingMethod,
      gradientsAggregationMethod, colocateGradientsWithOps)
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
    * @throws IllegalArgumentException If there are no variables to optimize.
    */
  @throws[IllegalArgumentException]
  def computeGradients[T: IsFloat32OrFloat64](
      loss: Output[T],
      lossGradients: Seq[OutputLike[Any]] = null,
      variables: Set[Variable[Any]] = null,
      gradientsGatingMethod: Gradients.GatingMethod = Gradients.OpGating,
      gradientsAggregationMethod: Gradients.AggregationMethod = Gradients.AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false
  ): Seq[(OutputLike[Any], Variable[Any])] = {
    // TODO: [VARIABLES] Settle on what keys to use for variables.
    val collectedVariables: Seq[Variable[Any]] = {
      {
        if (variables == null)
          loss.graph.trainableVariables
        else
          variables
      } ++ loss.graph.getCollection(Graph.Keys.STREAMING_MODEL_PORTS)
    }.toSeq
    if (collectedVariables.isEmpty)
      throw new IllegalArgumentException("There are no variables to optimize.")

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsFloat32OrFloat64[Any] = new IsFloat32OrFloat64[Any] {}

    val variableProcessors = collectedVariables.map(getVariableProcessor(_)(ev))
    val variableTargets = variableProcessors.map(_.target)
    val gradients = {
      val gradients = Gradients.gradients(
        ys = Seq(loss),
        xs = variableTargets,
        dys = lossGradients,
        gateGradients = gradientsGatingMethod == Gradients.OpGating,
        aggregationMethod = gradientsAggregationMethod,
        colocateGradientsWithOps = colocateGradientsWithOps)
      if (gradientsGatingMethod == Gradients.GraphGating) {
        ControlFlow.tuple(gradients)
      } else {
        gradients
      }
    }
    gradients.zip(collectedVariables)
  }

  /** Creates an op that applies the provided gradients to the provided variables.
    *
    * @param  gradientsAndVariables Sequence with gradient-variable pairs.
    * @param  iteration             Optional `Variable` to increment by one after the variables have been updated.
    * @param  name                  Name for the created op.
    * @return Created op.
    */
  def applyGradients[I: IsInt32OrInt64](
      gradientsAndVariables: Seq[(OutputLike[Any], Variable[Any])],
      iteration: Option[Variable[I]] = None,
      name: String = this.name
  ): UntypedOp = {
    // This is a default implementation of `applyGradients` that is shared by most optimizers. It relies on the subclass
    // implementing the following methods: `createSlots`, `prepare`, `finish`, `applyDense`, and `applySparse`.
    val variables: Seq[Variable[Any]] = gradientsAndVariables.filter(_._1 != null).map(_._2)
    if (variables.isEmpty)
      throw new IllegalArgumentException(
        "No gradients were provided for any of the variables: " +
            s"${gradientsAndVariables.map(_._2).mkString(", ")}.")

    Op.nameScope(name) {
      // Create the slots needed by the variables.
      Op.initializationScope {
        VariableScope.scope(name) {
          createSlots(variables)
        }
      }

      prepare(iteration)

      // Collect the update ops for all variables.
      val updateOps = mutable.Set.empty[UntypedOp]
      for ((g, v) <- gradientsAndVariables if g != null) {

        // TODO: [TYPES] !!! Super hacky. Remove in the future.
        implicit val ev: IsFloat32OrFloat64[Any] = new IsFloat32OrFloat64[Any] {}

        val p = getVariableProcessor(v)
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
          case Some(i) =>
            val finishOp = finish(updateOps.toSet, "Finish")
            Op.createWith(controlDependencies = Set(finishOp)) {
              Op.colocateWith(Set(i.op), ignoreExisting = true) {
                // The implicit read in the default assign add operation in `Variable` is slow and so we avoid that here.
                Variable.assignAdd(i.handle, Basic.ones(i.dataType, Shape()), name).asUntyped
              }
            }
          case None =>
            finish(updateOps.toSet, "Finish").asUntyped
        }
      }

      // Add the created op to the graph train ops collection.
      updateOps.head.graph.addToCollection(applyUpdates, Graph.Keys.TRAIN_OP)

      applyUpdates
    }
  }

  /** Create all slots needed by this optimizer. */
  def createSlots(variables: Seq[Variable[Any]]): Unit = {
    // No slots are created by default.
  }

  /** Creates all necessary tensors before applying the gradients. This function is called from within an op creation
    * context that uses as its name scope the name that users have chosen for the application of gradients. */
  def prepare[I: IsInt32OrInt64](iteration: Option[Variable[I]]): Unit = {
    // No preparation is done by default.
  }

  /** Creates an op that finishes the gradients application. This function is called from within an op creation context
    * that uses as its name scope the name that users have chosen for the application of gradients.
    *
    * @param  updateOps Set of ops needed to apply the gradients and update the variable values.
    * @param  nameScope Name scope to use for all the ops created by this function.
    * @return Created op output.
    */
  def finish(
      updateOps: Set[UntypedOp],
      nameScope: String
  ): UntypedOp = {
    ControlFlow.group(updateOps, nameScope).asUntyped
  }

  /** Applies the updates corresponding to the provided gradient, to the provided variable.
    *
    * @param  gradient  Gradient tensor.
    * @param  variable  Variable.
    * @param  iteration Option containing current iteration in the optimization loop, if one has been provided.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  def applyDense[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: Output[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp

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
  def applySparse[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp

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
  def applySparseDuplicateIndices[T: IsNotQuantized, I: IsInt32OrInt64](
      gradient: OutputIndexedSlices[T],
      variable: Variable[T],
      iteration: Option[Variable[I]]
  ): UntypedOp = {
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
  private def slotMap(name: String): mutable.Map[Variable[Any], Variable[Any]] = {
    slots.getOrElseUpdate(name, mutable.Map.empty[Variable[Any], Variable[Any]])
  }

  /** Gets an existing slot or creates a new one if none exists, for the provided arguments.
    *
    * @param  name          Slot name.
    * @param  variable      Slot primary variable.
    * @param  dataType      Slot variable data type.
    * @param  initializer   Slot variable initializer.
    * @param  shape         Slot variable shape.
    * @param  variableScope Name to use when scoping the variable that needs to be created for the slot.
    * @return Requested slot variable.
    */
  protected final def getSlot[T, R](
      name: String,
      variable: Variable[T],
      dataType: DataType[R],
      initializer: Initializer,
      shape: Shape,
      variableScope: String
  ): Variable[R] = {
    Op.colocateWith(Set(variable.op), ignoreExisting = true) {
      slotMap(name).getOrElseUpdate(variable, {
        Slot.create(variable, dataType, initializer, variableScope, shape)
      }).asInstanceOf[Variable[R]]
    }
  }

  /** Gets an existing slot.
    *
    * @param  name     Slot name.
    * @param  variable Slot primary variable.
    * @return Requested slot variable, or `null` if it cannot be found.
    */
  protected final def getSlot[T, R](
      name: String,
      variable: Variable[T]
  ): Variable[R] = {
    slots.getOrElse(name, Map.empty[Variable[Any], Variable[Any]])
        .getOrElse(variable, null)
        .asInstanceOf[Variable[R]]
  }

  /** Gets an existing slot or creates a new one using an initial value of zeros, if none exists.
    *
    * @param  name          Slot name.
    * @param  variable      Slot primary variable.
    * @param  variableScope Name to use when scoping the variable that needs to be created for the slot.
    * @return Requested slot variable.
    */
  protected final def zerosSlot[T](
      name: String,
      variable: Variable[T],
      variableScope: String
  ): Variable[T] = {
    Op.colocateWith(Set(variable.op), ignoreExisting = true) {
      slotMap(name).getOrElseUpdate(variable, {
        Slot.zeros(variable, variable.dataType, s"$variableScope/$name")
      }).asInstanceOf[Variable[T]]
    }
  }

  /** Gets or creates (and adds to this optimizer) a non-slot variable.
    *
    * @param  name          Variable name.
    * @param  initialValue  Variable initial value.
    * @param  colocationOps Set of colocation ops for the non-slot variable.
    * @return Created non-slot variable.
    */
  protected final def getOrCreateNonSlotVariable[T](
      name: String,
      initialValue: Tensor[T],
      colocationOps: Set[UntypedOp] = Set.empty,
      ignoreExisting: Boolean = false
  ): Variable[T] = {
    nonSlotVariables.getOrElseUpdate(
      (name, colocationOps.map(_.graph).headOption),
      Op.colocateWith(colocationOps, ignoreExisting) {
        Variable.getVariable[T](
          name, initialValue.shape, initializer = ConstantInitializer(initialValue), trainable = false)
      }).asInstanceOf[Variable[T]]
  }

  /** Gets a non-slot variable that has been added to this optimizer (or throws an error if no such non-slot variable
    * could be found in this optimizer).
    *
    * @param  name  Variable name.
    * @param  graph Graph in which the variable is defined.
    * @return Obtained non-slot variable.
    */
  protected final def getNonSlotVariable[T](
      name: String,
      graph: Graph = null
  ): Variable[T] = {
    nonSlotVariables((name, Option(graph))).asInstanceOf[Variable[T]]
  }

  /** Gets all the non-slot variables that have been added to this optimizer. */
  protected final def getNonSlotVariables: Iterable[Variable[Any]] = {
    nonSlotVariables.values
  }

  /** Returns a sequence of variables which encode the current state of this optimizer. The returned variables include
    * both slot variables and non-slot global variables created by this optimizer, in the current graph. */
  final def state: Seq[Variable[Any]] = {
    (getNonSlotVariables.filter(_.graph == Op.currentGraph) ++
        slots.values.flatMap(_.values)).toSeq.sortBy(_.name)
  }
}

private[optimizers] object Optimizer {
  /** Gets the appropriate variable processor to use for `variable`. */
  private[optimizers] def getVariableProcessor[T: IsFloat32OrFloat64](
      variable: Variable[T]
  ): VariableProcessor[T] = {
    variable match {
      // TODO: [VARIABLES] This is dummy for now.
      case v if v.op.opType == "VarHandleOp" =>
        ResourceVariableProcessor(v)
      case v if v.op.opType == "SubmodelPort" =>
        StreamingModelPortProcessor(v)
      case _ => throw new IllegalArgumentException(
        s"Unsupported variable op type '${variable.op.opType}'.")
    }
  }

  /** Trait for abstracting over variables in the optimizers. */
  private[Optimizer] sealed trait VariableProcessor[T] {
    /** Returns the optimization target for this variable. */
    def target: Output[Long]

    /** Returns the update ops for updating this variable using the gradient provided by `gradient`. */
    def updateOp[I: IsInt32OrInt64](
        optimizer: Optimizer,
        gradient: OutputLike[T],
        iteration: Option[Variable[I]]
    ): UntypedOp
  }

  /** Variable processor for resource-based variables. */
  private[Optimizer] case class ResourceVariableProcessor[T: IsFloat32OrFloat64](
      variable: Variable[T]
  ) extends VariableProcessor[T] {
    override def target: Output[Long] = {
      variable.handle
    }

    override def updateOp[I: IsInt32OrInt64](
        optimizer: Optimizer,
        gradient: OutputLike[T],
        iteration: Option[Variable[I]]
    ): UntypedOp = {
      gradient match {
        case g: Output[T] => optimizer.applyDense(g, variable, iteration)
        case g: OutputIndexedSlices[T] => optimizer.applySparseDuplicateIndices(g, variable, iteration)
        case _ => throw new IllegalArgumentException(
          "Unsupported gradient type. Currently only 'Output' and 'OutputIndexedSlices' are supported.")
      }
    }
  }

  /** Variable processor for streaming model ports. */
  private[Optimizer] case class StreamingModelPortProcessor[T](
      variable: Variable[T]
  ) extends VariableProcessor[T] {
    // TODO: [VARIABLES] This is probably wrong.
    override def target: Output[Long] = {
      variable.handle
    }

    override def updateOp[I: IsInt32OrInt64](
        optimizer: Optimizer,
        gradient: OutputLike[T],
        iteration: Option[Variable[I]]
    ): UntypedOp = {
      gradient.op
    }
  }

  /** Sums the values of the provided indexed slices associated with any non-unique indices and returns the resulting
    * de-duplicated version of the provided indexed slices.
    *
    * @param  input Indexed slices with potentially duplicate indices.
    * @return Indexed slices with de-duplicated indices and summed values slices associated with each unique index.
    */
  private[Optimizer] def deDuplicateOutputIndexedSlices[T: IsNumeric](
      input: OutputIndexedSlices[T]
  ): OutputIndexedSlices[T] = {
    val (uniqueIndices, newIndexPositions) = Basic.unique(input.indices, Tensor(0), INT32)
    val summedValues = Math.unsortedSegmentSum(
      data = input.values,
      segmentIndices = newIndexPositions,
      segmentsNumber = Basic.shape(uniqueIndices, INT32).slice(0))
    OutputIndexedSlices(indices = uniqueIndices, values = summedValues, denseShape = input.denseShape)
  }
}
