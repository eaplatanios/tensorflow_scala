package org.platanios.tensorflow.api.ops.optimizers

import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.Exception.InvalidDataTypeException
import org.platanios.tensorflow.api.tf.{DataType, FLOAT32, FLOAT64, INT32, RESOURCE, Variable}
import org.platanios.tensorflow.api.ops.optimizers.Optimizer._
import org.platanios.tensorflow.api.ops.{Basic, ControlFlow, Gradients, Math, Op}

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
trait Optimizer {
  /** Name of this optimizer. This name is used for the accumulators created for this optimizer. */
  val name: String

  /** Boolean value indicating whether to apply use locks to prevent concurrent updates to variables. */
  val useLocking: Boolean

  /** Some [[Optimizer]] subclasses use additional variables. For example, `MomentumOptimizer` and `AdaGradOptimizer`
    * use variables to accumulate updates. This map is where these variables are stored. */
  protected val slots: Map[String, Map[Variable, Variable]] = Map.empty[String, Map[Variable, Variable]]

  /** Returns the names of all slots used by this optimizer. */
  protected def slotNames: Set[String] = slots.keySet

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
    * @param  globalStep                 Optional `Variable` to increment by one after the variables have been updated.
    * @param  name                       Name for the created op.
    * @return Created op.
    */
  def minimize(loss: Op.Output, lossGradients: Seq[Op.OutputLike] = null, variables: Set[Variable] = null,
      gradientsGatingMethod: Gradients.GatingMethod = Gradients.OpGating,
      gradientsAggregationMethod: Gradients.AggregationMethod = Gradients.AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false, globalStep: Variable = null, name: String = "Minimize"): Op = {
    val gradientsAndVariables = computeGradients(
      loss, lossGradients, variables, gradientsGatingMethod, gradientsAggregationMethod, colocateGradientsWithOps)
    applyGradients(gradientsAndVariables, globalStep, name)
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
      loss: Op.Output, lossGradients: Seq[Op.OutputLike] = null, variables: Set[Variable] = null,
      gradientsGatingMethod: Gradients.GatingMethod = Gradients.OpGating,
      gradientsAggregationMethod: Gradients.AggregationMethod = Gradients.AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false): Seq[(Op.OutputLike, Variable)] = {
    assertSupportedDataTypes(Iterable[Op.OutputLike](loss))
    if (lossGradients != null)
      assertSupportedDataTypes(lossGradients)
    // TODO: [VARIABLES] Settle on what keys to use for variables.
    val collectedVariables: Seq[Variable] = {
      {
        if (variables == null)
          loss.graph.trainableVariables ++ loss.graph.getCollection(Graph.Keys.TRAINABLE_RESOURCE_VARIABLES)
        else
          variables
      } ++ loss.graph.getCollection(Graph.Keys.STREAMING_MODEL_PORTS)
    }.map(_.asInstanceOf[Variable]).toSeq
    if (collectedVariables.isEmpty)
      throw new IllegalArgumentException("There are no variables to optimize.")
    val variableProcessors: Seq[VariableProcessor] = collectedVariables.map(getVariableProcessor)
    val variableTargets: Seq[Op.Output] = variableProcessors.map(_.target)
    val gradients: Seq[Op.OutputLike] = {
      val gradients = Gradients.gradients(
        ys = Seq[Op.Output](loss),
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
    val gradientsAndVariables: Seq[(Op.OutputLike, Variable)] = gradients.zip(collectedVariables)
    assertSupportedDataTypes(
      gradientsAndVariables.filter(p => (p._1 ne null) && p._2.dataType != RESOURCE).map(_._2.value))
    gradientsAndVariables
  }

  /** Creates an op that applies the provided gradients to the provided variables.
    *
    * @param  gradientsAndVariables Sequence with gradient-variable pairs.
    * @param  globalStep            Optional `Variable` to increment by one after the variables have been updated.
    * @param  name                  Name for the created op.
    * @return Created op.
    */
  def applyGradients(
      gradientsAndVariables: Seq[(Op.OutputLike, Variable)], globalStep: Variable = null,
      name: String = this.name): Op = {
    // This is a default implementation of `applyGradients` that is shared by most optimizers. It relies on the subclass
    // implementing the following methods: `createSlots`, `prepare`, `finish`, `applyDense`, and `applySparse`.
    val variables: Seq[Variable] = gradientsAndVariables.filter(_._1 ne null).map(_._2)
    if (variables.isEmpty)
      throw new IllegalArgumentException(
        s"No gradients were provided for any of the variables: ${gradientsAndVariables.map(_._2).mkString(", ")}.")

    // Create the slots needed by the variables.
    Op.createWith(controlDependencies = Set.empty[Op]) {
      val mappedVariables = variables.map(variable => {
        if (variable.op.opType == "VarHandleOp") {
          val v = variable.graph.trainableVariables.find(v => v.isInstanceOf[Variable] && v.handle.op == variable.op)
          if (v.isEmpty)
            throw new IllegalArgumentException(s"Got handle '$variable', but could not locate the source variable.")
          v.get
        } else {
          variable
        }
      })
      createSlots(mappedVariables)
    }

    Op.createWithNameScope(name) {
      prepare()

      // Collect the update ops for all variables.
      val updateOps = mutable.Set.empty[Op]
      for ((g, v, p) <- gradientsAndVariables.map(p => (p._1, p._2, getVariableProcessor(p._2))).filter(_._1 ne null)) {
        // We colocate all ops created for variable application on the same device as the variable.
        Op.createWith(nameScope = s"${v.op.name}Update", colocationOps = Set[Op](v.op)) {
          updateOps.add(p.updateOp(this, g))
        }
      }

      // Create the op that applies the gradient updates to all variables.
      val applyUpdates = {
        if (globalStep == null) {
          finish(updateOps.toSet, name)
        } else {
          Op.createWith(
            colocationOps = Set[Op](globalStep.op),
            controlDependencies = Set[Op](finish(updateOps.toSet, "Update"))) {
            globalStep.assignAdd(Basic.constant(1, dataType = INT32), name).op
          }
        }
      }

      // Add the created op to the graph train ops collection.
      updateOps.head.graph.getCollectionReference(Graph.Keys.TRAIN_OP).add(applyUpdates)

      applyUpdates
    }
  }

  /** Supported data types for the loss function, the variables, and the gradients. Subclasses should override this
    * field allow other float types. */
  protected val supportedDataTypes: Set[DataType] = Set[DataType](FLOAT32, FLOAT64)

  /** Asserts that the provided `outputs` all have data types that are supported by this optimizer.
    *
    * @param  outputs Outputs whose data types to check.
    * @throws InvalidDataTypeException If any of the provided outputs has an unsupported data type.
    */
  @throws[InvalidDataTypeException]
  private[this] def assertSupportedDataTypes(outputs: Iterable[Op.OutputLike]): Unit = {
    outputs.foreach(output => {
      if (!supportedDataTypes.contains(output.dataType))
        throw InvalidDataTypeException(s"Data type '${output.dataType}' is not supported by this optimizer.")
    })
  }

  /** Create all slots needed by the variables. */
  protected def createSlots(variables: Seq[Variable]): Unit = {
    // No slots are created by default.
  }

  /** Creates all necessary tensors before applying the gradients. This function is called from within an op creation
    * context that uses as its name scope the name that users have chosen for the application of gradients. */
  protected def prepare(): Unit = {}

  /** Creates an op that finishes the gradients application. This function is called from within an op creation context
    * that uses as its name scope the name that users have chosen for the application of gradients.
    *
    * @param  updateOps Set of ops needed to apply the gradients and update the variable values.
    * @param  nameScope Name scope to use for all the ops created by this function.
    * @return Created op output.
    */
  protected def finish(updateOps: Set[Op], nameScope: String): Op = {
    ControlFlow.group(updateOps, nameScope)
  }

  /** Applies the updates corresponding to the provided gradient, to the provided variable.
    *
    * @param  gradient Gradient tensor.
    * @param  variable Variable.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  protected def applyDense(gradient: Op.Output, variable: Variable): Op

  /** Applies the updates corresponding to the provided gradient, to the provided variable.
    *
    * The [[Op.OutputIndexedSlices]] object specified by `gradient` in this function is by default pre-processed in
    * `applySparseDuplicateIndices` to remove duplicate indices (refer to that function's documentation for details).
    * Optimizers which can tolerate or have correct special cases for duplicate sparse indices may override
    * `applySparseDuplicateIndices` instead of this function, avoiding that overhead.
    *
    * @param  gradient Gradient tensor.
    * @param  variable Variable.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  protected def applySparse(gradient: Op.OutputIndexedSlices, variable: Variable): Op

  /** Applies the updates corresponding to the provided gradient (with potentially duplicate indices), to the provided
    * variable.
    *
    * Optimizers which override this method must deal with [[Op.OutputIndexedSlices]] objects such as the following:
    * `Op.OutputIndexedSlices(indices=[0, 0], values=[1, 1], denseShape=[1])`, which contain duplicate indices. The
    * correct interpretation in that case should be: `OutputIndexedSlices(values=[2], indices=[0], denseShape=[1])`.
    *
    * Many optimizers deal incorrectly with repeated indices when updating based on sparse gradients (e.g. summing
    * squares rather than squaring the sum, or applying momentum terms multiple times). Adding first is always the
    * correct behavior, so this is enforced here by reconstructing the [[Op.OutputIndexedSlices]] to have only unique
    * indices, and then calling [[applySparse]].
    *
    * Optimizers which deal correctly with repeated indices may instead override this method to avoid the induced
    * overhead.
    *
    * @param  gradient Gradient tensor.
    * @param  variable Variable.
    * @return Created op that applies the provided gradient to the provided variable.
    */
  protected def applySparseDuplicateIndices(gradient: Op.OutputIndexedSlices, variable: Variable): Op = {
    applySparse(deDuplicateOutputIndexedSlices(gradient), variable)
  }

  // TODO: [SLOTS] Add all of the slot creator functionality.
}

object Optimizer {
  /** Gets the appropriate variable processor to use for `variable`. */
  private[Optimizer] def getVariableProcessor(variable: Variable): VariableProcessor = variable match {
    // TODO: [VARIABLES] This is dummy for now.
    case v if v.op.opType == "VarHandleOp" => ResourceVariableProcessor(v)
    case v if v.op.opType == "SubmodelPort" => StreamingModelPortProcessor(v)
    case _ => throw new IllegalArgumentException(s"Unsupported variable op type '${variable.op.opType}'.")
  }

  /** Trait for abstracting over variables in the optimizers. */
  private[Optimizer] sealed trait VariableProcessor {
    /** Returns the optimization target for this variable. */
    def target: Op.Output

    /** Returns the update ops for updating this variable using the gradient provided by `gradient`. */
    def updateOp(optimizer: Optimizer, gradient: Op.OutputLike): Op
  }

  /** Variable processor for resource-based variables. */
  private[Optimizer] case class ResourceVariableProcessor(variable: Variable) extends VariableProcessor {
    override def target: Op.Output = variable.handle

    override def updateOp(optimizer: Optimizer, gradient: Op.OutputLike): Op = gradient match {
      case g: Op.Output => optimizer.applyDense(g, variable)
      case g: Op.OutputIndexedSlices => optimizer.applySparseDuplicateIndices(g, variable)
      case _ => throw new IllegalArgumentException(
        "Unsupported gradient type. Currently only 'Op.Output' and 'Op.OutputIndexedSlices' are supported.")
    }
  }

  /** Variable processor for streaming model ports. */
  private[Optimizer] case class StreamingModelPortProcessor(variable: Variable) extends VariableProcessor {
    // TODO: [VARIABLES] This is probably wrong.
    override def target: Op.Output = variable.handle

    override def updateOp(optimizer: Optimizer, gradient: Op.OutputLike): Op = gradient.op
  }

  /** Sums the values of the provided indexed slices associated with any non-unique indices and returns the resulting
    * de-duplicated version of the provided indexed slices.
    *
    * @param  input Indexed slices with potentially duplicate indices.
    * @return Indexed slices with de-duplicated indices and summed values slices associated with each unique index.
    */
  private[Optimizer] def deDuplicateOutputIndexedSlices(input: Op.OutputIndexedSlices): Op.OutputIndexedSlices = {
    val (uniqueIndices, newIndexPositions) = Basic.unique(input.indices)
    val summedValues = Math.unsortedSegmentSum(input.values, newIndexPositions, Basic.shape(uniqueIndices)(0))
    Op.OutputIndexedSlices(indices = uniqueIndices, values = summedValues, denseShape = input.denseShape)
  }
}
