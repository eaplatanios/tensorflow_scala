package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.Exception.InvalidDataTypeException
import org.platanios.tensorflow.api.types.DataType

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object Gradients {
  val logger = Logger(LoggerFactory.getLogger("Gradients"))

  //region Gradient Functions Registry

  private[this] val gradientFunctionRegistry = mutable.Map.empty[String, (Op, Seq[Op.OutputLike]) => Seq[Op.OutputLike]]

  ArrayOps.Gradients
  MathOps.Gradients

  /** Registers the provided gradient function to the gradient function registry.
    *
    * Note that if a gradient function for an op of the same type already exists in the registry, then it will be
    * overriden by the provided gradient function.
    *
    * @param  opType   Op type for which a gradient function is being registered.
    * @param  function Gradient function (takes op and output gradients as inputs and returns the input gradients).
    */
  def registerGradientFunction(opType: String, function: (Op, Seq[Op.OutputLike]) => Seq[Op.OutputLike]): Unit = {
    gradientFunctionRegistry.update(opType, function)
  }

  /** Registers the provided op type as non-differentiable (i.e., having `null` as its registered gradient function).
    *
    * @param  opType Op type to register as non-differentiable.
    */
  def registerNonDifferentiable(opType: String): Unit = {
    gradientFunctionRegistry.update(opType, null)
  }

  /** Gets the registered gradient function for the provided op type.
    *
    * @param  opType Op type whose gradient function is being looked up.
    * @return Gradient function registered for the provided op type.
    * @throws NoSuchElementException If no gradient has been registered for the provided op type.
    */
  @throws[NoSuchElementException]
  def getGradientFunction(opType: String): (Op, Seq[Op.OutputLike]) => Seq[Op.OutputLike] = {
    if (!gradientFunctionRegistry.contains(opType))
      throw new NoSuchElementException(s"No gradient registered for op type '$opType'.")
    gradientFunctionRegistry(opType)
  }

  //endregion Gradient Functions Registry

  // TODO: Convert variable to op output by using its handle.
  def gradients(
      ys: Seq[Op.Output], xs: Seq[Op.Output], dys: Seq[Op.OutputLike] = null, colocateGradientsWithOps: Boolean = false,
      aggregationMethod: AggregationMethod = AddAggregationMethod, name: String = "Gradients"): Seq[Op.OutputLike] = {
    // TODO: [CONTROL_FLOW] Gradients gating option.
    // The `gradients` variable collects the gradients received on each output endpoint of the op. The gradients for
    // each endpoint are initially collected as a sequence. When it is time to call the op's gradient function, for each
    // endpoint we aggregate the list of received gradients into a "add" operation, if there is more than one.
    val gradients = mutable.Map.empty[Op, mutable.Seq[Seq[Op.OutputLike]]]

    val ops = ys.map(_.op).toSet ++ xs.map(_.op).toSet ++ (if (dys != null) dys.map(_.op).toSet else Set.empty)
    Op.createWithNameScope(name, ops) {
      // The approach we take here is as follows: Create a list of all ops in the sub-graph between the ys and xs. Visit
      // these ops in reverse order of ids to ensure that when we visit an op the gradients with respect to its outputs
      // have been collected. Then, aggregate these gradients if needed, call the op's gradient function, and add the
      // generated gradients to the gradients for its input.

      // Initialize the pending counts for ops in the connected sub-graph between the ys and xs.
      val destinationOps = {
        if (ys.length > 1)
          ys.map(y => if (y.consumers.nonEmpty) ArrayOps.identity(y).op else y.op).toSet
        else
          ys.map(_.op).toSet
      }
      val sourceOps = xs.map(_.op).toSet
      val pendingCounts = initialPendingCounts(sourceOps, destinationOps, colocateGradientsWithOps)

      // Add the initial gradients for the ys.
      val dyInitial = defaultGradients(dys, ys, colocateGradientsWithOps)
      for ((y, dy) <- (ys, dyInitial).zipped)
        setGradient(gradients, y, dy)

      // Initialize the ops queue with the destination ops. We filter the destination ops based on whether one output
      // gradient relies on another output's gradient.
      val opsQueue = mutable.Queue[Op](destinationOps.filter(pendingCounts(_) == 0).toSeq: _*)

      // Stop ops form the frontier of the forward graph before which back-propagation should stop. Ops in this set will
      // not be differentiated. This set is defined as the subset of `sourceOps` containing ops that have no predecessor
      // in `sourceOps`. An op has predecessors in `sourceOps` if and only if `pendingCounts(op) > 0`.
      val stopOps = sourceOps.filter(op => op.inputs.forall(i => pendingCounts(i.op) <= 0))

      while (opsQueue.nonEmpty) {
        val op = opsQueue.dequeue()
        maybeColocateWith(op, colocateGradientsWithOps) {
          val opGradients = aggregationMethod.aggregateGradients(gradients, op)
          val hasOutputGradients = opGradients.nonEmpty
          val gradientFunction: (Op, Seq[Op.OutputLike]) => Seq[Op.OutputLike] = {
            if (hasOutputGradients && !stopOps.contains(op)) {
              getGradientFunction(op.opType)
            } else {
              null
            }
          }

          if (hasOutputGradients && gradientFunction != null) {
            // Note that, the gradient aggregation not computing a value for the i'th output, means that the cost does
            // not depend on output i and therefore the gradient with respect to that output is 0.
            for ((gradient, outputIndex) <- opGradients.zipWithIndex) {
              // Only floating-point outputs get a zero gradient. Gradient functions should ignore the gradient for
              // other outputs.
              val output = op.outputs(outputIndex)
              if (gradient.isEmpty && isTrainable(output))
                opGradients(outputIndex) = Seq(ArrayOps.zerosLike(output.toOpOutput))
            }

            // Compute the actual op gradients.
            Op.createWith(nameScope = s"${op.name}Grad") {
              // TODO: [CONTEXT] Add support for original op context.
              val outputGradients = opGradients.map(_.head)
              val inputGradients = maybeCompile(name, op, () => gradientFunction(op, outputGradients))
              verifyGradients(op, inputGradients)
              logGradients(op, outputGradients, inputGradients)
              op.inputs.zip(inputGradients).filter(_._2 ne null).foreach(i => {
                i._2 match {
                  case gradient: Op.Output if i._1.dataType != DataType.Resource => gradient.setShape(i._1.shape)
                  case _ =>
                }
                setGradient(gradients, i._1, i._2)
              })
            }
          }
        }

        // Update the pending counts for the inputs of `op` and enqueue ready ops.
        op.inputs.foreach(input => {
          pendingCounts.update(input.op, pendingCounts(input.op) - 1)
          if (pendingCounts(input.op) == 0)
            opsQueue.enqueue(input.op)
          // TODO: [CONTROL_FLOW] Some control flow gradient logic should go here.
        })
      }
    }

    // Collect the aggregated gradients for the requested tensors and return them.
    xs.map(x => {
      val collectedGradients = gradients.get(x.op).map(_.apply(x.index))
      if (collectedGradients.isDefined && collectedGradients.get.length > 1)
        throw new IllegalArgumentException("The gradients should have been aggregated by now.")
      collectedGradients.map(_.head).orNull
    })
  }

  /** If `colocateGradientsWithOps` is `true`, then all ops created within `block` will be colocated with `op`.
    *
    * @param  op                       Op to maybe colocate with.
    * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
    *                                  ops.
    * @param  block                    Block of code to execute using the specified colocation ops.
    * @return Return value of `block`.
    */
  private[this] def maybeColocateWith[R](op: Op, colocateGradientsWithOps: Boolean)(block: => R): R = {
    if (colocateGradientsWithOps)
      Op.createWith(colocationOps = Set[Op](op))(block)
    else
      block
  }

  // TODO: [FUNCTIONAL] Symbolic gradient ('_SymGrad').

  /** If the op was marked as compiled, this function compiles the calculation in `gradientFunction` (using XLA) and
    * returns the result of `gradientFunction`. Otherwise, it simply returns the result of `gradientFunction`.
    *
    * @param  nameScope        Name scope to use for the gradient ops.
    * @param  op               Op whose gradients are being computed.
    * @param  gradientFunction Function that computes the gradients for `op`.
    * @return Created gradients op.
    */
  private[this] def maybeCompile(
      nameScope: String, op: Op, gradientFunction: () => Seq[Op.OutputLike]): Seq[Op.OutputLike] = {
    // TODO: [FUNCTIONAL] Add extra 'func' argument.
    val cleanNameScope = nameScope.stripSuffix("/").replace('/', '_')
    try {
      val xlaCompile = op.booleanAttribute("_XlaCompile")
      if (!xlaCompile) {
        gradientFunction() // Exit early
      } else {
        val xlaSeparateCompileGradient = op.booleanAttribute("_XlaSeparateCompiledGradients")
        val xlaScope = op.stringAttribute("_XlaScope")
        // If the gradients are supposed to be compiled separately, we give them an '_XlaScope' name that is based on
        // the name_scope of the gradients. Otherwise, they just inherit the existing '_XlaScope' name, which lets them
        // be merged together with the non-gradient computation.
        val xlaGradientsScope = if (xlaSeparateCompileGradient) s"${xlaScope}_grad_$cleanNameScope" else xlaScope
        Op.createWith(attributes = Map("_XlaCompile" -> xlaCompile, "_XlaScope" -> xlaGradientsScope)) {
          gradientFunction()
        }
      }
    } catch {
      case _: IllegalArgumentException =>
        gradientFunction() // Something went wrong and so we exit
    }
  }

  /** Returns a boolean value indicating whether the data type of `tensor` is trainable. This means whether its
    * gradients can be computed. */
  private[this] def isTrainable(tensor: Op.OutputLike): Boolean = {
    Set[DataType](DataType.Float32, DataType.Float64).contains(tensor.dataType)
    // TODO: !! FLOAT16, COMPLEX64, COMPLEX128.
  }

  /** Computes default values for the provided gradients, and checks whether their data types are correct.
    *
    * @param  dys                      Sequence containing tensor gradients.
    * @param  ys                       Sequence containing the variables corresponding to `dys`.
    * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
    *                                  ops.
    * @return Sequence containing the default gradient values.
    * @throws InvalidDataTypeException If the gradient tensor data types are not compatible with the input data types.
    */
  @throws[InvalidDataTypeException]
  private[this] def defaultGradients(
      dys: Seq[Op.OutputLike], ys: Seq[Op.OutputLike], colocateGradientsWithOps: Boolean): Seq[Op.OutputLike] = {
    ys.zip(if (dys != null) dys else Seq.fill[Op.OutputLike](ys.length)(null)).map {
      case (y, dy) =>
        if (dy eq null) {
          if (y.dataType.isComplex)
            throw InvalidDataTypeException(
              s"Gradients of complex tensors must set 'gradients' (variable.dataType = '${y.dataType}').")
          maybeColocateWith(y.op, colocateGradientsWithOps) {
            y match {
              case o: Op.Output => ArrayOps.onesLike(o)
              case o: Op.OutputIndexedSlices =>
                if (o.denseShape eq null)
                  throw new IllegalArgumentException(
                    "The dense shape of output indexed slices must be known in order to obtain their gradients.")
                val values = ArrayOps.fill(o.denseShape, 1.0)
                Op.OutputIndexedSlices(indices = o.indices, values = values, denseShape = o.denseShape)
              case o: Op.SparseOutput =>
                val values = ArrayOps.fill(o.denseShape, 1.0)
                Op.SparseOutput(indices = o.indices, values = values, denseShape = o.denseShape)
            }
          }
        } else if (y.dataType.isFloatingPoint || y.dataType.isInteger) {
          if (!dy.dataType.isFloatingPoint && !dy.dataType.isInteger)
            throw InvalidDataTypeException(
              s"Gradient data type '${dy.dataType}' generated for real or integer-valued tensor '$y' with data type " +
                  s"'${y.dataType}' must be real or integer.")
          dy
        } else if (y.dataType.isComplex) {
          if (!dy.dataType.isComplex)
            throw InvalidDataTypeException(
              s"Gradient data type '${dy.dataType}' generated for complex-valued tensor '$y' with data type " +
                  s"'${y.dataType}' must be complex.")
          dy
        } else {
          throw InvalidDataTypeException(
            s"Tensor '$y' with data type '${y.dataType}' must be numeric in order to obtain a default gradient.")
        }
    }
  }

  /** Initializes the back-propagation input counts for ops between two sets of ops.
    *
    * 'outputMap(op)' indicates the number of back-propagation inputs to this op.
    *
    * @param  sourceOps                Set of source ops.
    * @param  destinationOps           Set of destination ops.
    * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
    *                                  ops.
    * @return Tuple containing: (1) Map from op to the number of back-propagation inputs to this op, and (2) a control
    *         flow state object which is not `null` if the ops between `sources` and `destinations` contain control flow
    *         loops.
    */
  private[this] def initialPendingCounts(
      sourceOps: Set[Op], destinationOps: Set[Op], colocateGradientsWithOps: Boolean): mutable.Map[Op, Int] = {
    // TODO: [CONTROL_FLOW]
    // Mark ops reached when going from 'sources' to 'destinations'
    val reached = mutable.Set[Op](destinationOps.toSeq: _*)
    val reachedQueue = mutable.Queue[Op](sourceOps.toSeq: _*)
    while (reachedQueue.nonEmpty) {
      val op = reachedQueue.dequeue()
      if (!reached.contains(op)) {
        reached += op
        op.outputs.foreach(o => reachedQueue.enqueue(o.consumers.map(_.op): _*))
      }
    }

    // Mark ops between 'sources' and 'destinations'
    val between = mutable.Set.empty[Op]
    // TODO: [CONTROL_FLOW] Do we need the list aside from the set?
    val betweenList = mutable.ListBuffer.empty[Op]
    val betweenQueue = mutable.Queue[Op](destinationOps.toSeq: _*)
    while (betweenQueue.nonEmpty) {
      val op = betweenQueue.dequeue()
      if (reached.contains(op)) {
        between += op
        betweenList += op
        reached -= op // Done so we don't go through the same ops twice
        op.inputs.foreach(i => betweenQueue.enqueue(i.op))
      }
    }

    // Initialize the pending counts for the between ops
    val pendingCounts = mutable.Map.empty[Op, Int]
    betweenList.flatMap(_.inputs).map(_.op).filter(between.contains).foreach(input => {
      pendingCounts.update(input, pendingCounts.getOrElse(input, 0) + 1)
    })

    pendingCounts
  }

  /** Adds the provided `gradient` to the sequence of `output`'s gradients that have been collected so far.
    *
    * @param  gradients Map where the collected gradients are stored.
    * @param  output    Op output whose gradient is provided.
    * @param  gradient  Gradient of `output` to add to the collected gradients.
    */
  private[this] def setGradient(
      gradients: mutable.Map[Op, mutable.Seq[Seq[Op.OutputLike]]], output: Op.Output, gradient: Op.OutputLike): Unit = {
    val opGradients = gradients.getOrElseUpdate(
      output.op, mutable.Seq(output.op.outputs.map(_ => Seq.empty[Op.OutputLike]): _*))
    opGradients(output.index) :+= gradient
  }

  /** Verifies that the provided `gradients` are valid in number and data type.
    *
    * @param  op        Op for which the gradients are being generated.
    * @param  gradients Sequence containing the generated gradients.
    * @throws IllegalStateException    If the generated gradients are not valid in number.
    * @throws InvalidDataTypeException If the generated gradients are not valid in data type.
    */
  @throws[IllegalStateException]
  @throws[InvalidDataTypeException]
  private[this] def verifyGradients(op: Op, gradients: Seq[Op.OutputLike]): Unit = {
    if (op.inputs.length != gradients.length)
      throw new IllegalStateException(
        s"The number of gradients (${gradients.length}) generated for op '$op' do not match its number of inputs " +
            s"(${op.inputs.length}).")
    for ((input, gradient) <- op.inputs.zip(gradients)) {
      if (gradient ne null) {
        if (gradient.dataType.isFloatingPoint && !input.dataType.isFloatingPoint)
          throw InvalidDataTypeException(
            s"Gradient data type '${gradient.dataType}' generated for real-valued op '$op' with data type " +
                s"'${input.dataType}' must be real.")
        else if (gradient.dataType.isComplex && !input.dataType.isComplex)
          throw InvalidDataTypeException(
            s"Gradient data type '${gradient.dataType}' generated for complex-valued op '$op' with data type " +
                s"'${input.dataType}' must be complex.")
        else
          throw InvalidDataTypeException(
            s"Gradient data type '${gradient.dataType}' generated for op '$op' with data type '${input.dataType}' " +
                s"must be either real or complex.")
      }
    }
  }

  /** Logs the input and output gradients of the provided op.
    *
    * @param  op              Op.
    * @param  outputGradients Output gradients of op.
    * @param  inputGradients  Input gradients of op.
    */
  private[this] def logGradients(
      op: Op, outputGradients: Seq[Op.OutputLike], inputGradients: Seq[Op.OutputLike]): Unit = {
    logger.debug(s"Gradients for op '${op.name}':")
    logger.debug(s"  in  --> ${outputGradients.filter(_ != null).map(_.name).mkString(", ")}")
    logger.debug(s"  out --> ${inputGradients.filter(_ != null).map(_.name).mkString(", ")}")
  }

  /** Aggregation method used to combine gradients.
    *
    * Computing partial derivatives can require aggregating gradient contributions. All such aggregation methods are
    * represented as objects extending this trait.
    */
  sealed trait AggregationMethod {
    /** Aggregate the gradients for op `op`.
      *
      * @param  gradients Map where the collected gradients are stored. The gradient sequences corresponding to `op`
      *                   will be replaced with sequences containing a single element corresponding to the aggregated
      *                   gradient.
      * @param  op        Op whose gradients to aggregate.
      */
    private[Gradients] def aggregateGradients(
        gradients: mutable.Map[Op, mutable.Seq[Seq[Op.OutputLike]]], op: Op): mutable.Seq[Seq[Op.OutputLike]] = {
      val grads = gradients.getOrElse(op, mutable.Seq.empty[Seq[Op.OutputLike]])
      grads.filter(_.length > 1).zipWithIndex.foreach(g => grads(g._2) = Seq[Op.OutputLike](aggregateGradients(g._1)))
      grads
    }

    /** Aggregates the gradients in `gradient` into a single gradient tensor.
      *
      * @param  gradients Sequence of gradients to aggregate.
      * @return Aggregated gradient tensor.
      */
    private[Gradients] def aggregateGradients(gradients: Seq[Op.OutputLike]): Op.OutputLike
  }

  /** Gradient aggregation method that simply adds up the collected gradients. */
  object AddAggregationMethod extends AggregationMethod {
    override private[Gradients] def aggregateGradients(gradients: Seq[Op.OutputLike]): Op.OutputLike = {
      gradients match {
        case g: Seq[Op.Output] =>
          // This function adds op outputs from potentially different devices.
          // We add the tensors of each device separately first, and we then add up the partial results.
          MathOps.addN(g.groupBy(_.device).values.map(
            deviceGradients =>
              Op.colocateWith(Set[Op](g.head.op), ignoreExisting = true) {
                MathOps.addN(deviceGradients.map(_.toOpOutput).toArray)
              }).toArray)
        case g: Seq[Op.OutputIndexedSlices] =>
          ???
        case _ => throw new IllegalArgumentException(
          "The gradients being aggregated need to be all of type 'Op.Output' or 'Op.OutputIndexedSlices'.")
      }
    }
  }

  // TODO: [GRADIENTS] Add support for more aggregation methods.
}
