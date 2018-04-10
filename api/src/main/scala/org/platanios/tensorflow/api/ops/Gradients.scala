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

package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.exception.InvalidDataTypeException
import org.platanios.tensorflow.api.ops
import org.platanios.tensorflow.api.ops.control_flow.{Context, ControlFlow, GradientState}
import org.platanios.tensorflow.api.types._
import org.platanios.tensorflow.jni.{Graph => NativeGraph, Output => NativeOutput}

import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
private[ops] object Gradients {
  val logger = Logger(LoggerFactory.getLogger("Gradients"))

  private[ops] trait API {
    val Gradients        : ops.Gradients.type          = ops.Gradients
    val GradientsRegistry: ops.Gradients.Registry.type = ops.Gradients.Registry
  }

  // TODO: [API] Expose in "tf".
  // TODO: [DOC] Document the "gradients" function.
  def gradients(
      ys: Seq[Output],
      xs: Seq[Output],
      dys: Seq[OutputLike] = null,
      gateGradients: Boolean = false,
      aggregationMethod: AggregationMethod = AddAggregationMethod,
      colocateGradientsWithOps: Boolean = false,
      name: String = "Gradients"
  ): Seq[OutputLike] = Op.currentGraph.synchronized {
    // The `accumulatedGradients` variable collects the gradients received on each output endpoint of the op. The
    // gradients for each endpoint are initially collected as a sequence. When it is time to call the op's gradient
    // function, for each endpoint we aggregate the list of received gradients into a "add" operation, if there is more
    // than one.
    val accumulatedGradients = mutable.Map.empty[Op, mutable.Seq[Seq[OutputLike]]]

    val ops = ys.map(_.op).toSet ++ xs.map(_.op).toSet ++ (if (dys != null) dys.map(_.op).toSet else Set.empty)
    Op.createWithNameScope(name, ops) {
      // The approach we take here is as follows: Create a list of all ops in the sub-graph between the ys and xs. Visit
      // these ops in reverse order of ids to ensure that when we visit an op the gradients with respect to its outputs
      // have been collected. Then, aggregate these gradients if needed, call the op's gradient function, and add the
      // generated gradients to the gradients for its input.

      // Initialize the pending counts for ops in the connected sub-graph between the ys and xs.
      val sourceOps = xs.map(_.op).toSet
      val destinationOps = {
        if (ys.lengthCompare(1) > 0)
          ys.map(y => if (y.consumers.nonEmpty) Basic.identity(y).op else y.op).toSet
        else
          ys.map(_.op).toSet
      }

      // `pendingCounts(op)` is a count-down counter for the expected gradients to accumulate for `op`. When
      // `pendingCounts(op)` becomes zero, we have collected all the backpropagation gradients for all outputs of `op`.
      val (pendingCounts, controlFlowGradientState) = initialPendingCounts(
        sourceOps, destinationOps, colocateGradientsWithOps)

      // `readyOps` keeps track of ops that have been completely processed. We initialize it with the destination ops.
      // We filter the destination ops based on whether one output gradient relies on another output's gradient.
      val readyOps = mutable.Queue[Op](destinationOps.filter(pendingCounts.getOrElse(_, 0) == 0).toSeq: _*)

      // Add the initial gradients for the ys.
      val dyInitial = initialGradients(ys, dys, colocateGradientsWithOps)
      for ((y, dy) <- (ys, dyInitial).zipped)
        setGradient(accumulatedGradients, y, dy)

      controlFlowGradientState.foreach(state => {
        state.processUnusedLoopExits(pendingCounts, destinationOps).filter(isTrainable).foreach(loopExit => {
          setGradient(accumulatedGradients, loopExit, state.zerosLikeForExit(loopExit))
          readyOps.enqueue(loopExit.op)
        })
      })

      // Stop ops form the frontier of the forward graph before which back-propagation should stop. Ops in this set will
      // not be differentiated. This set is defined as the subset of `sourceOps` containing ops that have no predecessor
      // in `sourceOps`. An op has predecessors in `sourceOps` if and only if `pendingCounts(op) > 0`.
      val stopOps = sourceOps.filter(_.inputs.forall(i => pendingCounts.getOrElse(i.op, 0) <= 0))

      while (readyOps.nonEmpty) {
        val op = readyOps.dequeue()
        maybeColocateWith(op, colocateGradientsWithOps) {
          controlFlowGradientState.foreach(_.enterGradientWhileLoopContext(op, before = true))
          val opGradients = aggregationMethod.aggregateGradients(accumulatedGradients, op)
          controlFlowGradientState.foreach(_.exitGradientWhileLoopContext(op, before = true))
          val hasOutputGradients = opGradients.nonEmpty
          val gradientFunction: Registry.GradientFunction = {
            if (hasOutputGradients && !stopOps.contains(op)) {
              Registry(op.opType)
            } else {
              null
            }
          }

          controlFlowGradientState.foreach(_.enterGradientWhileLoopContext(op, before = false))
          if (hasOutputGradients && gradientFunction != null) {
            // Note that, the gradient aggregation not computing a value for the i'th output, means that the cost does
            // not depend on output i and therefore the gradient with respect to that output is 0.
            for ((gradient, outputIndex) <- opGradients.zipWithIndex) {
              // Only floating-point outputs get a zero gradient. Gradient functions should ignore the gradient for
              // other outputs.
              val output = op.outputs(outputIndex)
              if (gradient.isEmpty && isTrainable(output))
              // TODO: !!! [GRADIENTS] Gradients of resource handles might be an issue here because of the zeros.
                opGradients(outputIndex) = Seq({
                  controlFlowGradientState
                      .map(_.zerosLike(op, outputIndex))
                      .getOrElse(Some(Context.zerosLikeOutsideLoop(op, outputIndex)))
                      .orNull
                })
            }

            // Compute the actual op gradients.
            Op.createWith(nameScope = s"${op.name}Gradient") {
              // TODO: [CONTEXT] Add support for original op context.
              val outputGradients = opGradients.map(_.headOption.orNull)
              var inputGradients = maybeCompile(name, op, () => gradientFunction(op, outputGradients))
              if (gateGradients && inputGradients.count(_ != null) > 1) {
                Op.createWith(device = null) {
                  Op.colocateWith(Set.empty[Op], ignoreExisting = true) {
                    inputGradients = ControlFlow.tuple(inputGradients.toArray).toSeq
                  }
                }
              }
              val nInp = op.inputs.length
              val nGrd = inputGradients.length
              assert(nInp == nGrd, s"Gradients size ($nGrd) for op '$op' does not match inputs size ($nInp).")
              logGradients(op, outputGradients, inputGradients)
              // TODO: Report somehow the non-differentiable ops in the graph. This is currently hard to debug.
              op.inputs.zip(inputGradients).filter(_._2 != null).foreach(i => {
                i._2 match {
                  case gradient: Output if i._1.dataType != RESOURCE => gradient.setShape(i._1.shape)
                  case _ =>
                }
                setGradient(accumulatedGradients, i._1, i._2)
              })
            }
          }
          controlFlowGradientState.foreach(_.exitGradientWhileLoopContext(op, before = false))
        }

        // Update the pending counts for the inputs of `op` and enqueue ready ops.
        op.inputs.foreach(input => {
          pendingCounts.update(input.op, pendingCounts.getOrElse(input.op, 0) - 1)
          var ready = pendingCounts(input.op) == 0
          if (!ready)
            controlFlowGradientState.foreach(_ => {
              ready = pendingCounts(input.op) > 0 && ControlFlow.isLoopSwitch(input.op)
            })
          if (ready) {
            if (ControlFlow.isLoopExit(input.op)) {
              // If `input` is an exit without real gradient, defer processing them.
              controlFlowGradientState.flatMap(_.getGradientLoopState(input.op, before = false)).foreach(state => {
                state.deferredExits += input
                state.pendingExitsCount -= 1
                if (state.pendingExitsCount == 0) {
                  // We now have all the exits and so we process them.
                  var hasRealGradient = false
                  state.deferredExits.foreach(exit => {
                    if (accumulatedGradients.get(exit.op).exists(_.exists(_.exists(_ != null)))) {
                      hasRealGradient = true
                      readyOps.enqueue(exit.op)
                    } else {
                      state.unusedExits += exit
                    }
                  })
                  if (hasRealGradient) {
                    // For an unused exit, if it has floating-point outputs, we back-propagate a zero gradient.
                    // Otherwise, we just ignore it.
                    state.unusedExits.foreach(exit => {
                      if (isTrainable(exit))
                        setGradient(accumulatedGradients, exit, controlFlowGradientState.get.zerosLikeForExit(exit))
                      readyOps.enqueue(exit.op)
                    })
                  } else {
                    // All exits are "unused" and so we use `null` as the gradient.
                    state.unusedExits.foreach(exit => readyOps.enqueue(exit.op))
                  }
                }
              })
            } else {
              readyOps.enqueue(input.op)
            }
          }
        })
      }

      controlFlowGradientState.foreach(_.postProcess())
    }

    // Collect the aggregated gradients for the requested tensors and return them.
    xs.map(x => {
      val gradients = accumulatedGradients.get(x.op).map(_.apply(x.index))
      if (gradients.isDefined && gradients.get.lengthCompare(1) > 0)
        throw new IllegalArgumentException("The gradients should have been aggregated by now.")
      gradients.map(_.head).orNull
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
      Op.colocateWith(Set(op))(block)
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
      nameScope: String, op: Op, gradientFunction: () => Seq[OutputLike]): Seq[OutputLike] = {
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
      case _: IllegalArgumentException => gradientFunction() // Something went wrong and so we exit
    }
  }

  /** Returns a boolean value indicating whether the data type of `tensor` is trainable. This means whether its
    * gradients can be computed. */
  private[this] def isTrainable(tensor: OutputLike): Boolean = {
    Set[DataType](FLOAT16, FLOAT32, FLOAT64, COMPLEX64, COMPLEX128).contains(tensor.dataType)
  }

  /** Computes initial values for the provided gradients, and checks whether their data types are correct.
    *
    * @param  ys                       Sequence containing the variables corresponding to `dys`.
    * @param  dys                      Sequence containing tensor gradients.
    * @param  colocateGradientsWithOps Boolean value indicating whether to colocate the gradient ops with the original
    *                                  ops.
    * @return Sequence containing the default gradient values.
    * @throws InvalidDataTypeException If the gradient tensor data types are not compatible with the input data types.
    */
  @throws[InvalidDataTypeException]
  private[this] def initialGradients(
      ys: Seq[OutputLike], dys: Seq[OutputLike], colocateGradientsWithOps: Boolean): Seq[OutputLike] = {
    ys.zip(if (dys != null) dys else Seq.fill[OutputLike](ys.length)(null)).zipWithIndex.map {
      case ((y, dy), index) =>
        if (dy == null) {
          if (y.dataType.isComplex)
            throw InvalidDataTypeException(
              s"Gradients of complex tensors must set 'gradients' (variable.dataType = '${y.dataType}').")
          maybeColocateWith(y.op, colocateGradientsWithOps) {
            y match {
              case o: Output => Basic.onesLike(o, name = s"Gradients_$index")
              case o: OutputIndexedSlices =>
                if (o.denseShape == null)
                  throw new IllegalArgumentException(
                    "The dense shape of output indexed slices must be known in order to obtain their gradients.")
                Basic.ones(o.dataType, o.denseShape, name = s"Gradients_$index")
              case o: SparseOutput =>
                Basic.ones(o.dataType, o.denseShape, name = s"Gradients_$index")
            }
          }
        } else {
          if (y.dataType.isFloatingPoint || y.dataType.isInteger) {
            if (!dy.dataType.isFloatingPoint && !dy.dataType.isInteger)
              throw InvalidDataTypeException(
                s"Gradient data type '${dy.dataType}' generated for real or integer-valued tensor '$y' with data type " +
                    s"'${y.dataType}' must be real or integer.")
          } else if (y.dataType.isComplex) {
            if (!dy.dataType.isComplex)
              throw InvalidDataTypeException(
                s"Gradient data type '${dy.dataType}' generated for complex-valued tensor '$y' with data type " +
                    s"'${y.dataType}' must be complex.")
          } else {
            throw InvalidDataTypeException(
              s"Tensor '$y' with data type '${y.dataType}' must be numeric in order to obtain a default gradient.")
          }
          // Create a gradients tensor in the name scope of the gradients. This is required in order for tensor arrays
          // to identify which gradient call a gradient value is coming from.
          dy match {
            case o: Output => Basic.identity(o, name = s"Gradients_$index")
            case o: OutputIndexedSlices =>
              OutputIndexedSlices(
                Basic.identity(o.indices, name = s"Gradients_${index}_Indices"),
                Basic.identity(o.values, name = s"Gradients_${index}_Values"),
                if (o.denseShape == null)
                  o.denseShape
                else
                  Basic.identity(o.denseShape, name = s"Gradients_${index}_DenseShape"))
            case o: SparseOutput =>
              SparseOutput(
                Basic.identity(o.indices, name = s"Gradients_${index}_Indices"),
                Basic.identity(o.values, name = s"Gradients_${index}_Values"),
                if (o.denseShape == null)
                  o.denseShape
                else
                  Basic.identity(o.denseShape, name = s"Gradients_${index}_DenseShape"))
          }
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
    *         flow gradient state object which is not `None` if the ops between `sources` and `destinations` contain
    *         control flow loops.
    */
  private[this] def initialPendingCounts(
      sourceOps: Set[Op], destinationOps: Set[Op],
      colocateGradientsWithOps: Boolean): (mutable.Map[Op, Int], Option[GradientState]) = {
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

    // `controlFlowGradientState` is `None` if there are no while loops.
    val controlFlowGradientState = GradientState.maybeCreate(between, betweenList, colocateGradientsWithOps)

    // Initialize the pending counts for the between ops
    val pendingCounts = mutable.Map.empty[Op, Int]
    betweenList.flatMap(_.inputs).map(_.op).filter(between.contains).foreach(input => {
      pendingCounts.update(input, pendingCounts.getOrElse(input, 0) + 1)
    })

    (pendingCounts, controlFlowGradientState)
  }

  /** Adds the provided `gradient` to the sequence of `output`'s gradients that have been collected so far.
    *
    * @param  gradients Map where the collected gradients are stored.
    * @param  output    Op output whose gradient is provided.
    * @param  gradient  Gradient of `output` to add to the collected gradients.
    */
  private[this] def setGradient(
      gradients: mutable.Map[Op, mutable.Seq[Seq[OutputLike]]], output: Output, gradient: OutputLike): Unit = {
    val opGradients = gradients.getOrElseUpdate(
      output.op, mutable.Seq(output.op.outputs.map(_ => Seq.empty[OutputLike]): _*))
    if (ControlFlow.isLoopSwitch(output.op))
      opGradients(output.index) = Seq(gradient)
    else
      opGradients(output.index) :+= gradient
  }

  /** Logs the input and output gradients of the provided op.
    *
    * @param  op              Op.
    * @param  outputGradients Output gradients of op.
    * @param  inputGradients  Input gradients of op.
    */
  private[this] def logGradients(op: Op, outputGradients: Seq[OutputLike], inputGradients: Seq[OutputLike]): Unit = {
    logger.debug(s"Gradients for op '${op.name}':")
    logger.debug(s"  in  --> ${outputGradients.filter(_ != null).map(_.name).mkString(", ")}")
    logger.debug(s"  out --> ${inputGradients.filter(_ != null).map(_.name).mkString(", ")}")
  }

  sealed trait GatingMethod
  object NoGating extends GatingMethod
  object OpGating extends GatingMethod
  object GraphGating extends GatingMethod

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
        gradients: mutable.Map[Op, mutable.Seq[Seq[OutputLike]]], op: Op): mutable.Seq[Seq[OutputLike]] = {
      val opGradients = gradients.getOrElse(op, mutable.Seq.empty[Seq[OutputLike]])
      if (ControlFlow.isLoopSwitch(op)) {
        opGradients
      } else {
        opGradients.zipWithIndex.foreach {
          case (grads, index) =>
            if (grads.length < 2) {
              grads
            } else {
              opGradients(index) = Seq[OutputLike](aggregateGradients(grads.filter(_ != null)))
            }
        }
        opGradients
      }
    }

    /** Aggregates the gradients in `gradient` into a single gradient tensor.
      *
      * @param  gradients Sequence of gradients to aggregate.
      * @return Aggregated gradient tensor.
      */
    private[Gradients] def aggregateGradients(gradients: Seq[OutputLike]): OutputLike
  }

  /** Gradient aggregation method that simply adds up the collected gradients. */
  object AddAggregationMethod extends AggregationMethod {
    override private[Gradients] def aggregateGradients(gradients: Seq[OutputLike]): OutputLike = {
       if (gradients.forall(_.isInstanceOf[Output])) {
        // This function adds op outputs from potentially different devices.
        // We add the tensors of each device separately first, and we then add up the partial results.
        val deviceContributions = gradients.groupBy(_.device).toSeq.sortBy(_._1).map {
          case (_, outputs) =>
            Op.colocateWith(Set[Op](gradients.head.op), ignoreExisting = true) {
              Math.addN(outputs.map(_.asInstanceOf[Output]))
            }
        }
        Math.addN(deviceContributions)
      } else if (gradients.forall(_.isInstanceOf[OutputIndexedSlices])) {
         def addNOutputIndexedSlices(gradients: Seq[OutputIndexedSlices]): OutputIndexedSlices = {
           if (gradients.isEmpty) {
             throw new IllegalArgumentException(
               "Can not aggregate empty gradients list.")
           } else if (gradients.length == 1) {
             gradients.head
           } else {
             OutputIndexedSlices(
               Basic.concatenate(gradients.map(_.indices)),
               Basic.concatenate(gradients.map(_.values)),
               gradients.head.denseShape)
           }
         }
         val deviceContributions = gradients.groupBy(_.device).toSeq.sortBy(_._1).map {
           case (_, outputs) =>
             Op.colocateWith(Set[Op](gradients.head.op), ignoreExisting = true) {
               addNOutputIndexedSlices(outputs.map(_.asInstanceOf[OutputIndexedSlices]))
             }
         }
         addNOutputIndexedSlices(deviceContributions)
      } else {
        throw new IllegalArgumentException(
          "The gradients being aggregated need to be all of type 'Output' or 'OutputIndexedSlices'.")
      }
    }
  }

  /** Gradient aggregation method that simply adds up the collected gradients, without first waiting for all of them to
    * become available at once.
    *
    * The benefit of using this method is that its inputs can be combined in any order and this can allow the expression
    * to be evaluated with a smaller memory footprint. With this method, it is possible to compute a sum of terms which
    * are much larger than total GPU memory.
    */
  object AccumulateAggregationMethod extends AggregationMethod {
    override private[Gradients] def aggregateGradients(gradients: Seq[OutputLike]): OutputLike = {
      if (gradients.forall(_.isInstanceOf[Output])) {
        Math.accumulateN(gradients.map(_.asInstanceOf[Output]))
      } else if (gradients.forall(_.isInstanceOf[OutputIndexedSlices])) {
        def addNOutputIndexedSlices(gradients: Seq[OutputIndexedSlices]): OutputIndexedSlices = {
          if (gradients.isEmpty) {
            throw new IllegalArgumentException(
              "Can not aggregate empty gradients list.")
          } else if (gradients.length == 1) {
            gradients.head
          } else {
            OutputIndexedSlices(
              Basic.concatenate(gradients.map(_.indices)),
              Basic.concatenate(gradients.map(_.values)),
              gradients.head.denseShape)
          }
        }
        val deviceContributions = gradients.groupBy(_.device).toSeq.sortBy(_._1).map {
          case (_, outputs) =>
            Op.colocateWith(Set[Op](gradients.head.op), ignoreExisting = true) {
              addNOutputIndexedSlices(outputs.map(_.asInstanceOf[OutputIndexedSlices]))
            }
        }
        addNOutputIndexedSlices(deviceContributions)
      } else {
        throw new IllegalArgumentException(
          "The gradients being aggregated need to be all of type 'Output' or 'OutputIndexedSlices'.")
      }
    }
  }

  /** Registry that contains the gradient functions to be used when creating gradient ops. Gradient functions for all
    * types of ops that are being differentiated need to be registered using either the [[Registry.register]] or the
    * [[Registry.registerNonDifferentiable]] functions. In an attempt to obtain the gradient of an op whose type has no
    * gradient function registered, a [[NoSuchElementException]] will be thrown. */
  object Registry {
    type GradientFunction = (Op, Seq[OutputLike]) => Seq[OutputLike]

    private[this] val registry = mutable.Map.empty[String, GradientFunction]

    /** Registers the provided gradient function to the gradient function registry.
      *
      * Note that if a gradient function for an op of the same type already exists in the registry, then it will be
      * overriden by the provided gradient function.
      *
      * @param  opType   Op type for which a gradient function is being registered.
      * @param  function Gradient function (takes op and output gradients as inputs and returns the input gradients).
      */
    def register(opType: String, function: GradientFunction): Unit = registry.update(opType, function)

    /** Registers the provided op type as non-differentiable (i.e., having `null` as its registered gradient function).
      *
      * This function should *not* be used for ops that have a well-defined gradient that is not yet implemented. It
      * should only be used when defining a new op type. It may be used for ops such as `size` that are not
      * differentiable.
      *
      * The gradient computed for 'opType' will then propagate zeros.
      *
      * For ops that have a well-defined gradient but are not yet implemented, no declaration should be made, and an error
      * *must* be thrown if an attempt to request their gradient is made.
      *
      * @param  opType Op type to register as non-differentiable.
      */
    def registerNonDifferentiable(opType: String): Unit = registry.update(opType, null)

    /** Gets the registered gradient function for the provided op type.
      *
      * @param  opType Op type whose gradient function is being looked up.
      * @return Gradient function registered for the provided op type.
      * @throws NoSuchElementException If no gradient has been registered for the provided op type.
      */
    @throws[NoSuchElementException]
    def apply(opType: String): GradientFunction = {
      if (!registry.contains(opType))
        throw new NoSuchElementException(s"No gradient registered for op type '$opType'.")
      registry(opType)
    }
  }

  /** Adds ops to the graph to compute the partial derivatives of the sum of `y`s with respect to the `x`s, using the
    * C++ gradients support of the TensorFlow native library.
    *
    * Note that the C++ gradients support of the TensorFlow native library is incomplete and will not be sufficient for
    * many use cases. It is mainly exposed as means of comparison to the Scala API functionality.
    *
    * The result of this function is an array containing: `d(y_1 + y_2 + ...)/dx_1`, `d(y_1 + y_2 + ...)/dx_2`, `...`.
    *
    * @param  y  Tensors whose partial derivatives are computed.
    * @param  x  Tensors with respect to which the gradients are computed.
    * @param  dx Tensors to use as the initial gradients. They represent the symbolic partial derivatives of some loss
    *            function `L` with respect to `y`. If `null`, then ones are used. The number of tensors in `dx` must
    *            match the number of tensors in `y`.
    * @return Partial derivatives of the `y`s given each one of the `x`s.
    */
  def ccGradients(y: Array[Output], x: Array[Output], dx: Array[Output] = null): Array[Output] = {
    // TODO: Overload this method with all possible uses for it.
    if (dx != null && dx.length != y.length)
      throw new IllegalArgumentException(s"The number of ys (${y.length}) must match the number of dxs (${dx.length}).")

    // Obtain the graph and verify that all provided op outputs are defined over the same graph
    val graph = y.head.graph
    y.foreach(o => Op.assertSameGraph(o.op, y.head.op))
    x.foreach(o => Op.assertSameGraph(o.op, y.head.op))
    if (dx != null)
      dx.foreach(o => Op.assertSameGraph(o.op, y.head.op))

    // Map all arrays to the corresponding data structures used by the JNI layer
    val yJNI = y.map(o => NativeOutput(o.op.nativeHandle, o.index))
    val xJNI = x.map(o => NativeOutput(o.op.nativeHandle, o.index))
    val dxJNI = if (dx == null) null else dx.map(o => NativeOutput(o.op.nativeHandle, o.index))

    // Add the gradients to the graph and collect them to the array that is returned
    val jniGradients = NativeGraph.addGradients(graph.nativeHandle, yJNI, xJNI, dxJNI)
    jniGradients.map(o => {
      val op = graph.opsCache.getOrElseUpdate(o.opHandle, Op(graph, o.opHandle))
      Output(op, o.outputIndex)
    })
  }
}
