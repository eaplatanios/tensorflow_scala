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

package org.platanios.tensorflow.api.ops.training

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, Slot, UntypedOp}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.variables._

import scala.collection.mutable

// TODO: Add support for restoring averaged values for the actual variables.

/** Maintains moving averages of variables by employing an exponential decay.
  *
  * When training a model, it is often beneficial to maintain moving averages of the trained parameters. Evaluations
  * that use averaged parameters sometimes produce significantly better results than the final trained values.
  *
  * The `computeForVariables(...)` and `computeForValues(...)` methods add shadow copies of the provided variables and
  * values, along with ops that maintain their moving averages, in their shadow copies. They are used when building the
  * training model. The ops that maintain moving averages are typically run after each training step. The `average(...)`
  * and `averageName(...)` methods provide access to the shadow variables and their names. They are useful when building
  * an evaluation model, or when restoring a model from a checkpoint file. They help use the moving averages in place of
  * the last trained values for evaluations.
  *
  * The moving averages are computed using exponential decay. The decay value must be provided when creating an
  * `ExponentialMovingAverage` object. The shadow variables are initialized with the same initial values as the
  * corresponding variables, or with zeros for the case of values. When the ops used to maintain the moving averages are
  * executed, each shadow variable is updated using the formula:
  * {{{
  *   shadowVariable -= (1 - decay) * (shadowVariable - value)
  * }}}
  * This is mathematically equivalent to the classic formula below, but the use of an `assignSub` op (the `-=` in the
  * formula) allows concurrent lock-free updates to the variables:
  * {{{
  *   shadowVariable = decay * shadow_variable + (1 - decay) * value
  * }}}
  * Reasonable values for `decay` are close to `1.0f`, typically in the "multiple-nines" range: `0.999f`, etc.
  *
  * Example usage when creating a training model:
  * {{{
  *   // Create variables
  *   val v0 = tf.variable(...)
  *   val v1 = tf.variable(...)
  *   // Use the variables to build a training model
  *   ...
  *   // Create an op that applies the optimizer. This is what we usually would use as a training op.
  *   val optOp = opt.minimize(loss, variables = Set(v0, v1))
  *
  *   // Create an exponential moving average object.
  *   val ema = tf.train.ExponentialMovingAverage(decay = 0.999f)
  *
  *   val trainOp = tf.createWith(controlDependencies = Set(optOp)) {
  *     // Create the shadow variables, and add ops used to maintain the moving averages of `v0` and `v1`. This also
  *     // creates an op that will update the moving averages after each training step. This is what we will use in
  *     // place of the usual training op.
  *     ema.computeForVariables(Set(v0, v1))
  *   }
  *
  *   // Train the model by running `trainOp`.
  * }}}
  *
  * There are two ways to use moving averages for evaluations:
  *
  *   - Build a model that uses the shadow variables instead of the variables. For this, use the `average(...)` method
  *     which returns the shadow variable for a given variable.
  *   - Build a model normally but load the checkpoint files to evaluate by using the shadow variable names. For this
  *     use the `averageName(...)` method. Please refer to the `Saver` class documentation for more information on how
  *     to restore saved variables.
  *
  * Example of restoring the shadow variable values:
  * {{{
  *   // Create a saver that loads variables from their saved shadow values.
  *   val shadowV0Name = ema.averageName(v0)
  *   val shadowV1Name = ema.averageName(v1)
  *   val saver = tf.saver(Map(shadowV0Name -> v0, shadowV1Name -> v1))
  *   saver.restore(...checkpoint filename...)
  *   // `v0` and `v1` now hold the moving average values.
  * }}}
  *
  * The optional `numUpdates` parameter allows one to tweak the decay rate dynamically. It is typical to pass the count
  * of training steps, usually kept in a variable that is incremented at each step, in which case the decay rate is
  * lower at the start of training. This makes moving averages move faster. If passed, the actual decay rate used is
  * defined as: `min(decay, (1 + numUpdates) / (10 + numUpdates))`.
  *
  * @param  decay      Decay value to use.
  * @param  numUpdates Optional count of number of updates applied to the variables.
  * @param  zeroDebias If `true`, the moving averages computed for values provided in `computeForValues` will be
  *                    zero-debiased.
  * @param  name       Name prefix to use for all created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class ExponentialMovingAverage protected (
    val decay: Float,
    val numUpdates: Option[Int] = None,
    val zeroDebias: Boolean = false,
    val name: String = "ExponentialMovingAverage"
) {
  protected val decayTensor: Output[Float] = {
    Op.nameScope(name) {
      var decayTensor = Basic.constant(decay, name = "Decay")
      numUpdates.foreach(n => {
        val numUpdatesTensor = Basic.constant(n.toFloat, name = "NumUpdates")
        decayTensor = Math.minimum(decayTensor, (1.0f + numUpdatesTensor) / (10.0f + numUpdatesTensor))
      })
      decayTensor
    }
  }

  protected val variableAverages: mutable.Map[Variable[Any], Variable[Any]] = mutable.HashMap.empty
  protected val valueAverages   : mutable.Map[Output[Any], Variable[Any]]   = mutable.HashMap.empty

  /** Computes moving averages of the provided variables.
    *
    * This method creates shadow variables for all elements of `variables`. The shadow variables for each variable are
    * created with `trainable = false`, initialized to the variable's initial value, and added to the
    * `Graph.Keys.MOVING_AVERAGE_VARIABLES` and the `Graph.Keys.GLOBAL_VARIABLES` collections.
    *
    * @param  variables Variables for which to compute moving averages.
    * @return Created op that updates all the shadow variables, as described above.
    */
  def computeForVariables(variables: Set[Variable[Any]] = Op.currentGraph.trainableVariables): UntypedOp = {
    variables.foreach(v => {
      if (!Set[DataType[_]](FLOAT16, FLOAT32, FLOAT64).contains(v.dataType))
        throw InvalidArgumentException(
          s"Moving averages can only be computed for `FLOAT16`, `FLOAT32`, and `FLOAT64` variables " +
              s"(i.e., not for `${v.dataType}` variables).")
      if (variableAverages.contains(v))
        throw InvalidArgumentException(s"The moving average for variable '${v.name}' is already being computed.")
      // In order to lower communication bandwidth across devices we keep the moving averages on the same device as the
      // original variables. For other tensors, we rely on the existing device allocation mechanism.
      Op.initializationScope {
        val evTF = TF.fromDataType(v.dataType)
        val average = Slot.create(
          v, v.dataType, DynamicConstantInitializer(v.initializedValue)(evTF),
          name, colocateWithPrimary = true)(evTF, evTF)
        Op.currentGraph.addToCollection(Graph.Keys.MOVING_AVERAGE_VARIABLES)(average)
        variableAverages.update(v, average)
      }
    })

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNotQuantized[Any] = new IsNotQuantized[Any] {}

    Op.nameScope(name) {
      val updates = variables.map(v => {
        ExponentialMovingAverage.assignMovingAverage(
          variableAverages(v), v.value, decayTensor, zeroDebias = false
        )(TF.fromDataType(v.dataType), ev).op
      })
      ControlFlow.group(updates)
    }
  }

  /** Computes moving averages of the provided values.
    *
    * This method creates shadow variables for all elements of `values`. The shadow variables for each value are
    * created with `trainable = false`, initialized to `0` and optionally zero-debiased, and added to the
    * `Graph.Keys.MOVING_AVERAGE_VARIABLES` and the `Graph.Keys.GLOBAL_VARIABLES` collections.
    *
    * @param  values Values for which to compute moving averages.
    * @return Created op that updates all the shadow variables, as described above.
    */
  def computeForValues(values: Set[Output[Any]]): UntypedOp = {
    val zeroDebiasVariables = mutable.Set.empty[Variable[Any]]
    values.foreach(v => {
      if (!Set[DataType[_]](FLOAT16, FLOAT32, FLOAT64).contains(v.dataType))
        throw InvalidArgumentException(
          s"Moving averages can only be computed for `FLOAT16`, `FLOAT32`, and `FLOAT64` values " +
              s"(i.e., not for `${v.dataType}` values).")
      if (valueAverages.contains(v))
        throw InvalidArgumentException(s"The moving average for value '${v.name}' is already being computed.")
      // In order to lower communication bandwidth across devices we keep the moving averages on the same device as the
      // original variables. For other tensors, we rely on the existing device allocation mechanism.
      Op.initializationScope {
        val colocateWithPrimary = Set("Variable", "VariableV2", "VarHandleOp").contains(v.op.opType)
        val evTF = TF.fromDataType(v.dataType)
        val average = Slot.zerosForOutput(
          v, v.dataType, name, colocateWithPrimary = colocateWithPrimary)(evTF, evTF)
        Op.currentGraph.addToCollection(Graph.Keys.MOVING_AVERAGE_VARIABLES)(average)
        if (zeroDebias)
          zeroDebiasVariables += average
        valueAverages.update(v, average)
      }
    })

    // TODO: [TYPES] !!! Super hacky. Remove in the future.
    implicit val ev: IsNotQuantized[Any] = new IsNotQuantized[Any] {}

    Op.nameScope(name) {
      val updates = values.map(v => {
        val average = valueAverages(v)
        val zeroDebias = zeroDebiasVariables.contains(average)
        ExponentialMovingAverage.assignMovingAverage(
          average, v, decayTensor, zeroDebias = zeroDebias
        )(TF.fromDataType(v.dataType), ev).op
      })
      ControlFlow.group(updates)
    }
  }

  /** Returns the variable holding the average for `variable`. */
  def average[T: TF](variable: Variable[T]): Option[Variable[T]] = {
    variableAverages.get(variable).map(_.asInstanceOf[Variable[T]])
  }

  /** Returns the variable holding the average for `value`. */
  def average[T: TF](value: Output[T]): Option[Variable[T]] = {
    valueAverages.get(value).map(_.asInstanceOf[Variable[T]])
  }

  // TODO: Support calling `averageName` before `computeForX` has been called.

  /** Returns the name of the variable holding the average for `variable`. */
  def averageName(variable: Variable[_]): Option[String] = {
    variableAverages.get(variable).map(_.op.name)
  }

  /** Returns the name of the variable holding the average for `value`. */
  def averageName(value: Output[_]): Option[String] = {
    valueAverages.get(value).map(_.op.name)
  }
}

object ExponentialMovingAverage {
  def apply(
      decay: Float,
      numUpdates: Option[Int] = None,
      zeroDebias: Boolean = false,
      name: String = "ExponentialMovingAverage"
  ): ExponentialMovingAverage = {
    new ExponentialMovingAverage(decay, numUpdates, zeroDebias, name)
  }

  /** Computes the moving average of a variable.
    *
    * The moving average of `variable` updated with `value` is defined as: `variable * decay + value * (1 - decay)`.
    * The returned op sets `variable` to the newly computed moving average. The new value of `variable` can be set with
    * the `assignSub` op as: `variable -= (1 - decay) * (variable - value)`. Since variables that are initialized to a
    * `0` value will be `0` biased, `zeroDebias` optionally enables scaling by the mathematically correct debiasing
    * factor of `1 - decay ** num_updates`. Please refer to
    * [ADAM: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) section 3 for more details.
    *
    * The names of the debiasing shadow variables include, by default, both the scope they were created in and the
    * scope of the variables they debias. They are also given a uniqifying-suffix. For example:
    * {{{
    *   tf.createWithVariableScope("scope1") {
    *     tf.createWithVariableScope("scope2") {
    *       val variable = tf.variable("foo", FLOAT32, Shape())
    *       assignMovingAverage(variable, 0.0f, 1.0f)
    *       assignMovingAverage(variable, 0.0f, 0.9f)
    *     }
    *   }
    * }}}
    * The variable name in this case is `"scope1/scope2/foo"`, whereas the shadow variable names are
    * `"scope1/scope2/scope1/scope2/foo/Biased"` and `"scope1/scope2/scope1/scope2/foo/Biased_1"`.
    *
    * @param  variable   Variable whose moving average is to be computed.
    * @param  value      Tensor with the same shape as `variable`, that is used to update the moving average.
    * @param  decay      `FLAOT32` tensor representing the moving average decay.
    * @param  zeroDebias If `true`, it will be assumed that the variable was `0`-initialized and the created op will
    *                    debias it.
    * @param  name       Name for the created ops.
    * @return Value of `variable` after the moving average update.
    */
  private[ExponentialMovingAverage] def assignMovingAverage[T: TF : IsNotQuantized](
      variable: Variable[T],
      value: Output[T],
      decay: Output[Float],
      zeroDebias: Boolean,
      name: String = "Assign"
  ): Output[T] = {
    Op.createWith(nameScope = name) {
      Op.colocateWith(Set(variable.op), ignoreExisting = true) {
        val one = Basic.ones(decay.dataType, Shape())
        val processedDecay = (one - decay).castTo(variable.dataType)
        val updateDelta = {
          if (zeroDebias)
            ExponentialMovingAverage.zeroDebias(variable, value, processedDecay)
          else
            (variable.value - value) * processedDecay
        }
        variable.assignSub(updateDelta)
      }
    }
  }

  /** Computes the difference required to de-bias an exponential moving average (EMA) variable.
    *
    * All exponential moving averages (EMAs) initialized with tensors are initialized to `0`, and therefore are biased
    * towards `0`. Variables initialized to `0` and used as EMAs are similarly biased. This function computes the
    * de-biased updated amount according to a scale factor, as shown in
    * [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980).
    *
    * To demonstrate the bias that results from `0`-initialization, take an EMA that was initialized to `0`, with decay
    * `b`. After `t` steps of seeing the constant `c`, the variable will have the following value:
    * {{{
    *   EMA = 0 * b^(t) + c * (1 - b) * b^(t - 1) + c * (1 - b) * b^(t - 2) + ... = c * (1 - b^t)
    * }}}
    * To have the true value `c`, we should divide by the scale factor `1 - b^t`. In order to perform de-biasing, we
    * use two shadow variables. One keeps track of the biased estimate, and the other keeps track of the number of
    * updates that have occurred.
    *
    * @param  unbiasedVariable Variable representing the current value of the de-biased EMA.
    * @param  value            Most recent value.
    * @param  decay            EMA one-step decay value.
    * @return Tensor containing the amount that should be added to the unbiased variable. Computing this tensor will
    *         also update the shadow variables appropriately.
    */
  private[ExponentialMovingAverage] def zeroDebias[T: TF : IsNotQuantized](
      unbiasedVariable: Variable[T],
      value: Output[T],
      decay: Output[T]
  ): Output[T] = {
    VariableScope.scope(unbiasedVariable.name) {
      Op.colocateWith(Set(unbiasedVariable.op), ignoreExisting = true) {
        val biased = Variable.getVariable[T](
          "Biased", unbiasedVariable.shape, ZerosInitializer, trainable = false)
        val localStep = Variable.getVariable[T](
          "LocalStep", Shape(), ZerosInitializer, trainable = false)
        val biasedUpdate = biased.assignSub((biased - value) * decay, VariableScope.current.name)
        val localStepUpdate = localStep.assignAdd(Basic.ones[T](Shape()))
        // Compute the value of the delta to update the unbiased EMA. Make sure to use the new values of the biased
        // variable and the local step.
        Op.createWith(controlDependencies = Set(biasedUpdate.op, localStepUpdate.op)) {
          // This function gets `1 - decay`, and so we use `1.0 - decay` in the exponent.
          val one = Basic.ones[T](Shape())
          unbiasedVariable - (biased.read() / (one - Math.pow(one - decay, localStep.read())))
        }
      }
    }
  }
}
