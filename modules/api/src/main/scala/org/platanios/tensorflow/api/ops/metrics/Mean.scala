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

package org.platanios.tensorflow.api.ops.metrics

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.metrics.Metric._
import org.platanios.tensorflow.api.ops.variables.{Variable, VariableScope, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output, UntypedOp}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{FLOAT32, INT64}

/** Mean metric.
  *
  * The metric creates two local variables, `total` and `count` that are used to compute the average of some provided
  * values. This average is ultimately returned as an idempotent operation that simply divides `total` by `count`.
  *
  * For estimation of the metric over a stream of data, the function creates an `update` operation that updates these
  * variables and returns the mean. `update` increments `total` with the reduced sum of the product of `values` and
  * `weights`, and increments `count` with the reduced sum of `weights`.
  *
  * If `weights` is `None`, the weights default to 1. Use weights of `0` to mask values.
  *
  * @param  nameScope            Name prefix for the created ops.
  * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
  * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
  * @param  valuesCollections    Graph collections in which to add the metric values.
  * @param  updatesCollections   Graph collections in which to add the metric updates.
  * @param  resetsCollections    Graph collections in which to add the metric resets.
  *
  * @author Emmanouil Antonios Platanios
  */
class Mean(
    val nameScope: String,
    protected val defaultWeights: Option[Tensor[Float]] = None,
    val variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
    val valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
    val updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
    val resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS)
) extends Metric[Output[Float], Output[Float]] {
  /** Name of this metric. */
  override def name: String = nameScope

  /** Weights to multiply the provided values with when computing the value of this metric. */
  override def weights: Option[Tensor[Float]] = defaultWeights

  /** Computes the value of this metric for the provided values, optionally weighted by `weights`.
    *
    * @param  values  Values.
    * @param  weights Optional tensor containing weights for the values.
    * @param  name    Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  override def compute(
      values: Output[Float],
      weights: Option[Output[Float]] = None,
      name: String = s"$name/Compute"
  ): Output[Float] = {
    var ops = Set(values.op)
    val computedWeights = getWeights(weights)
    computedWeights.foreach(ops += _.op)
    Op.nameScope(name) {
      val (processedValues, numValues) = computedWeights match {
        case None =>
          (values, Basic.size(values, INT64).toFloat32)
        case Some(_) =>
          var (matchedValues, _, Some(matchedWeights)) = matchAxes(values, None, computedWeights)
          matchedWeights = weightsBroadcast(matchedValues, matchedWeights.toFloat32)
          matchedValues = Math.multiply(matchedValues, matchedWeights)
          val numValues = Math.sum(matchedWeights)
          (matchedValues, numValues)
      }
      val value = safeDiv(Math.sum(processedValues), numValues, name = "Value")
      valuesCollections.foreach(Op.currentGraph.addToCollection(value, _))
      value
    }
  }

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  values  Values.
    * @param  weights Optional tensor containing weights for the predictions.
    * @param  name    Name prefix for the created ops.
    * @return Tuple containing: (i) an output representing the current value of the metric, (ii) an op used to update
    *         its current value and obtain the new value, and (iii) an op used to reset its value.
    */
  override def streaming(
      values: Output[Float],
      weights: Option[Output[Float]] = None,
      name: String = s"$name/Streaming"
  ): Metric.StreamingInstance[Output[Float]] = {
    val computedWeights = getWeights(weights)
    VariableScope.scope(name) {
      Op.nameScope(name) {
        val total = Metric.variable("Total", FLOAT32, Shape.scalar(), ZerosInitializer, variablesCollections)
        val count = Metric.variable("Count", FLOAT32, Shape.scalar(), ZerosInitializer, variablesCollections)
        val (processedValues, numValues) = computedWeights match {
          case None =>
            (values, Basic.size(values, INT64).toFloat32)
          case Some(_) =>
            var (matchedValues, _, Some(matchedWeights)) = matchAxes(values, None, computedWeights)
            matchedWeights = weightsBroadcast(matchedValues, matchedWeights.toFloat32)
            matchedValues = Math.multiply(matchedValues, matchedWeights)
            val numValues = Math.sum(matchedWeights)
            (matchedValues, numValues)
        }
        val updateTotal = total.assignAdd(Math.sum(processedValues))
        val updateCount = count.assignAdd(numValues)
        val value = safeDiv(total.value, count.value, name = "Value")
        val update = safeDiv(updateTotal, updateCount, name = "Update")
        val reset = ControlFlow.group(Set(total.initializer, count.initializer), name = "Reset").asUntyped
        valuesCollections.foreach(Op.currentGraph.addToCollection(value, _))
        updatesCollections.foreach(Op.currentGraph.addToCollection(update, _))
        resetsCollections.foreach(Op.currentGraph.addToCollection(reset, _))
        Metric.StreamingInstance(value, update, reset, Set(total, count))
      }
    }
  }
}

object Mean {
  /** Creates a new mean metric.
    *
    * @param  nameScope            Name prefix for the created ops.
    * @param  defaultWeights       Default weights with which all computed metric values are multiplied.
    * @param  variablesCollections Graph collections in which to add the metric variables (for streaming metrics).
    * @param  valuesCollections    Graph collections in which to add the metric values.
    * @param  updatesCollections   Graph collections in which to add the metric updates.
    * @param  resetsCollections    Graph collections in which to add the metric resets.
    * @return New mean metric.
    */
  def apply(
      nameScope: String,
      defaultWeights: Option[Tensor[Float]] = None,
      variablesCollections: Set[Graph.Key[Variable[Any]]] = Set(METRIC_VARIABLES),
      valuesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_VALUES),
      updatesCollections: Set[Graph.Key[Output[Any]]] = Set(METRIC_UPDATES),
      resetsCollections: Set[Graph.Key[UntypedOp]] = Set(METRIC_RESETS)
  ): Mean = {
    new Mean(
      nameScope, defaultWeights, variablesCollections,
      valuesCollections, updatesCollections, resetsCollections)
  }
}
