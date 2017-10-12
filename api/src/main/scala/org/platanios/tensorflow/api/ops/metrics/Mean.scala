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

package org.platanios.tensorflow.api.ops.metrics

import org.platanios.tensorflow.api.core.{Graph, Shape}
import org.platanios.tensorflow.api.ops.control_flow.ControlFlow
import org.platanios.tensorflow.api.ops.metrics.Metric._
import org.platanios.tensorflow.api.ops.variables.{Variable, ZerosInitializer}
import org.platanios.tensorflow.api.ops.{Basic, Math, Op, Output}
import org.platanios.tensorflow.api.types.{FLOAT32, FLOAT64}

/** Mean metric.
  *
  * The metric creates two local variables, `total` and `count` that are used to compute the average of some provided
  * values. This average is ultimately returned as an idempotent operation that simply divides `total` by `count`.
  *
  * For estimation of the metric over a stream of data, the function creates an `update` operation that updates these
  * variables and returns the mean. `update` increments `total` with the reduced sum of the product of `values` and
  * `weights`, and increments `count` with the reduced sum of `weights`.
  *
  * If `weights` is `null`, the weights default to 1. Use weights of `0` to mask values.
  *
  * @param  metricsCollections       Graph collections in which to add the metric value op.
  * @param  metricUpdatesCollections Graph collections in which to add the metric update op.
  * @param  name                     Name prefix for the created ops.
  *
  * @author Emmanouil Antonios Platanios
  */
class Mean private[metrics] (
    metricsCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
    metricUpdatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
    override val name: String = "Mean"
) extends Metric[Output, Output] {
  /** Computes the value of this metric for the provided values, optionally weighted by `weights`.
    *
    * @param  values  Values.
    * @param  weights Tensor containing weights for the values.
    * @param  name    Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  def compute(values: Output, weights: Output = null, name: String = name): Output = {
    var ops = Set(values.op)
    if (weights != null)
      ops += weights.op
    Op.createWithNameScope(name, ops) {
      val castedValues = if (values.dataType != FLOAT64) values.cast(FLOAT32) else values
      val (processedValues, numValues) = {
        if (weights == null) {
          (castedValues, Basic.size(castedValues).cast(castedValues.dataType))
        } else {
          var (matchedValues, _, matchedWeights) = matchAxes(castedValues, null, weights)
          matchedWeights = weightsBroadcast(matchedValues, matchedWeights.cast(castedValues.dataType))
          matchedValues = Math.multiply(matchedValues, matchedWeights)
          val numValues = Math.sum(matchedWeights)
          (matchedValues, numValues)
        }
      }
      val value = safeDiv(Math.sum(processedValues), numValues, name = "Value")
      metricsCollections.foreach(Op.currentGraph.addToCollection(value, _))
      value
    }
  }

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  values  Values.
    * @param  weights Tensor containing weights for the predictions.
    * @param  name    Name prefix for the created ops.
    * @return Tuple containing: (i) output representing the current value of the metric, (ii) op used to reset its
    *         value, and (iii) op used to update its current value and obtain the new value.
    */
  def streaming(values: Output, weights: Output = null, name: String = name): (Output, Op, Output) = {
    var ops = Set(values.op)
    if (weights != null)
      ops += weights.op
    Op.createWithNameScope(name, ops) {
      val castedValues = if (values.dataType != FLOAT64) values.cast(FLOAT32) else values
      val total = localVariable(s"$name/Total", castedValues.dataType, Shape.scalar(), ZerosInitializer)
      val count = localVariable(s"$name/Count", castedValues.dataType, Shape.scalar(), ZerosInitializer)
      val (processedValues, numValues) = {
        if (weights == null) {
          (castedValues, Basic.size(castedValues).cast(castedValues.dataType))
        } else {
          var (matchedValues, _, matchedWeights) = matchAxes(castedValues, null, weights)
          matchedWeights = weightsBroadcast(matchedValues, matchedWeights.cast(castedValues.dataType))
          matchedValues = Math.multiply(matchedValues, matchedWeights)
          val numValues = Math.sum(matchedWeights)
          (matchedValues, numValues)
        }
      }
      val updateTotal = total.assignAdd(Math.sum(processedValues))
      val updateCount = count.assignAdd(numValues)
      val value = safeDiv(total.value, count.value, name = "Value")
      val update = safeDiv(updateTotal, updateCount, name = "Update")
      metricsCollections.foreach(Op.currentGraph.addToCollection(value, _))
      metricUpdatesCollections.foreach(Op.currentGraph.addToCollection(update, _))
      val reset = ControlFlow.group(Set(total.initializer, count.initializer))
      (value, reset, update)
    }
  }
}

object Mean {
  /** Creates a new mean metric.
    *
    * @param  metricsCollections       Graph collections in which to add the metric value op.
    * @param  metricUpdatesCollections Graph collections in which to add the metric update op.
    * @param  name                     Name prefix for the created ops.
    * @return New mean metric.
    */
  def apply(
      metricsCollections: Set[Graph.Key[Variable]] = Set(METRIC_VARIABLES),
      metricUpdatesCollections: Set[Graph.Key[Output]] = Set(METRIC_UPDATES),
      name: String = "Mean"
  ): Mean = {
    new Mean(metricsCollections, metricUpdatesCollections, name)
  }
}
