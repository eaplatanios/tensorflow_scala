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

import org.platanios.tensorflow.api.ops.Output

/** Map metric wrapper.
  *
  * The map metric wraps around an existing metric, but also applies the provided `mapFn` to its inputs, before passing
  * them on to `metric`.
  *
  * @param  mapFn  Mapping function to use for the metric inputs.
  * @param  metric Metric being wrapped that takes the output of `mapFn` as its input type.
  *
  * @author Emmanouil Antonios Platanios
  */
class MapMetric[S, T, R](
    val mapFn: S => T,
    val metric: Metric[T, R]
) extends Metric[S, R] {
  /** Name of this metric. */
  override val name: String = metric.name

  /** Computes the value of this metric for the provided values, optionally weighted by `weights`.
    *
    * @param  values  Values.
    * @param  weights Tensor containing weights for the values.
    * @param  name    Name prefix for the created ops.
    * @return Created output containing the metric value.
    */
  override def compute(values: S, weights: Output, name: String): R = {
    metric.compute(mapFn(values), weights, name)
  }

  /** Creates ops for computing the value of this metric in a streaming fashion. This function returns an op for
    * obtaining the value of this metric, as well as a pair of ops to update its accumulated value and reset it.
    *
    * @param  values  Values.
    * @param  weights Tensor containing weights for the predictions.
    * @param  name    Name prefix for the created ops.
    * @return Tuple containing: (i) an output representing the current value of the metric, (ii) an op used to update
    *         its current value and obtain the new value, and (iii) an op used to reset its value.
    */
  override def streaming(values: S, weights: Output, name: String): Metric.StreamingInstance[R] = {
    metric.streaming(mapFn(values), weights, name)
  }
}

object MapMetric {
  /** Creates a new map metric.
    *
    * @param  mapFn  Mapping function to use for the metric inputs.
    * @param  metric Metric being wrapped that takes the output of `mapFn` as its input type.
    * @return New map metric.
    */
  def apply[S, T, R](mapFn: S => T, metric: Metric[T, R]): MapMetric[S, T, R] = {
    new MapMetric(mapFn, metric)
  }
}
