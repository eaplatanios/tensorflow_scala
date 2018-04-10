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

/**
  * @author Emmanouil Antonios Platanios
  */
package object metrics {
  private[api] trait API {
    type Metric[T, R] = metrics.Metric[T, R]
    type MapMetric[S, T, R] = metrics.MapMetric[S, T, R]
    type Mean = metrics.Mean
    type Accuracy = metrics.Accuracy
    type ConfusionMatrix = metrics.ConfusionMatrix

    val Metric         : metrics.Metric.type          = metrics.Metric
    val MapMetric      : metrics.MapMetric.type       = metrics.MapMetric
    val Mean           : metrics.Mean.type            = metrics.Mean
    val Accuracy       : metrics.Accuracy.type        = metrics.Accuracy
    val ConfusionMatrix: metrics.ConfusionMatrix.type = metrics.ConfusionMatrix
  }
}
