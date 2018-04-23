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

package org.platanios.tensorflow.api.ops.training.distribute

import org.platanios.tensorflow.api.core.DeviceSpecification
import org.platanios.tensorflow.api.ops.{Math, Output}

/** Represents a reduction method.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Reduction {
  def reduce(
      values: Seq[Output],
      accumulateFn: Seq[Output] => Output = Math.addN(_, name = "ReductionAccumulate")
  ): Output

  def processUngroupedValue(value: Output, devices: Seq[DeviceSpecification]): Output = value
  def processRestoredTensor(value: Output, devices: Seq[DeviceSpecification]): Output = value
}

/** Reduces the variable updates by summing them. */
case object SumReduction extends Reduction {
  override def reduce(
      values: Seq[Output],
      accumulateFn: Seq[Output] => Output = Math.addN(_, name = "ReductionAccumulate")
  ): Output = {
    accumulateFn(values)
  }

  override def processRestoredTensor(value: Output, devices: Seq[DeviceSpecification]): Output = {
    // To preserve the sum across save and restore, we have to divide the total across all devices when restoring a
    // variable that was summed when saving.
    value / devices.length
  }
}

/** Reduces the variable updates by averaging them. */
case object MeanReduction extends Reduction {
  override def reduce(
      values: Seq[Output],
      accumulateFn: Seq[Output] => Output = Math.addN(_, name = "ReductionAccumulate")
  ): Output = {
    accumulateFn(values) / values.length
  }

  override def processUngroupedValue(value: Output, devices: Seq[DeviceSpecification]): Output = {
    value / devices.length
  }
}
