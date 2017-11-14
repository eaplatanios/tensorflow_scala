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

package org.platanios.tensorflow.api.ops.io.data

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.ops.{Basic, Op, Output}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.types.{DataType, INT64}

/** Dataset that wraps the application of the `range` op.
  *
  * $OpDocDatasetRange
  *
  * @param  start Starting value of the number sequence.
  * @param  limit Ending value (exclusive) of the number sequence.
  * @param  delta Difference between consecutive numbers in the sequence.
  * @param  name  Name for this dataset.
  *
  * @author Emmanouil Antonios Platanios
  */
case class RangeDataset(
    start: Long,
    limit: Long,
    delta: Long = 1L,
    override val name: String = "RangeDataset"
) extends Dataset[Tensor, Output, DataType, Shape](name) {
  override def createHandle(): Output = {
    Op.Builder(opType = "RangeDataset", name = name)
        .addInput(Op.createWithNameScope(name)(Basic.constant(start)))
        .addInput(Op.createWithNameScope(name)(Basic.constant(limit)))
        .addInput(Op.createWithNameScope(name)(Basic.constant(delta)))
        .setAttribute("output_types", flattenedOutputDataTypes.toArray)
        .setAttribute("output_shapes", flattenedOutputShapes.toArray)
        .build().outputs(0)
  }

  override def outputDataTypes: DataType = INT64
  override def outputShapes: Shape = Shape.scalar()
}

object RangeDataset {
  /** @define OpDocDatasetRange
    *   The dataset `range` op creates a new dataset that contains a range of values.
    */
  private[data] trait Documentation
}
