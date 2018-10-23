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

package org.platanios.tensorflow.api.implicits

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.types.{DataType, Variant}
import org.platanios.tensorflow.api.ops.Output
import org.platanios.tensorflow.api.ops.data.Dataset

package object helpers {
  type SparseDataType[T] = (DataType[Long], DataType[T], DataType[Long])
  type SparseShape = (Shape, Shape, Shape)

  // TODO: [FUNCTIONS] !!! Find a better way to deal with this for use in the reduce function of the "GroupByWindowDataset".

  case class VariantDataset[T: OutputStructure, DD, SS] protected(
      handle: Output[Variant],
      private val _outputDataTypes: Any = null,
      private val _outputShapes: Any = null
  )(implicit
      _evOutputToDataType: OutputToDataType.Aux[T, DD],
      _evOutputToShape: OutputToShape.Aux[T, SS]
  ) extends Dataset[T] {
    override type D = DD
    override type S = SS

    override implicit def evOutputToDataType: OutputToDataType.Aux[T, DD] = _evOutputToDataType
    override implicit def evOutputToShape: OutputToShape.Aux[T, SS] = _evOutputToShape

    override val name: String = "VariantDataset"

    override def createHandle(): Output[Variant] = {
      handle
    }

    override def outputDataTypes: D = {
      _outputDataTypes.asInstanceOf[D]
    }

    override def outputShapes: S = {
      _outputShapes.asInstanceOf[S]
    }
  }
}
