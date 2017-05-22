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

package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.core

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait CoreAPI
    extends client.ClientAPI {
  type Graph = core.Graph
  val Graph = core.Graph

  val defaultGraph: core.Graph = api.defaultGraph

  type Indexer = core.Indexer
  type Index = core.Index
  type Slice = core.Slice

  val Indexer  = core.Indexer
  val Index    = core.Index
  val Slice    = core.Slice
  val NewAxis  = core.NewAxis
  val Ellipsis = core.Ellipsis

  type Shape = core.Shape
  val Shape = core.Shape

  type DeviceSpecification = core.DeviceSpecification
  val DeviceSpecification = core.DeviceSpecification

  type ShapeMismatchException = core.exception.ShapeMismatchException
  type GraphMismatchException = core.exception.GraphMismatchException
  type IllegalNameException = core.exception.IllegalNameException
  type InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
  type InvalidGraphElementException = core.exception.InvalidGraphElementException
  type InvalidShapeException = core.exception.InvalidShapeException
  type InvalidIndexerException = core.exception.InvalidIndexerException
  type InvalidDataTypeException = core.exception.InvalidDataTypeException
  type OpBuilderUsedException = core.exception.OpBuilderUsedException

  val ShapeMismatchException              = core.exception.ShapeMismatchException
  val GraphMismatchException              = core.exception.GraphMismatchException
  val IllegalNameException                = core.exception.IllegalNameException
  val InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
  val InvalidGraphElementException        = core.exception.InvalidGraphElementException
  val InvalidShapeException               = core.exception.InvalidShapeException
  val InvalidIndexerException             = core.exception.InvalidIndexerException
  val InvalidDataTypeException            = core.exception.InvalidDataTypeException
  val OpBuilderUsedException              = core.exception.OpBuilderUsedException
}
