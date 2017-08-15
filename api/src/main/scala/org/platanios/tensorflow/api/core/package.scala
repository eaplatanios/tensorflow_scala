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

package org.platanios.tensorflow.api

import org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
package object core {
  private[api] trait API extends Indexer.API

  private[api] object API extends API

  private[api] trait ScopedAPI extends client.API {
    type Graph = core.Graph
    type Shape = core.Shape

    val Graph: core.Graph.type = core.Graph
    val Shape: core.Shape.type = core.Shape

    val defaultGraph: core.Graph = api.defaultGraph

    type DeviceSpecification = core.DeviceSpecification
    type ShapeMismatchException = core.exception.ShapeMismatchException
    type GraphMismatchException = core.exception.GraphMismatchException
    type IllegalNameException = core.exception.IllegalNameException
    type InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
    type InvalidGraphElementException = core.exception.InvalidGraphElementException
    type InvalidShapeException = core.exception.InvalidShapeException
    type InvalidIndexerException = core.exception.InvalidIndexerException
    type InvalidDataTypeException = core.exception.InvalidDataTypeException
    type OpBuilderUsedException = core.exception.OpBuilderUsedException

    val ShapeMismatchException             : exception.ShapeMismatchException.type              = core.exception.ShapeMismatchException
    val GraphMismatchException             : exception.GraphMismatchException.type              = core.exception.GraphMismatchException
    val IllegalNameException               : exception.IllegalNameException.type                = core.exception.IllegalNameException
    val InvalidDeviceSpecificationException: exception.InvalidDeviceSpecificationException.type = core.exception.InvalidDeviceSpecificationException
    val InvalidGraphElementException       : exception.InvalidGraphElementException.type        = core.exception.InvalidGraphElementException
    val InvalidShapeException              : exception.InvalidShapeException.type               = core.exception.InvalidShapeException
    val InvalidIndexerException            : exception.InvalidIndexerException.type             = core.exception.InvalidIndexerException
    val InvalidDataTypeException           : exception.InvalidDataTypeException.type            = core.exception.InvalidDataTypeException
    val OpBuilderUsedException             : exception.OpBuilderUsedException.type              = core.exception.OpBuilderUsedException
  }

  object exception {
    case class ShapeMismatchException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class GraphMismatchException(message: String = null, cause: Throwable = null)
        extends IllegalStateException(message, cause)

    case class IllegalNameException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class InvalidDeviceSpecificationException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class InvalidGraphElementException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class InvalidShapeException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class InvalidIndexerException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class InvalidDataTypeException(message: String = null, cause: Throwable = null)
        extends IllegalArgumentException(message, cause)

    case class OpBuilderUsedException(message: String = null, cause: Throwable = null)
        extends IllegalStateException(message, cause)
  }

  private[api] object ScopedAPI extends ScopedAPI
}
