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

import org.platanios.tensorflow.jni

/**
  * @author Emmanouil Antonios Platanios
  */
package object core {
  private[core] val META_GRAPH_UNBOUND_INPUT_PREFIX: String = "$unbound_inputs_"

  private[api] val defaultGraph: core.Graph = core.Graph()

  private[api] trait API {
    type DeviceSpecification = core.DeviceSpecification

    type CancelledException = exception.CancelledException
    type UnknownException = exception.UnknownException
    type InvalidArgumentException = exception.InvalidArgumentException
    type DeadlineExceededException = exception.DeadlineExceededException
    type NotFoundException = exception.NotFoundException
    type AlreadyExistsException = exception.AlreadyExistsException
    type PermissionDeniedException = exception.PermissionDeniedException
    type UnauthenticatedException = exception.UnauthenticatedException
    type ResourceExhaustedException = exception.ResourceExhaustedException
    type FailedPreconditionException = exception.FailedPreconditionException
    type AbortedException = exception.AbortedException
    type OutOfRangeException = exception.OutOfRangeException
    type UnimplementedException = exception.UnimplementedException
    type InternalException = exception.InternalException
    type UnavailableException = exception.UnavailableException
    type DataLossException = exception.DataLossException

    val CancelledException         : exception.CancelledException.type          = exception.CancelledException
    val UnknownException           : exception.UnknownException.type            = exception.UnknownException
    val InvalidArgumentException   : exception.InvalidArgumentException.type    = exception.InvalidArgumentException
    val DeadlineExceededException  : exception.DeadlineExceededException.type   = exception.DeadlineExceededException
    val NotFoundException          : exception.NotFoundException.type           = exception.NotFoundException
    val AlreadyExistsException     : exception.AlreadyExistsException.type      = exception.AlreadyExistsException
    val PermissionDeniedException  : exception.PermissionDeniedException.type   = exception.PermissionDeniedException
    val UnauthenticatedException   : exception.UnauthenticatedException.type    = exception.UnauthenticatedException
    val ResourceExhaustedException : exception.ResourceExhaustedException.type  = exception.ResourceExhaustedException
    val FailedPreconditionException: exception.FailedPreconditionException.type = exception.FailedPreconditionException
    val AbortedException           : exception.AbortedException.type            = exception.AbortedException
    val OutOfRangeException        : exception.OutOfRangeException.type         = exception.OutOfRangeException
    val UnimplementedException     : exception.UnimplementedException.type      = exception.UnimplementedException
    val InternalException          : exception.InternalException.type           = exception.InternalException
    val UnavailableException       : exception.UnavailableException.type        = exception.UnavailableException
    val DataLossException          : exception.DataLossException.type           = exception.DataLossException

    type ShapeMismatchException = exception.ShapeMismatchException
    type GraphMismatchException = exception.GraphMismatchException
    type IllegalNameException = exception.IllegalNameException
    type InvalidDeviceException = exception.InvalidDeviceException
    type InvalidShapeException = exception.InvalidShapeException
    type InvalidIndexerException = exception.InvalidIndexerException
    type InvalidDataTypeException = exception.InvalidDataTypeException
    type OpBuilderUsedException = exception.OpBuilderUsedException
    type CheckpointNotFoundException = exception.CheckpointNotFoundException

    val ShapeMismatchException     : exception.ShapeMismatchException.type      = exception.ShapeMismatchException
    val GraphMismatchException     : exception.GraphMismatchException.type      = exception.GraphMismatchException
    val IllegalNameException       : exception.IllegalNameException.type        = exception.IllegalNameException
    val InvalidDeviceException     : exception.InvalidDeviceException.type      = exception.InvalidDeviceException
    val InvalidShapeException      : exception.InvalidShapeException.type       = exception.InvalidShapeException
    val InvalidIndexerException    : exception.InvalidIndexerException.type     = exception.InvalidIndexerException
    val InvalidDataTypeException   : exception.InvalidDataTypeException.type    = exception.InvalidDataTypeException
    val OpBuilderUsedException     : exception.OpBuilderUsedException.type      = exception.OpBuilderUsedException
    val CheckpointNotFoundException: exception.CheckpointNotFoundException.type = exception.CheckpointNotFoundException
  }

  object exception {
    type CancelledException = jni.CancelledException
    type UnknownException = jni.UnknownException
    type InvalidArgumentException = jni.InvalidArgumentException
    type DeadlineExceededException = jni.DeadlineExceededException
    type NotFoundException = jni.NotFoundException
    type AlreadyExistsException = jni.AlreadyExistsException
    type PermissionDeniedException = jni.PermissionDeniedException
    type UnauthenticatedException = jni.UnauthenticatedException
    type ResourceExhaustedException = jni.ResourceExhaustedException
    type FailedPreconditionException = jni.FailedPreconditionException
    type AbortedException = jni.AbortedException
    type OutOfRangeException = jni.OutOfRangeException
    type UnimplementedException = jni.UnimplementedException
    type InternalException = jni.InternalException
    type UnavailableException = jni.UnavailableException
    type DataLossException = jni.DataLossException

    val CancelledException         : jni.CancelledException.type          = jni.CancelledException
    val UnknownException           : jni.UnknownException.type            = jni.UnknownException
    val InvalidArgumentException   : jni.InvalidArgumentException.type    = jni.InvalidArgumentException
    val DeadlineExceededException  : jni.DeadlineExceededException.type   = jni.DeadlineExceededException
    val NotFoundException          : jni.NotFoundException.type           = jni.NotFoundException
    val AlreadyExistsException     : jni.AlreadyExistsException.type      = jni.AlreadyExistsException
    val PermissionDeniedException  : jni.PermissionDeniedException.type   = jni.PermissionDeniedException
    val UnauthenticatedException   : jni.UnauthenticatedException.type    = jni.UnauthenticatedException
    val ResourceExhaustedException : jni.ResourceExhaustedException.type  = jni.ResourceExhaustedException
    val FailedPreconditionException: jni.FailedPreconditionException.type = jni.FailedPreconditionException
    val AbortedException           : jni.AbortedException.type            = jni.AbortedException
    val OutOfRangeException        : jni.OutOfRangeException.type         = jni.OutOfRangeException
    val UnimplementedException     : jni.UnimplementedException.type      = jni.UnimplementedException
    val InternalException          : jni.InternalException.type           = jni.InternalException
    val UnavailableException       : jni.UnavailableException.type        = jni.UnavailableException
    val DataLossException          : jni.DataLossException.type           = jni.DataLossException

    case class ShapeMismatchException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class GraphMismatchException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class IllegalNameException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class InvalidDeviceException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class InvalidShapeException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class InvalidIndexerException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class InvalidDataTypeException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class OpBuilderUsedException(message: String = null, cause: Throwable = null)
        extends InvalidArgumentException(message, cause)

    case class CheckpointNotFoundException(message: String = null, cause: Throwable = null)
        extends jni.TensorFlowException(message, cause)
  }
}
