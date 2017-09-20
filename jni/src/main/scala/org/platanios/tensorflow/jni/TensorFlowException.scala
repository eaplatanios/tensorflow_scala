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

package org.platanios.tensorflow.jni

/** Abstract class for representing TensorFlow exceptions.
  *
  * @author Emmanouil Antonios Platanios
  */
abstract class TensorFlowException(message: String) extends RuntimeException(message)

final class CancelledException(message: String) extends TensorFlowException(message)
final class UnknownException(message: String) extends TensorFlowException(message)
final class InvalidArgumentException(message: String) extends TensorFlowException(message)
final class DeadlineExceededException(message: String) extends TensorFlowException(message)
final class NotFoundException(message: String) extends TensorFlowException(message)
final class AlreadyExistsException(message: String) extends TensorFlowException(message)
final class PermissionDeniedException(message: String) extends TensorFlowException(message)
final class UnauthenticatedException(message: String) extends TensorFlowException(message)
final class ResourceExhaustedException(message: String) extends TensorFlowException(message)
final class FailedPreconditionException(message: String) extends TensorFlowException(message)
final class AbortedException(message: String) extends TensorFlowException(message)
final class OutOfRangeException(message: String) extends TensorFlowException(message)
final class UnimplementedException(message: String) extends TensorFlowException(message)
final class InternalException(message: String) extends TensorFlowException(message)
final class UnavailableException(message: String) extends TensorFlowException(message)
final class DataLossException(message: String) extends TensorFlowException(message)
