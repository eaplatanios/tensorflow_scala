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

class CancelledException(message: String) extends TensorFlowException(message)
class UnknownException(message: String) extends TensorFlowException(message)
class InvalidArgumentException(message: String) extends TensorFlowException(message)
class DeadlineExceededException(message: String) extends TensorFlowException(message)
class NotFoundException(message: String) extends TensorFlowException(message)
class AlreadyExistsException(message: String) extends TensorFlowException(message)
class PermissionDeniedException(message: String) extends TensorFlowException(message)
class UnauthenticatedException(message: String) extends TensorFlowException(message)
class ResourceExhaustedException(message: String) extends TensorFlowException(message)
class FailedPreconditionException(message: String) extends TensorFlowException(message)
class AbortedException(message: String) extends TensorFlowException(message)
class OutOfRangeException(message: String) extends TensorFlowException(message)
class UnimplementedException(message: String) extends TensorFlowException(message)
class InternalException(message: String) extends TensorFlowException(message)
class UnavailableException(message: String) extends TensorFlowException(message)
class DataLossException(message: String) extends TensorFlowException(message)

object CancelledException {
  def apply(message: String): CancelledException = new CancelledException(message)
}

object UnknownException {
  def apply(message: String): UnknownException = new UnknownException(message)
}

object InvalidArgumentException {
  def apply(message: String): InvalidArgumentException = new InvalidArgumentException(message)
}

object DeadlineExceededException {
  def apply(message: String): DeadlineExceededException = new DeadlineExceededException(message)
}

object NotFoundException {
  def apply(message: String): NotFoundException = new NotFoundException(message)
}

object AlreadyExistsException {
  def apply(message: String): AlreadyExistsException = new AlreadyExistsException(message)
}

object PermissionDeniedException {
  def apply(message: String): PermissionDeniedException = new PermissionDeniedException(message)
}

object UnauthenticatedException {
  def apply(message: String): UnauthenticatedException = new UnauthenticatedException(message)
}

object ResourceExhaustedException {
  def apply(message: String): ResourceExhaustedException = new ResourceExhaustedException(message)
}

object FailedPreconditionException {
  def apply(message: String): FailedPreconditionException = new FailedPreconditionException(message)
}

object AbortedException {
  def apply(message: String): AbortedException = new AbortedException(message)
}

object OutOfRangeException {
  def apply(message: String): OutOfRangeException = new OutOfRangeException(message)
}

object UnimplementedException {
  def apply(message: String): UnimplementedException = new UnimplementedException(message)
}

object InternalException {
  def apply(message: String): InternalException = new InternalException(message)
}

object UnavailableException {
  def apply(message: String): UnavailableException = new UnavailableException(message)
}

object DataLossException {
  def apply(message: String): DataLossException = new DataLossException(message)
}
