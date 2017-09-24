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
abstract class TensorFlowException(message: String, cause: Throwable) extends RuntimeException(message, cause)

class CancelledException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class UnknownException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class InvalidArgumentException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class DeadlineExceededException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class NotFoundException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class AlreadyExistsException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class PermissionDeniedException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class UnauthenticatedException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class ResourceExhaustedException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class FailedPreconditionException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class AbortedException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class OutOfRangeException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class UnimplementedException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class InternalException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class UnavailableException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

class DataLossException(message: String, cause: Throwable) extends TensorFlowException(message, cause) {
  def this(message: String) = this(message, null)
}

object CancelledException {
  def apply(message: String): CancelledException = new CancelledException(message, null)
  def apply(message: String, cause: Throwable): CancelledException = new CancelledException(message, cause)
}

object UnknownException {
  def apply(message: String): UnknownException = new UnknownException(message, null)
  def apply(message: String, cause: Throwable): UnknownException = new UnknownException(message, cause)
}

object InvalidArgumentException {
  def apply(message: String): InvalidArgumentException = new InvalidArgumentException(message, null)
  def apply(message: String, cause: Throwable): InvalidArgumentException = new InvalidArgumentException(message, cause)
}

object DeadlineExceededException {
  def apply(message: String): DeadlineExceededException = new DeadlineExceededException(message, null)
  def apply(message: String, cause: Throwable): DeadlineExceededException = new DeadlineExceededException(message, cause)
}

object NotFoundException {
  def apply(message: String): NotFoundException = new NotFoundException(message, null)
  def apply(message: String, cause: Throwable): NotFoundException = new NotFoundException(message, cause)
}

object AlreadyExistsException {
  def apply(message: String): AlreadyExistsException = new AlreadyExistsException(message, null)
  def apply(message: String, cause: Throwable): AlreadyExistsException = new AlreadyExistsException(message, cause)
}

object PermissionDeniedException {
  def apply(message: String): PermissionDeniedException = new PermissionDeniedException(message, null)
  def apply(message: String, cause: Throwable): PermissionDeniedException = new PermissionDeniedException(message, cause)
}

object UnauthenticatedException {
  def apply(message: String): UnauthenticatedException = new UnauthenticatedException(message, null)
  def apply(message: String, cause: Throwable): UnauthenticatedException = new UnauthenticatedException(message, cause)
}

object ResourceExhaustedException {
  def apply(message: String): ResourceExhaustedException = new ResourceExhaustedException(message, null)
  def apply(message: String, cause: Throwable): ResourceExhaustedException = new ResourceExhaustedException(message, cause)
}

object FailedPreconditionException {
  def apply(message: String): FailedPreconditionException = new FailedPreconditionException(message, null)
  def apply(message: String, cause: Throwable): FailedPreconditionException = new FailedPreconditionException(message, cause)
}

object AbortedException {
  def apply(message: String): AbortedException = new AbortedException(message, null)
  def apply(message: String, cause: Throwable): AbortedException = new AbortedException(message, cause)
}

object OutOfRangeException {
  def apply(message: String): OutOfRangeException = new OutOfRangeException(message, null)
  def apply(message: String, cause: Throwable): OutOfRangeException = new OutOfRangeException(message, cause)
}

object UnimplementedException {
  def apply(message: String): UnimplementedException = new UnimplementedException(message, null)
  def apply(message: String, cause: Throwable): UnimplementedException = new UnimplementedException(message, cause)
}

object InternalException {
  def apply(message: String): InternalException = new InternalException(message, null)
  def apply(message: String, cause: Throwable): InternalException = new InternalException(message, cause)
}

object UnavailableException {
  def apply(message: String): UnavailableException = new UnavailableException(message, null)
  def apply(message: String, cause: Throwable): UnavailableException = new UnavailableException(message, cause)
}

object DataLossException {
  def apply(message: String): DataLossException = new DataLossException(message, null)
  def apply(message: String, cause: Throwable): DataLossException = new DataLossException(message, cause)
}
