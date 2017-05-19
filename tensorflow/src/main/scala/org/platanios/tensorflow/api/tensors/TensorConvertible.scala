// Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

package org.platanios.tensorflow.api.tensors

/** Helper trait for tagging tensor convertible objects so that implicit conversions to tensors can be used.
  *
  * @author Emmanouil Antonios Platanios
  */
trait TensorConvertible {
  /** Returns the [[Tensor]] that this [[TensorConvertible]] object represents. */
  def toTensor: Tensor
}
