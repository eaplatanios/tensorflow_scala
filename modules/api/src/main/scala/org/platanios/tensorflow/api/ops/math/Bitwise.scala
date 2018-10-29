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

package org.platanios.tensorflow.api.ops.math

import org.platanios.tensorflow.api.core.types.{UByte, IsIntOrUInt, TF}
import org.platanios.tensorflow.api.ops.{Op, Output}

/** Contains functions for constructing bitwise math-related ops.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Bitwise {
  def invert[T: TF : IsIntOrUInt](
      x: Output[T],
      name: String = "BitwiseInvert"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "Invert",
      name = name,
      input = x
    ).build().output
  }

  def populationCount[T: TF : IsIntOrUInt](
      x: Output[T],
      name: String = "BitwisePopulationCount"
  ): Output[UByte] = {
    Op.Builder[Output[T], Output[UByte]](
      opType = "PopulationCount",
      name = name,
      input = x
    ).build().output
  }

  def and[T: TF : IsIntOrUInt](
      x: Output[T],
      y: Output[T],
      name: String = "BitwiseAnd"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "BitwiseAnd",
      name = name,
      input = (x, y)
    ).build().output
  }

  def or[T: TF : IsIntOrUInt](
      x: Output[T],
      y: Output[T],
      name: String = "BitwiseOr"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "BitwiseOr",
      name = name,
      input = (x, y)
    ).build().output
  }

  def xor[T: TF : IsIntOrUInt](
      x: Output[T],
      y: Output[T],
      name: String = "BitwiseXor"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "BitwiseXor",
      name = name,
      input = (x, y)
    ).build().output
  }

  def leftShift[T: TF : IsIntOrUInt](
      x: Output[T],
      y: Output[T],
      name: String = "LeftShift"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "LeftShift",
      name = name,
      input = (x, y)
    ).build().output
  }

  def rightShift[T: TF : IsIntOrUInt](
      x: Output[T],
      y: Output[T],
      name: String = "RightShift"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "RightShift",
      name = name,
      input = (x, y)
    ).build().output
  }
}
