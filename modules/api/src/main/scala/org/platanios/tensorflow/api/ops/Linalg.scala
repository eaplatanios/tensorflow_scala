/* Copyright 2019, T.AI Labs. All Rights Reserved.
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
package org.platanios.tensorflow.api.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidArgumentException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault
//import com.google.protobuf.ByteString.Output

import org.tensorflow.framework.AttrValue

import scala.language.postfixOps
import com.google.protobuf.Descriptors.FieldDescriptor

/**
  * Defines linear algebra ops similar to the
  * ones defined in tf.linalg package of the Python TF API
  *
  *
  */
trait Linalg {

  def matrixDeterminant[T: TF: IsRealOrComplex](matrix: Output[T], name: String = "MatrixDeterminant"): Output[T] = {
    Op.Builder[Output[T], Output[T]](
        opType = "MatrixDeterminant",
        name = name,
        input = matrix
      ).build().output
  }

  /**
    * Computes (sign(det(x)) log(|det(x)|)) for an input x.
    *
    * @tparam T The underlying scala type of the matrix elements.
    * 
    * @param matrix A matrix of shape [N, M, M]
    * @param name An optional name to assign to the op.
    *
    * @return A tuple having the results.
    *
    */
  def logMatrixDeterminant[T: TF: IsRealOrComplex](
      matrix: Output[T],
      name: String = "LogMatrixDeterminant"
  ): (Output[T], Output[T]) = {
    Op.Builder[Output[T], (Output[T], Output[T])](
        opType = "LogMatrixDeterminant",
        name = name,
        input = matrix
      ).build().output
  }

  /**
    * Computes inv(A), assuming matrix A is invertible and of shape [..., M, M]
    *
    * @tparam T The underlying scala type of the matrix elements.
    * @param matrix The matrix to invert.
    * @param adjoint If set to true, returns the adjoint, defaults to false.
    * @param name An optional name to assign to the op.
    *
    */
  def matrixInverse[T: TF: IsRealOrComplex](
      matrix: Output[T],
      adjoint: Boolean = false,
      name: String = "MatrixInverse"
  ): Output[T] =
    Op.Builder[Output[T], Output[T]](
        opType = "MatrixInverse",
        name = name,
        input = matrix
      ).setAttribute("adjoint", adjoint).build().output

}
