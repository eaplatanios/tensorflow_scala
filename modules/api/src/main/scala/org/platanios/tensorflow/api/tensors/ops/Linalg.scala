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
package org.platanios.tensorflow.api.tensors.ops

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.core.exception.InvalidShapeException
import org.platanios.tensorflow.api.core.types._
import org.platanios.tensorflow.api.implicits.Implicits._
import org.platanios.tensorflow.api.tensors._
import org.platanios.tensorflow.api.utilities.DefaultsTo.IntDefault
import org.platanios.tensorflow.jni.generated.tensors.{Linalg => NativeTensorOpsLinAlg}

import scala.language.postfixOps

/**
  * Defines linear algebra ops similar to the
  * ones defined in tf.linalg package of the Python TF API
  *
  *
  */
trait Linalg {

  def matrixDeterminant[T: TF: IsRealOrComplex](matrix: Tensor[T]): Tensor[T] = {
    Tensor.fromNativeHandle[T](
      NativeTensorOpsLinAlg.matrixDeterminant(executionContext.value.nativeHandle, matrix.nativeHandle)
    )
  }

  /**
    * Computes (sign(det(x)) log(|det(x)|)) for an input x.
    *
    * @param matrix A matrix of shape [N, M, M]
    * 
    * @return A tuple having the results.
    *
    */
  def logMatrixDeterminant[T: TF: IsRealOrComplex](matrix: Tensor[T]): (Tensor[T], Tensor[T]) = {
    val results = NativeTensorOpsLinAlg
      .logMatrixDeterminant(executionContext.value.nativeHandle, matrix.nativeHandle).map(
        h => Tensor.fromNativeHandle[T](h)
      )
    (results.head, results.last)
  }

}
