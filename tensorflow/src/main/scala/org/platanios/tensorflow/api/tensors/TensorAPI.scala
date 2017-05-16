package org.platanios.tensorflow.api.tensors

import org.platanios.tensorflow.api.tensors

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait TensorAPI {
  type Tensor = tensors.Tensor
  type FixedSizeTensor = tensors.FixedSizeTensor
  type NumericTensor = tensors.NumericTensor

  val Tensor = tensors.Tensor

  type Order = tensors.Order
  val RowMajorOrder = tensors.RowMajorOrder
}
