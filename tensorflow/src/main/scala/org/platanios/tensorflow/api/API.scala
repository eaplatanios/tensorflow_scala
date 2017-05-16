package org.platanios.tensorflow.api

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait API
    extends core.CoreAPI
        with ops.OpAPI
        with tensors.TensorAPI
        with types.DataTypeAPI
