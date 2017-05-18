package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.core.Indexer.{Implicits => IndexerImplicits}
import org.platanios.tensorflow.api.core.Shape.{Implicits => ShapeImplicits}
import org.platanios.tensorflow.api.core.client.Fetchable.{Implicits => FetchableImplicits}
import org.platanios.tensorflow.api.ops.{Op, OpSpecification}
import org.platanios.tensorflow.api.ops.Op.{Implicits => OpImplicits}
import org.platanios.tensorflow.api.tensors.Tensor.{Implicits => TensorImplicits}
import org.platanios.tensorflow.api.tensors.TensorFlowNative.{Implicits => TensorNativeImplicits}
import org.platanios.tensorflow.api.types.SupportedType.{Implicits => SupportedTypeImplicits}

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait Implicits
    extends LowPriorityImplicits
        with IndexerImplicits {
  implicit def deviceImplicitConversion(device: String): (OpSpecification => String) = {
    Op.deviceImplicitConversion(device)
  }

  implicit def opOutputToInitialValueFunction(opOutput: Op.Output): () => Op.Output = () => opOutput
}

object Implicits extends Implicits

private[api] trait LowPriorityImplicits
    extends LowestPriorityImplicits
        with ShapeImplicits
        with OpImplicits
        with FetchableImplicits

private[api] trait LowestPriorityImplicits
    extends TensorImplicits
        with TensorNativeImplicits
        with SupportedTypeImplicits
