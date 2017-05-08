package org.platanios.tensorflow.api

import org.platanios.tensorflow.api.Indexer.{Implicits => IndexerImplicits}
import org.platanios.tensorflow.api.Shape.{Implicits => ShapeImplicits}
import org.platanios.tensorflow.api.Tensor.{Implicits => TensorImplicits}
import org.platanios.tensorflow.api.ops.OpSpecification
import org.platanios.tensorflow.api.ops.Op.{Implicits => OpImplicits}
import org.platanios.tensorflow.api.types.SupportedType.{Implicits => SupportedTypeImplicits}
import org.platanios.tensorflow.api.utilities.Proto.{Implicits => ProtoImplicits}

/**
  * @author Emmanouil Antonios Platanios
  */
trait Implicits
    extends LowPriorityImplicits
        with IndexerImplicits
        with SupportedTypeImplicits
        with ProtoImplicits {
  implicit def deviceImplicitConversion(device: String): OpSpecification => String = Op.deviceImplicitConversion(device)
  implicit def opOutputToInitialValueFunction(opOutput: Op.Output): () => Op.Output = () => opOutput
}

trait LowPriorityImplicits
    extends LowestPriorityImplicits
        with ShapeImplicits
        with OpImplicits

trait LowestPriorityImplicits extends TensorImplicits
