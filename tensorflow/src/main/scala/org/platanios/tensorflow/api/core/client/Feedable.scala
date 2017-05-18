package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.ops.Op
import org.platanios.tensorflow.api.tensors.Tensor

/**
  * @author Emmanouil Antonios Platanios
  */
trait Feedable[T] {
  def toFeedMap(value: T): Map[Op.Output, Tensor]
}

case class FeedMap(values: Map[Op.Output, Tensor] = Map.empty) {
  def feed[T](feedable: Feedable[T], value: T): FeedMap = {
    FeedMap(values ++ feedable.toFeedMap(value))
  }
}

object FeedMap {
  val empty = FeedMap()

  trait Implicits {
    implicit def feedMap(feeds: Map[Op.Output, Tensor]): FeedMap = FeedMap(feeds)
  }

  object Implicits extends Implicits
}
