package org.platanios.tensorflow.jni

import org.scalatest._
import org.tensorflow.framework.OpList

import scala.collection.JavaConverters._

/**
  * @author Emmanouil Antonios Platanios
  */
class TensorFlowSpec extends FlatSpec {
  "The TensorFlow library version" should "have non-zero length" in {
    val opList = OpList.parseFrom(Operation.allOps).getOpList.asScala
    assert(TensorFlow.version.length > 0)
  }
}
