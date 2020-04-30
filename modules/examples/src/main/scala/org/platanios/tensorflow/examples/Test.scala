package org.platanios.tensorflow.examples

import java.nio.file.Paths

import org.platanios.tensorflow.api.core.Shape
import org.platanios.tensorflow.api.implicits.helpers.{OutputStructure, OutputToDataType, OutputToShape}
import org.platanios.tensorflow.api.learn.Model
import org.platanios.tensorflow.api.learn.estimators.InMemoryEstimator
import org.platanios.tensorflow.api.learn.layers._
import org.platanios.tensorflow.api.ops.training.optimizers.AdaGrad
import org.platanios.tensorflow.api.ops.variables.{GlorotNormalInitializer, ZerosInitializer}
import org.platanios.tensorflow.api.tensors.Tensor
import org.platanios.tensorflow.api.{FLOAT32, INT64, Output, tf}

object EvaluatorExample {

  def main(args: Array[String]): Unit = {
    val path = Paths.get("/Users/eaplatanios/Desktop/test.npy")

    // val tensor = Tensor[Double](1.5, -30.4)
    // tensor.writeNPY(path)
    val readTensor = Tensor.fromNPY[Double](path)
    println(readTensor.summarize())
  }
}
