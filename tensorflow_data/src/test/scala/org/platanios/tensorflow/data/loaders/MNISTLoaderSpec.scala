package org.platanios.tensorflow.data.loaders

import java.nio.file.{Files, Path, Paths}

import org.scalatest.{FlatSpec, Matchers}

/**
  * @author Emmanouil Antonios Platanios
  */
class MNISTLoaderSpec extends FlatSpec with Matchers {
  val directory: Path = Paths.get("/Users/Anthony/Development/GitHub/tensorflow_scala/temp/data/mnist")
  Files.createDirectories(directory)

  "The MNIST data set loader" must "work" in {
    val dataSet = MNISTLoader.load(directory)
    val label0 = dataSet.trainLabels.summarize(10)
    print(dataSet)
  }
}
