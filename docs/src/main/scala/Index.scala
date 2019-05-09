import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.tf.learn._
import org.platanios.tensorflow.data.image.MNISTLoader

import java.nio.file.Paths

trait IndexTensorsExample {
  // #tensors_example
  val t1 = Tensor(1.2, 4.5)
  val t2 = Tensor(-0.2, 1.1)
  t1 + t2 == Tensor(1.0, 5.6)
  // #tensors_example
}

trait IndexLowLevelExample {
  // #low_level_example
  val inputs      = tf.placeholder[Float](Shape(-1, 10))
  val outputs     = tf.placeholder[Float](Shape(-1, 10))
  val predictions = tf.nameScope("Linear") {
    val weights = tf.variable[Float]("weights", Shape(10, 1), tf.ZerosInitializer)
    tf.matmul(inputs, weights)
  }
  val loss        = tf.sum(tf.square(predictions - outputs))
  val optimizer   = tf.train.AdaGrad(1.0f)
  val trainOp     = optimizer.minimize(loss)
  // #low_level_example
}

trait IndexSliceExample {
  val tensor = Tensor.zeros[Float](Shape(10, 2, 3, 4, 5, 20))
  // #slice_example
  tensor(2 :: 5, ---, 1) // is equivalent to numpy's 'tensor[2:5, ..., 1]'
  // #slice_example
}

trait IndexMNISTExample {
  // #mnist_example
  // Load and batch data using pre-fetching.
  val dataset = MNISTLoader.load(Paths.get("/tmp"))
  val trainImages = tf.data.datasetFromTensorSlices(dataset.trainImages.toFloat)
  val trainLabels = tf.data.datasetFromTensorSlices(dataset.trainLabels.toLong)
  val trainData =
    trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(256)
        .prefetch(10)

  // Create the MLP model.
  val input = Input(FLOAT32, Shape(-1, 28, 28))
  val trainInput = Input(INT64, Shape(-1))
  val layer = Flatten[Float]("Input/Flatten") >>
      Linear[Float]("Layer_0", 128) >> ReLU[Float]("Layer_0/Activation", 0.1f) >>
      Linear[Float]("Layer_1", 64) >> ReLU[Float]("Layer_1/Activation", 0.1f) >>
      Linear[Float]("Layer_2", 32) >> ReLU[Float]("Layer_2/Activation", 0.1f) >>
      Linear[Float]("OutputLayer", 10)
  val loss = SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss") >>
      Mean("Loss/Mean")
  val optimizer = tf.train.GradientDescent(1e-6f)
  val model = Model.simpleSupervised(input, trainInput, layer, loss, optimizer)

  // Create an estimator and train the model.
  val estimator = InMemoryEstimator(model)
  estimator.train(() => trainData, StopCriteria(maxSteps = Some(1000000)))
  // #mnist_example
}

trait IndexTensorBoard extends IndexMNISTExample {
  // #tensorboard_example
  override val loss = SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss") >>
      Mean("Loss/Mean") >>
      ScalarSummary(name = "Loss", tag = "Loss")
  val summariesDir = Paths.get("/tmp/summaries")
  override val estimator = InMemoryEstimator(
    modelFunction = model,
    configurationBase = Configuration(Some(summariesDir)),
    trainHooks = Set(
      SummarySaver(summariesDir, StepHookTrigger(100)),
      CheckpointSaver(summariesDir, StepHookTrigger(1000))),
    tensorBoardConfig = TensorBoardConfig(summariesDir))
  estimator.train(() => trainData, StopCriteria(maxSteps = Some(100000)))
  // #tensorboard_example
}
