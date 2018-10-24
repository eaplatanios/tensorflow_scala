import org.platanios.tensorflow.api._
import org.platanios.tensorflow.api.core.types.UByte
import org.platanios.tensorflow.api.tf.learn._
import org.platanios.tensorflow.data.image.MNISTLoader

import java.nio.file.Paths

object Index {
  // #tensors_example
  val t1 = Tensor(1.2, 4.5)
  val t2 = Tensor(-0.2, 1.1)
  t1 + t2 == Tensor(1.0, 5.6)
  // #tensors_example

  // #mnist_example
  // Load and batch data using pre-fetching.
  val dataset = MNISTLoader.load(Paths.get("/tmp"))
  val trainImages = tf.data.datasetFromTensorSlices(dataset.trainImages)
  val trainLabels = tf.data.datasetFromTensorSlices(dataset.trainLabels)
  val trainData =
    trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(256)
        .prefetch(10)

  // Create the MLP model.
  val input = Input(UINT8, Shape(-1, 28, 28))
  val trainInput = Input(UINT8, Shape(-1))
  val layer = Flatten("Input/Flatten") >> Cast[UByte, Float]("Input/CastToFloat") >>
      Linear("Layer_0", 128) >> ReLU("Layer_0/Activation", 0.1f) >>
      Linear("Layer_1", 64) >> ReLU("Layer_1/Activation", 0.1f) >>
      Linear("Layer_2", 32) >> ReLU("Layer_2/Activation", 0.1f) >>
      Linear("OutputLayer", 10)
  val trainingInputLayer = Cast[UByte, Long]("TrainInput/CastToLong")
  val loss = SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss") >>
      Mean("Loss/Mean")
  val optimizer = tf.train.GradientDescent(1e-6f)
  val model = Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)

  // Create an estimator and train the model.
  val estimator = InMemoryEstimator(model)
  estimator.train(() => trainData, StopCriteria(maxSteps = Some(1000000)))
  // #mnist_example

  // #tensorboard_example
  val loss = SparseSoftmaxCrossEntropy[Float, Long, Float]("Loss") >>
      Mean("Loss/Mean") >>
      ScalarSummary(name = "Loss", tag = "Loss")
  val summariesDir = Paths.get("/tmp/summaries")
  val estimator = InMemoryEstimator(
    modelFunction = model,
    configurationBase = Configuration(Some(summariesDir)),
    trainHooks = Set(
      SummarySaver(summariesDir, StepHookTrigger(100)),
      CheckpointSaver(summariesDir, StepHookTrigger(1000))),
    tensorBoardConfig = TensorBoardConfig(summariesDir))
  estimator.train(() => trainData, StopCriteria(maxSteps = Some(100000)))
  // #tensorboard_example
}
