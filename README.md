<div align="center">
  <img src="https://raw.githubusercontent.com/eaplatanios/tensorflow_scala/master/docs/images/logo.svg?sanitize=true" width="350px" height="242px"><br>
</div>

-----------------

[![CircleCI](https://img.shields.io/circleci/project/github/eaplatanios/tensorflow_scala.svg?style=flat-square)](https://circleci.com/gh/eaplatanios/tensorflow_scala/tree/master)
[![Codacy Badge](https://img.shields.io/codacy/grade/7fae7fba84df4831a80bc20c3bd021df.svg?style=flat-square)](https://www.codacy.com/app/eaplatanios/tensorflow_scala?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=eaplatanios/tensorflow_scala&amp;utm_campaign=Badge_Grade)
![License](https://img.shields.io/github/license/eaplatanios/tensorflow_scala.svg?style=flat-square)
[![API Docs](https://img.shields.io/badge/docs-api-lightgrey.svg?longCache=true&style=flat-square&logo=read-the-docs&logoColor=white)](http://platanios.org/tensorflow_scala/api/api)
[![JNI Docs](https://img.shields.io/badge/docs-jni-lightgrey.svg?longCache=true&style=flat-square&logo=read-the-docs&logoColor=white)](http://platanios.org/tensorflow_scala/api/jni)
[![Data Docs](https://img.shields.io/badge/docs-data-lightgrey.svg?longCache=true&style=flat-square&logo=read-the-docs&logoColor=white)](http://platanios.org/tensorflow_scala/api/data)
[![Examples Docs](https://img.shields.io/badge/docs-examples-lightgrey.svg?longCache=true&style=flat-square&logo=read-the-docs&logoColor=white)](http://platanios.org/tensorflow_scala/api/examples)

This library is a Scala API for [https://www.tensorflow.org](https://www.tensorflow.org). It attempts to provide most of
the functionality provided by the official Python API, while at the same type being strongly-typed and adding some new
features. It is a work in progress and a project I started working on for my personal research purposes. Much of the API
should be relatively stable by now, but things are still likely to change.

[![Chat Room](https://img.shields.io/badge/chat-gitter-ed1965.svg?longCache=true&style=flat-square&logo=gitter)](https://gitter.im/eaplatanios/tensorflow_scala?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please refer to the main website for documentation and tutorials. Here
are a few useful links:

  - [Installation](https://eaplatanios.github.io/tensorflow_scala/installation.html)
  - [Getting Started Guide](https://eaplatanios.github.io/tensorflow_scala/getting_started.html)
  - [Library Architecture](https://eaplatanios.github.io/tensorflow_scala/architecture.html)
  - [Contributing](https://eaplatanios.github.io/tensorflow_scala/contributing.html)

## Main Features

  - Easy manipulation of tensors and computations involving tensors (similar to NumPy in Python):

    ```scala
    val t1 = Tensor(1.2, 4.5)
    val t2 = Tensor(-0.2, 1.1)
    t1 + t2 == Tensor(1.0, 5.6)
    ```

  - Low-level graph construction API, similar to that of the Python API, but strongly typed wherever possible:

    ```scala
    val inputs      = tf.placeholder[Float](Shape(-1, 10))
    val outputs     = tf.placeholder[Float](Shape(-1, 10))
    val predictions = tf.nameScope("Linear") {
      val weights = tf.variable[Float]("weights", Shape(10, 1), tf.ZerosInitializer)
      tf.matmul(inputs, weights)
    }
    val loss        = tf.sum(tf.square(predictions - outputs))
    val optimizer   = tf.train.AdaGrad(1.0f)
    val trainOp     = optimizer.minimize(loss)
    ```

  - Numpy-like indexing/slicing for tensors. For example:

    ```scala
    tensor(2 :: 5, ---, 1) // is equivalent to numpy's 'tensor[2:5, ..., 1]'
    ```

  - High-level API for creating, training, and using neural networks. For example, the following code shows how simple it
    is to train a multi-layer perceptron for MNIST using TensorFlow for Scala. Here we omit a lot of very powerful
    features such as summary and checkpoint savers, for simplicity, but these are also very simple to use.

    ```scala
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
    ```

    And by changing a few lines to the following code, you can get checkpoint capability, summaries, and seamless
    integration with TensorBoard:

    ```scala
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
    ```

    If you now browse to `https://127.0.0.1:6006` while training, you can see the training progress:

    <img src="https://platanios.org/tensorflow_scala/assets/images/tensorboard_mnist_example_plot.png" alt="tensorboard_mnist_example_plot" width="600px">

  - Efficient interaction with the native library that avoids unnecessary copying of data. All tensors are created and
    managed by the native TensorFlow library. When they are passed to the Scala API (e.g., fetched from a TensorFlow
    session), we use a combination of weak references and a disposing thread running in the background. Please refer to
    `tensorflow/src/main/scala/org/platanios/tensorflow/api/utilities/Disposer.scala`, for the implementation.

## Tutorials

- [Object Detection using Pre-Trained Models](https://brunk.io/deep-learning-in-scala-part-3-object-detection.html)

## Funding

Funding for the development of this library has been generously provided by the following sponsors:

|<img src="https://platanios.org/tensorflow_scala/assets/images/cmu_logo.svg" alt="cmu_logo" width="200px" height="150px">|<img src="https://platanios.org/tensorflow_scala/assets/images/nsf_logo.svg" alt="nsf_logo" width="150px" height="150px">|<img src="https://platanios.org/tensorflow_scala/assets/images/afosr_logo.gif" alt="afosr_logo" width="150px" height="150px">|
|:---------------------------------------:|:---------------------------------:|:-----------------------------------------------:|
| **CMU Presidential Fellowship**         | **National Science Foundation**   | **Air Force Office of Scientific Research**     | 
| awarded to Emmanouil Antonios Platanios | Grant #: IIS1250956               | Grant #: FA95501710218                          |

TensorFlow, the TensorFlow logo, and any related marks are trademarks of Google Inc.

<!---

## Some TODOs

  - [ ] Figure out what the proper to way to handle Int vs Long shapes is, so that we can use Long shapes without hurting GPU performance.
  - [ ] Make the optimizers typed (with respect to their state, at least).
  - [ ] Make the gradients function retain types (we need a type trait for that).
  - [ ] Dispose dataset iterators automatically.
  - [ ] Fixed all `[TYPE] !!!` code TODOs.

  - [ ] Session execution context (I'm not sure if that's good to have)
  - [ ] Session reset functionality
  - [ ] Variables slicing
  - [ ] Slice assignment
  - [ ] Support for `CriticalSection`.
  - [ ] tfdbg / debugging support
  - [ ] tfprof / op statistics collection

  - Switch to using JUnit for all tests.
  - Add convenience implicit conversions for shapes (e.g., from tuples or sequences of integers).
  - Create a "Scope" class and companion object.
  - Variables API:
    - Clean up the implementation of variable scopes and stores and integrate it with "Scope".
    - Make 'PartitionedVariable' extend 'Variable'.
    - After that change, all 'getPartitionedVariable' methods can be integrated with the 'getVariable' methods, which will
      simplify the variables API.
  - Switch to using "Seq" instead of "Array" wherever possible.
  - Op creation:
    - Reset default graph
    - Register op statistics
  - Fix Travis CI support (somehow load the native library)

- Website margins are a little large relative to the content in mobile
- Make the code blocks scroll rather than wrap

To publish the documentation website we use the following commands:

```bash
sbt docs/previewSite     # To preview the website
sbt docs/ghpagesPushSite # To publish the website
```

```bash
find . -name '*.h' | cpio -pdmu ../tensorflow_scala/jni/src/main/native/include/
```

-->
