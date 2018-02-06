---
layout: home
title:  "Home"
section: "home"
position: 1
---

# TensorFlow for Scala

[![CircleCI](https://circleci.com/gh/eaplatanios/tensorflow_scala/tree/master.svg?style=shield&circle-token=5ee39233fd9f055c3c65529a2737f7666b26f51a)](https://circleci.com/gh/eaplatanios/tensorflow_scala/tree/master)

This library is a Scala API for [https://www.tensorflow.org](https://www.tensorflow.org). It attempts to provide most of 
the functionality provided by the official Python API, while at the same type being strongly-typed and adding some new 
features. It is a work in progress and a project I started working on for my personal research purposes. Much of the API 
should be relatively stable by now, but things are still likely to change. That is why there is no official release of 
this library yet.

## Main Features

- Easy manipulation of tensors and computations involving tensors (similar to NumPy in Python):
  
  ```scala
  val t1 = Tensor( 1.2, 4.5)
  val t2 = Tensor(-0.2, 1.1)
  t1 + t2 == Tensor(1.0, 5.6)
  ```
  
- High-level API for creating, training, and using neural networks. For example, the following code shows how simple it 
  is to train a multi-layer perceptron for MNIST using TensorFlow for Scala. Here we omit a lot of very powerful 
  features such as summary and checkpoint savers, for simplicity, but these are also very simple to use.
    
  ```scala
  import org.platanios.tensorflow.api._
  import org.platanios.tensorflow.api.tf.learn._
  import org.platanios.tensorflow.data.loaders.MNISTLoader
  
  // Load and batch data using pre-fetching.
  val dataSet = MNISTLoader.load(Paths.get("/tmp"))
  val trainImages = DatasetFromSlices(dataSet.trainImages)
  val trainLabels = DatasetFromSlices(dataSet.trainLabels)
  val trainData =
    trainImages.zip(trainLabels)
        .repeat()
        .shuffle(10000)
        .batch(256)
        .prefetch(10)
  
  // Create the MLP model.
  val input = Input(UINT8, Shape(-1, 28, 28))
  val trainInput = Input(UINT8, Shape(-1))
  val layer = Flatten() >> Cast(FLOAT32) >> 
      Linear(128, name = "Layer_0") >> ReLU(0.1f) >>
      Linear(64, name = "Layer_1") >> ReLU(0.1f) >>
      Linear(32, name = "Layer_2") >> ReLU(0.1f) >>
      Linear(10, name = "OutputLayer")
  val trainingInputLayer = Cast(INT64)
  val loss = SparseSoftmaxCrossEntropy() >> Mean()
  val optimizer = GradientDescent(1e-6)
  val model = Model(input, layer, trainInput, trainingInputLayer, loss, optimizer)
  
  // Create an estimator and train the model.
  val estimator = Estimator(model)
  estimator.train(trainData, StopCriteria(maxSteps = Some(1000000)))
  ```
  
  And by changing a few lines to the following code, you can get checkpoint capability, summaries, and seamless 
  integration with TensorBoard:
  
  ```scala
  val loss = SparseSoftmaxCrossEntropy() >> Mean() >> tf.learn.ScalarSummary("Loss")
  val summariesDir = Paths.get("/tmp/summaries")
  val estimator = Estimator(model, Configuration(Some(summariesDir)))
  estimator.train(
    trainData, StopCriteria(maxSteps = Some(1000000)),
    Seq(
      SummarySaverHook(summariesDir, StepHookTrigger(100)),
      CheckpointSaverHook(summariesDir, StepHookTrigger(1000))),
    tensorBoardConfig = TensorBoardConfig(summariesDir))
  ```
  
  If you now browse to `https://127.0.0.1:6006` while training, you can see the training progress:
  
  <img src="https://eaplatanios.github.io/tensorflow_scala/img/tensorboard_mnist_example_plot.png" alt="tensorboard_mnist_example_plot" width="600px">

- Low-level graph construction API, similar to that of the Python API, but strongly typed wherever possible:

  ```scala
  import org.platanios.tensorflow.api._
  
  val inputs = tf.placeholder(FLOAT32, Shape(-1, 10))
  val outputs = tf.placeholder(FLOAT32, Shape(-1, 10))
  val predictions = tf.createWith(nameScope = "Linear") {
    val weights = tf.variable("weights", FLOAT32, Shape(10, 1), tf.zerosInitializer)
    tf.matmul(inputs, weights)
  }
  val loss = tf.sum(tf.square(predictions - outputs))
  val optimizer = tf.train.AdaGrad(1.0)
  val trainOp = optimizer.minimize(loss)
  ```

- Numpy-like indexing/slicing for tensors. For example:
  
  ```scala
  tensor(2 :: 5, ---, 1) // is equivalent to numpy's 'tensor[2:5, ..., 1]'
  ```
  
- Efficient interaction with the native library that avoids unnecessary copying of data. All tensors are created and 
  managed by the native TensorFlow library. When they are passed to the Scala API (e.g., fetched from a TensorFlow 
  session), we use a combination of weak references and a disposing thread running in the background. Please refer to 
  `tensorflow/src/main/scala/org/platanios/tensorflow/api/utilities/Disposer.scala`, for the implementation.

## Tutorials

- [Object Detection using Pre-Trained Models](https://brunk.io/deep-learning-in-scala-part-3-object-detection.html)

## Funding

Funding for the development of this library has been generously provided by the following sponsors:

[cmu_logo]: {{site.baseurl}}/img/cmu_logo.svg
{: height="150px" width="200px"}
[nsf_logo]: {{site.baseurl}}/img/nsf_logo.svg
{: height="150px" width="150px"}
[afosr_logo]: {{site.baseurl}}/img/afosr_logo.gif
{: height="150px" width="150px"}

{:.funding_table}
|         ![cmu_logo]                     |            ![nsf_logo]            |                  ![afosr_logo]                  |
|:---------------------------------------:|:---------------------------------:|:-----------------------------------------------:|
| **CMU Presidential Fellowship**         | **National Science Foundation**   | **Air Force Office of Scientific Research**     | 
| awarded to Emmanouil Antonios Platanios | Grant #: IIS1250956               | Grant #: FA95501710218                          |
