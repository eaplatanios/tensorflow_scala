# TensorFlow Scala

[![CircleCI](https://img.shields.io/circleci/project/github/eaplatanios/tensorflow_scala.svg?style=flat-square)](https://circleci.com/gh/eaplatanios/tensorflow_scala/tree/master)
[![Codacy Badge](https://img.shields.io/codacy/grade/7fae7fba84df4831a80bc20c3bd021df.svg?style=flat-square)](https://www.codacy.com/app/eaplatanios/tensorflow_scala?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=eaplatanios/tensorflow_scala&amp;utm_campaign=Badge_Grade)
[![Chat Room](https://img.shields.io/gitter/room/nwjs/nw.js.svg?style=flat-square)](https://gitter.im/eaplatanios/tensorflow_scala?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![License](https://img.shields.io/github/license/eaplatanios/tensorflow_scala.svg?style=flat-square)

This library is a Scala API for [https://www.tensorflow.org](https://www.tensorflow.org). It attempts to provide most of
the functionality provided by the official Python API, while at the same type being strongly-typed and adding some new
features. It is a work in progress and a project I started working on for my personal research purposes. Much of the API
should be relatively stable by now, but things are still likely to change. That is why there is no official release of
this library yet.

@@@index

* [Installation](installation.md)
* [Getting Started](getting_started.md)
* [Architecture](architecture.md)
* [Contributing](contributing.md)
* [Graph Collections](guides/graph_collections.md)

@@@

## Main Features

  - Easy manipulation of tensors and computations involving tensors (similar to NumPy in Python):

    @@snip [Index.scala](/docs/src/main/scala/Index.scala) { #tensors_example }

  - High-level API for creating, training, and using neural networks. For example, the following code shows how simple it
    is to train a multi-layer perceptron for MNIST using TensorFlow for Scala. Here we omit a lot of very powerful
    features such as summary and checkpoint savers, for simplicity, but these are also very simple to use.

    @@snip [Index.scala](/docs/src/main/scala/Index.scala) { #mnist_example }

    And by changing a few lines to the following code, you can get checkpoint capability, summaries, and seamless
    integration with TensorBoard:

    @@snip [Index.scala](/docs/src/main/scala/Index.scala) { #tensorboard_example }

    If you now browse to `https://127.0.0.1:6006` while training, you can see the training progress:

    <img src="img/tensorboard_mnist_example_plot.png" alt="tensorboard_mnist_example_plot" width="600px">

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

[cmu_logo]: img/cmu_logo.svg
{: height="150px" width="200px"}
[nsf_logo]: img/nsf_logo.svg
{: height="150px" width="150px"}
[afosr_logo]: img/afosr_logo.gif
{: height="150px" width="150px"}

{:.funding_table}
|         ![cmu_logo]                     |            ![nsf_logo]            |                  ![afosr_logo]                  |
|:---------------------------------------:|:---------------------------------:|:-----------------------------------------------:|
| **CMU Presidential Fellowship**         | **National Science Foundation**   | **Air Force Office of Scientific Research**     |
| awarded to Emmanouil Antonios Platanios | Grant #: IIS1250956               | Grant #: FA95501710218                          |
