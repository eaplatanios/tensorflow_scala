---
layout: home
title:  "Home"
section: "home"
position: 1
---

# TensorFlow for Scala

[![Build Status](https://travis-ci.com/eaplatanios/tensorflow_scala.svg?token=VBPxqvcGXTuwbjkVyN68&branch=master)](https://travis-ci.com/eaplatanios/tensorflow_scala)

This is an attempt to replicate most of the TensorFlow Python API 
functionality in Scala. It is a work in progress and a project I started 
working on for my personal research purposes. The API is thus very 
likely to keep changing until I settle with something that makes me 
happy. Having said that, I try to stick as close as possible to the 
Python API usage to make transition easy for users and much of what I 
have already implemented is likely to stay as is. Most of the code is 
ported from the Python API with changes to make it more Scala-friendly 
and to make use of useful Scala features and strong-typing.

People who are new to TensorFlow should first go through the official 
Python API documentation at 
[https://www.tensorflow.org](https://www.tensorflow.org). Most of what 
you read there applies here too.

## Main Features

- Exposed API namespace similar to that of the Python API. For example:
  - `tf.constant(...)` creates a constant op.
  - `tf.reshape(...)` creates a reshape op.
  - `tf.Graph(...)` creates a graph.
  - `tf.Session(...)` creates a session.
  - etc.
- Straightforward API for graph creation. For example:
  ```scala
  import org.platanios.tensorflow.api._
  
  val inputs = tf.placeholder(FLOAT32, Shape(-1, 10))
  val outputs = tf.placeholder(FLOAT32, Shape(-1, 10))
  val predictions = tf.createWith(nameScope = "Linear") {
    val weights = tf.variable("weights", FLOAT32, Shape(10, 1), tf.zerosInitializer)
    val predictions = tf.matmul(inputs, weights)
    predictions
  }
  val loss = tf.sum(tf.square(predictions - outputs))
  val optimizer = tf.train.AdaGrad(1.0)
  val trainOp = optimizer.minimize(loss)
  ```
- Efficient interaction with the native library that avoids unnecessary 
  copying of data.
  - All tensors are backed by a `DirectByteBuffer` so that the memory can 
    be shared with the native TensorFlow library.
  - For tensors that are created in the Scala API and passed on the 
    native library (e.g., fed into a TensorFlow session), we create a 
    global reference to make the JVM garbage collector aware of the fact 
    that the native library is using that 
    tensor. We also pass a custom deallocator to the native library that 
    simply deletes that global reference.
  - For tensors created by the TensorFlow native library and passed on the 
    Scala API (e.g., fetched from a TensorFlow session), we use a 
    combination of weak references and a disposing thread running in the 
    background. Please refer to 
    `tensorflow/src/main/scala/org/platanios/tensorflow/api/utilities/Disposer.scala`.
- Numpy-like indexing/slicing for tensors. For example:
  ```scala
  tensor(2 :: 5, ---, 1) // is equivalent to numpy's 'tensor[2:5, ..., 1]'
  ```
- Useful implicits that make using the library almost as simple as using 
  the Python API, while retaining type-safety (for the most part).

## Funding

Funding for the development of this library has been generously provided by the following sponsors:

[nsf_logo]: nsf_logo.svg
{: height="72px" width="72px"}

|                             |            ![nsf_logo]            |                                                 |
|-----------------------------|-----------------------------------|-------------------------------------------------|
| CMU Presidential Fellowship | National Science Foundation (NSF) | Air Force Office of Scientific Research (AFOSR) | 
|                             | Grant #: IIS1250956               | Grant #: FA95501710218                          |
