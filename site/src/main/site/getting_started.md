---
layout: docs
title: "Getting Started"
section: "getting_started"
position: 3
---

[tf_python]: https://www.tensorflow.org/get_started/get_started

# Introduction

Similar to the [TensorFlow Python API][tf_python], by Google, TensorFlow for Scala provides multiple APIs. The lowest
level API -- TensorFlow Core -- provides you with complete programming control. TensorFlow Core is suitable for machine
learning researchers and others who require fine levels of control over their models. The higher level APIs are built on
top of TensorFlow Core. These higher level APIs are typically easier to learn and use than TensorFlow Core. In addition,
the higher level APIs make repetitive tasks easier and more consistent between different users. A high-level API like
the Learn API (which is currently under development) helps you manage data sets, models, training, and inference.

The [Getting Started](#getting-started-1) guide begins with an introduction to the fundamentals of TensorFlow for Scala. 
It has some overlap with the official [Python API documentation][tf_python], but also introduces new constructs specific 
to the Scala API.

# Tensors

Tensors are the main data structure used in TensorFlow and are thus a good place to start.

```tut:invisible
import org.platanios.tensorflow.api._
```

A tensor is the main data structure used in TensorFlow. It represents a multi-dimensional array and can hold elements of 
various data types. For example, the following code creates an integer tensor filled with zeros with shape `[2, 5]` 
(i.e., a two-dimensional array holding integer values, where the first dimension size is 2 and the second is 5):
```tut
val tensor = Tensor.zeros(INT32, Shape(2, 5))
```
You can print the contents of a tensor as follows:
```tut
tensor.summarize()
```

# Graph

