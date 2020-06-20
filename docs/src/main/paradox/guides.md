[tf_python]: https://www.tensorflow.org/get_started/get_started
[tf_estimators]: https://www.tensorflow.org/programmers_guide/estimators

# Guides

Similar to the [TensorFlow Python API][tf_python], by
Google, TensorFlow for Scala provides multiple APIs. The
lowest level API -- Core API -- provides you with complete
programming control. the core API is suitable for machine
learning researchers and others who require fine levels of
control over their models. The higher level APIs are built
on top of the Core API. These higher level APIs are
typically easier to learn and use. In addition, the higher
level APIs make repetitive tasks easier and more consistent
between different users. A high-level API like the Learn
API helps you manage datasets, models, (distributed)
training, and inference.

The main APIs of TensorFlow Scala are the following:

@@@ index

  - **[Tensors:](guides/tensors.md)** Provides a simple way
    for manipulating tensors and performing computations
    involving tensors. This is similar in functionality to
    the [NumPy](http://www.numpy.org) library used by
    Python programmers.
  - **[Graph Construction:](guides/graph_construction.md)**
    Low-level graph construction interface, similar to that
    offered by the TensorFlow Python API, with the main
    difference being that this interface is
    statically-typed.
  - **[High-Level Learning:](guides/estimators.md)** High-level
    interface for creating, training, and using neural
    networks. This is similar in functionality to the
    [Keras](https://keras.io) library used by Python
    programmers, with the main difference being that it is
    strongly-typed and offers a much richer functional
    interface for building neural networks. Furthermore, it
    supports distributed training in a way that is very
    similar to the [TensorFlow Estimators API](tf_estimators).
  - **[Adding Ops](guides/adding_ops.md)**
    Example for how to add TensorFlow ops to TensorFlow Scala.

@@@

The fact that this library is statically-typed is mentioned
a couple times in the above paragraph and that's because it
is a very important feature. It means that many problems
with the code you write will show themselves at compile
time, which means that your chances of running into the
experience of waiting for a neural network to train for a
week only to find out that your evaluation code crashed and
you lost everything, decrease significantly.

It is recommended to first go through the **Tensors** guide,
and then go from high-level to low-level concepts as you
progress (i.e., read through the **High-Level Learning**
guide first and then through the **Graph Construction**
guide). Concepts such as the TensorFlow graph and sessions
only appear in the **Graph Construction** guide.

@@@ warning { title='Relationship to the TensorFlow Python API' }

These guides borrow a lot of material from the official
[Python API documentation][tf_python] of TensorFlow and
adapt it for the purposes of TensorFlow Scala. They also
introduce a lot of new constructs specific to this library.

@@@
