# 0.3.0 Static Data Types and More

With this release we have finally added support for static data type
information for tensors (not for symbolic tensors yet though -- for now
we effectively have support for a statically-typed version of `numpy`
for Scala). This is an important milestone and contributes significantly
to type safety, which can help catch errors at compile time, rather than
runtime. For example:

```scala
val t1 = Tensor(0.5, 1) // The inferred type is Tensor[FLOAT64].
val t2 = Tensor(1, 2)   // The inferred type is Tensor[INT32].
val t3 = t1 + t2        // The inferred type is Tensor[FLOAT64].
val t4 = t3.isNaN       // The inferred type is Tensor[BOOLEAN].
val t5 = t3.any()       // Fails at compile-time because `any()` is only
                        // supported for Tensor[BOOLEAN].
```

Other new features include:

- Improvements to the high-level learn API:
  - Layers can now provide and use their own parameter generator, and
    can also access the current training step
    (using `Layer.currentStep`).
  - Layers now support `.map(...)`.
- Added support for `tf.logSigmoid` and `tf.lrn`.
- Added support for the following new metrics:
  - Grouped precision.
  - Precision-at-k.
- Added support for loading the extreme classification repository
  datasets (i.e., `data.XCLoader`).

# 0.2.4 Minor Fix

Fixed an issue with the packaged pre-compiled TensorFlow binaries that
affected Linux platforms.

# 0.2.3 Compatibility with TensorFlow 1.9-rc1

Added compatibility with TensorFlow 1.9-rc1.

# 0.2.2 Pre-compiled Binaries Update

In this release we have updated the precompiled TensorFlow binaries
distributed with this library.

# 0.2.1 Packaging Fix

In this release we have fixed an issue related to the packaging and
distributing of the pre-compiled TensorFlow shared libraries.

# 0.2.0 Updates

In this release we have:

  - Added support for incremental compilation.
  - Added support for [Horovod](https://github.com/uber/horovod).
  - Added support for timelines to allow for easy profiling of
    TensorFlow graphs.
  - Fixed a major memory leak (issue #87).
  - Updated the JNI bindings to be compatible with the TensorFlow
    1.9.0 release.
  - Added support for obtaining the list of available devices from
    within Scala.
  - Fixed bugs for some control flow ops.
  - Added support for `tf.cases`.
  - Added support for the RMSProp optimizer, the lazy Adam optimizer,
    the [AMSGrad](https://openreview.net/pdf?id=ryQu7f-RZ) optimizer,
    the lazy AMSGrad optimizer, and the
    [YellowFin](https://arxiv.org/pdf/1706.03471.pdf) optimizer.
  - Added more learning rate decay schemes:
    - Cosine decay.
    - Cycle-linear 10x decay.
    - Square-root decay.
    - More warm-up decay schedules.
  - Added support for dataset interleave ops.
  - Fixed some bugs related to variable scopes and variable sharing.
  - Fixed some bugs related to functional ops.
  - Added support for some new image-related ops, under the namespace
    `tf.image`.
  - Improved consistency for the creation of initializer ops.
  - Added support for the `tf.initializer` op creation context.
  - Exposed part of the `TensorArray` API.
  - Exposed `tf.Op.Builder` in the public API.
  - Improvements to the learn API:
    - Refactored `mode` into an implicit argument.
    - Improved the evaluator hook.
    - Removed the layer creation context mechanism, to be refactored
      later. It was causing some issues due to bad design and unclear
      semantics. The plan is to implement this, in the near future, as
      wrapper creation context layers.
    - Improved the `Model` class.
    - Fixed a bug that was causing some issues related to inference
      hooks in the in-memory estimator.
    - Improved logging.
  - Added support for reading and writing numpy (i.e., `.npy`) files.
  - Added a logo. :)

# 0.1.1 Minor Fix

This release fixes the following bugs:

  - Issue with the packaged pre-compiled TensorFlow binaries that
    affected Linux platforms.
  - Learn API bug where the shared name of input iterators was being
    set incorrectly.

I also switched to using CircleCI for continuous integration, instead
of TravisCI.

# 0.1.0 First Official Release

This is the first official release of TensorFlow for Scala. The library
website will soon be updated with information about the functionality
supported by this API. Most of the main TensorFlow Python API
functionality is already supported.
