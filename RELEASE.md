# 0.5.1

This release introduces support for TensorFlow 2.2 and
Scala 2.13 and drops support for Scala 2.11. The
distributed precompiled binaries for this version will only
work with CUDA 10.1 on Linux. Finally, this release also
brings improved support for implicit derivations in some
cases where case classes over tensors are used.

# 0.5.0

This release introduces support for TensorFlow 2.0.

# 0.4.2

Minor release that:
  - Added support for a TF records reader.
  - Fixed a bug related to reading and writing NPY files.

# 0.4.1 Fixed Precompiled TF Binaries and Added Some New Features

Fixed the precompiled TensorFlow binaries, and also added the following
new features:

  - `io` module:
    - Added support for a new `TFRecordWriter`.
  - `ops` module:
    - Added a new ops namespace, `sparse`, that includes all sparse ops.
    - Added support for `sparse.reorder` and `sparse.merge`.
    - Added support for parsing TF records.
    - `data` module:
      - Added support for `Dataset.shuffleAndRepeat`.
    - `optimizers` module:
      - Added support for the Adafactor optimizer.
      - Renamed `SqrtDecay` to `RSqrtDecay` which is more appropriate.
    - `math` module:
      - Added support for `batchGather`.
      - Added support for bitwise ops.
    - `rnn` module:
      - Simplified the attention mechanisms functionality so that it is
        now not required to tile memory tensors for beam search outside
        the beam search decoder.
    - Moved the `seq2seq` module to a separate repository (that of
      [Symphony Machine Translation](https://github.com/eaplatanios/symphony-mt)).

# 0.4.0 More Static Data Types

This is a major release with a lot of new features related to static
types for tensors and ops. The graph construction API is now
statically-typed, thus enabling much better type safety than before.

Tensors and outputs are now statically-typed and the types used are the
Scala types that correspond to the tensors' TensorFlow data types. For
example:

```scala
val t1 = Tensor(0.5, 1) // The inferred type is Tensor[Double].
val t2 = Tensor(1, 2)   // The inferred type is Tensor[Int].
val t3 = t1 + t2        // The inferred type is Tensor[Double].
val t4 = t3.isNaN       // The inferred type is Tensor[Boolean].
val t5 = t3.any()       // Fails at compile-time because `any()` is only
                        // supported for Tensor[Boolean].
```

A similar situation now applies to `Output`s. `Op`s are also typed and
so is the auto-differentiation implementation.

This resulted in major simplifications in the data pipeline and the high
level learn API. Datasets and dataset iterators do not "carry" `T`, `V`,
`D`, and `S` types with them now, but rather just the type of the
elements they contain/produce.

A new type trait called `TF` is also introduced that denotes supported
Scala types in TensorFlow (e.g., `TF[Int]` and `TF[Float]`). Similarly,
some more type traits are introduced to denote type constraints for
various ops (e.g., `IsIntOrUInt[Int]`, `IsIntOrUInt[Long]`,
`IsFloatOrDouble[Float]`, etc.). These type traits are powered by a
general implementation of union types for Scala.

Other new features include:

  - `data` module:
    - Added support for the `mapAndBatch` transformation.

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
    - Added support for batch normalization.
  - Added support for `tf.logSigmoid` and `tf.lrn`.
  - Added support for the following new metrics:
    - Grouped precision.
    - Precision-at-k.
  - `data` module:
    - Added support for loading the extreme classification repository
      datasets (i.e., `data.XCLoader`).
    - Added support for randomly splitting datasets.

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
