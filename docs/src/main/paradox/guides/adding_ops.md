# Adding Support for New Ops

TensorFlow graphs are constructed by buildings ops that
receive tensors as input and produce tensors as output.
Internally, multiple *kernels* are registered for each op,
which are implementations of the op for different
architectures (e.g., CPU kernels, CUDA GPU kernels, etc.).
TensorFlow Scala offers the
@scaladoc[Op.Builder](org.platanios.tensorflow.api.ops.Op.Builder)
interface to allow users to create arbitrary ops that the
TensorFlow runtime supports.

For example, the implentation of `tf.add(x, y)` in
TensorFlow Scala looks like this:

@@snip [AddingOps.scala](/docs/src/main/scala/AddingOps.scala) { #add_op_example }
