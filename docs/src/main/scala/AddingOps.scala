import org.platanios.tensorflow.api._

object AddingOps {
  // #add_op_example
  def add[T: TF : IsNotQuantized](
      x: Output[T],
      y: Output[T],
      name: String = "Add"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[T]), Output[T]](
      opType = "Add",
      name = name,
      input = (x, y)
    ).setGradientFn(addGradient(_, _)(TF[T], IsNotQuantized[T]))
        .build().output
  }

  protected def addGradient[T: TF : IsNotQuantized](
      op: Op[(Output[T], Output[T]), Output[T]],
      outputGradient: Output[T]
  ): (Output[T], Output[T]) = {
    val xShape = tf.shape(op.input._1)
    val yShape = tf.shape(op.input._2)
    val (rx, ry) = tf.broadcastGradientArguments(xShape, yShape)
    (tf.reshape(tf.sum(outputGradient, rx), xShape),
        tf.reshape(tf.sum(outputGradient, ry), yShape))
  }
  // #add_op_example
}
