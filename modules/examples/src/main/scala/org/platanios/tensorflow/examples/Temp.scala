//import org.platanios.tensorflow.api._
//
//object Example {
//  val inputs = tf.placeholder(FLOAT32, Shape(-1, 28, 28), "Inputs")
//  val labels = tf.placeholder(INT32, Shape(-1), "Labels")
//  val predictions = tf.createWith(nameScope = "Layer0") {
//    val weights = tf.variable("Weights", FLOAT32, Shape(28 * 28, 10), tf.ZerosInitializer)
//    val bias = tf.variable("Bias", FLOAT32, Shape(10), tf.ZerosInitializer)
//    inputs * weights + bias
//  }
//  val loss = tf.softmaxCrossEntropy(predictions, labels.cast(FLOAT32))
//  val trainOp = tf.train.AMSGrad().minimize(loss)
//
//  tf.cond(
//    tf.equal(tf.rank(predictions), tf.rank(targets) + 1),
//    () => Basic.expandDims(targets, -1),
//    () => targets)
//
//  val t6 = t5(2 :: 5, ---, 1) // Equivalent to numpy's `t5[2:5, ..., 1]`
//}
