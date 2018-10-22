import org.platanios.tensorflow.api._

object Index {
  // #tensors_example
  val t1 = Tensor(1.2, 4.5)
  val t2 = Tensor(-0.2, 1.1)
  t1 + t2 == Tensor(1.0, 5.6)
  // #tensors_example
}
