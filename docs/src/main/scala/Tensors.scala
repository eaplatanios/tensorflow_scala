import org.platanios.tensorflow.api._

object Tensors {
  // #zeros_tensor_example
  val tensor = Tensor.zeros[Int](Shape(2, 5))
  // #zeros_tensor_example

  // #tensor_summarize_example
  tensor.summarize()
  // Prints the following:
  //   Tensor[Int, [2, 5]]
  //   [[0, 0, 0, 0, 0],
  //    [0, 0, 0, 0, 0]]
  // #tensor_summarize_example

  // #tensor_creation_examples
  val a = Tensor[Int](1, 2)                  // Creates a Tensor[Int] with shape [2]
  val b = Tensor[Long](1L, 2)                // Creates a Tensor[Long] with shape [2]
  val c = Tensor[Float](3.0f)                // Creates a Tensor[Float] with shape [1]
  val d = Tensor[Double](-4.0)               // Creates a Tensor[Double] with shape [1]
  val e = Tensor.empty[Int]                  // Creates an empty Tensor[Int] with shape [0]
  val z = Tensor.zeros[Float](Shape(5, 2))   // Creates a zeros Tensor[Float] with shape [5, 2]
  val r = Tensor.randn(Double, Shape(10, 3)) // Creates a Tensor[Double] with shape [10, 3] and
                                             // elements drawn from the standard Normal distribution.
  // #tensor_creation_examples

  // #tensor_cast_examples
  val floatTensor = Tensor[Float](1, 2, 3) // Floating point vector containing the elements: 1.0f, 2.0f, and 3.0f.
  floatTensor.toInt                        // Integer vector containing the elements: 1, 2, and 3.
  floatTensor.castTo[Int]                  // Integer vector containing the elements: 1, 2, and 3.
  // #tensor_cast_examples

  // #tensor_datatype_example
  floatTensor.dataType // Returns FLOAT32
  // #tensor_datatype_example
}
