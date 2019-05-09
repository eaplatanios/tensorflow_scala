import org.platanios.tensorflow.api._

import scala.language.postfixOps

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

  // #tensor_shape_examples
  val t0 = Tensor.ones[Int](Shape())     // Creates a scalar equal to the value 1
  val t1 = Tensor.ones[Int](Shape(10))   // Creates a vector with 10 elements, all of which are equal to 1
  val t2 = Tensor.ones[Int](Shape(5, 2)) // Creates a matrix with 5 rows with 2 columns

  // You can also create tensors in the following way:
  val t3 = Tensor(2.0, 5.6)                                 // Creates a vector that contains the numbers 2.0 and 5.6
  val t4 = Tensor(Tensor(1.2f, -8.4f), Tensor(-2.3f, 0.4f)) // Creates a matrix with 2 rows and 2 columns
  // #tensor_shape_examples

  // #tensor_inspect_shape_example
  t4.shape // Returns the value Shape(2, 2)
  // #tensor_inspect_shape_example

  // #tensor_inspect_rank_example
  t4.rank // Returns the value 2
  // #tensor_inspect_rank_example

  // #tensor_indexer_examples
  val t = Tensor.zeros[Float](Shape(4, 2, 3, 8))
  t(::, ::, 1, ::)            // Tensor with shape [4, 2, 1, 8]
  t(1 :: -2, ---, 2)          // Tensor with shape [1, 2, 3, 1]
  t(---)                      // Tensor with shape [4, 2, 3, 8]
  t(1 :: -2, ---, NewAxis, 2) // Tensor with shape [1, 2, 3, 1, 1]
  t(1 ::, ---, NewAxis, 2)    // Tensor with shape [3, 2, 3, 1, 1]
  // #tensor_indexer_examples
}
