package org.platanios.tensorflow.api.tensors.ops

import org.junit.Test
import org.scalatest.junit.JUnitSuite
import org.platanios.tensorflow.api._

class MathSuite extends JUnitSuite
{
  @Test def testArgmax() =
  {
    val rng = new util.Random(1337) // <- fixed seed (reproducability)
    for( run <- 0 to 1024 )
    {
      val shape = Shape apply Array.tabulate(rng.nextInt(4)+1){ _ => rng.nextInt(8)+1 }
      val tensor = Random.randomUniform(FLOAT64,shape)(
        -1e8: Tensor[FLOAT64],
        +1e8: Tensor[FLOAT64],
        Some(rng.nextInt)
      )
      val iMax = tensor.argmax(null,INT32)
      val idx = iMax.entriesIterator.map( i => i: Indexer ).toSeq
      val maxVal = tensor( idx.head, idx.tail: _* )
      assert( maxVal == tensor.max() )
    }
  }
}