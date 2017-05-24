package org.platanios.tensorflow.api.ops

/** Contains functions for constructing ops related to processing image data.
  *
  * @author Emmanouil Antonios Platanios
  */
trait Image {
  /** Creates an op that extracts `patches` from `images` and puts them in the `depth` output dimension.
    *
    * @param  images             `4`-dimensional tensor with shape `[batch, inputRows, inputCols, depth]`.
    * @param  slidingWindowSizes Sequence containing the size of the sliding window for each dimension of `images`.
    * @param  strides            Sequence where each element represents how far the centers of two consecutive patches
    *                            are in the images.
    * @param  rates              Sequence where each element represents how far two consecutive patch samples are in the
    *                            input. This induces a behavior which is equivalent to extracting patches with
    *                            `effectivePatchSizes = patchSizes + (patchSizes - 1) * (rates - 1)`, followed by
    *                            subsampling them spatially by a factor of `rates`.
    * @param  name               Name for the created op.
    * @return Created op output, which is a `4`-dimensional tensor with shape
    *         `[batch, outputRows, outputCols, slidingWindowSizes(1) * slidingWindowSizes(2) * depth]`.
    */
  @throws[IllegalArgumentException]
  def extractImagePatches(
      images: Op.Output, slidingWindowSizes: Seq[Int], strides: Seq[Int], rates: Seq[Int],
      name: String = "ExtractImagePatches"): Op.Output = {
    if (images.rank != 4 && images.rank != -1)
      throw new IllegalArgumentException("'images' must have rank equal to 4.")
    if (slidingWindowSizes.length != 4)
      throw new IllegalArgumentException("'slidingWindowSizes' must have length equal to 4.")
    if (strides.length != 4)
      throw new IllegalArgumentException("'strides' must have length equal to 4.")
    if (rates.length != 4)
      throw new IllegalArgumentException("'rates' must have length equal to 4.")
    Op.Builder(opType = "ExtractImagePatches", name = name)
        .addInput(images)
        .setAttribute("ksizes", slidingWindowSizes.map(_.toLong).toArray)
        .setAttribute("strides", strides.map(_.toLong).toArray)
        .setAttribute("rates", rates.map(_.toLong).toArray)
        .build().outputs(0)
  }
}

object Image extends Image
