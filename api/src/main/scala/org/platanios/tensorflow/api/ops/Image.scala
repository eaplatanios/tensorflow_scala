/* Copyright 2017, Emmanouil Antonios Platanios. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package org.platanios.tensorflow.api.ops

/** Contains functions for constructing ops related to processing image data.
  *
  * @author Emmanouil Antonios Platanios
  * @author Sören Brunk
  */
private[ops] trait Image {
  /** $OpDocImageExtractImagePatches
    *
    * @group ImageOps
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
  def extractImagePatches(
      images: Output, slidingWindowSizes: Seq[Int], strides: Seq[Int], rates: Seq[Int],
      name: String = "ExtractImagePatches"): Output = {
    Op.Builder(opType = "ExtractImagePatches", name = name)
        .addInput(images)
        .setAttribute("ksizes", slidingWindowSizes.map(_.toLong).toArray)
        .setAttribute("strides", strides.map(_.toLong).toArray)
        .setAttribute("rates", rates.map(_.toLong).toArray)
        .build().outputs(0)
  }

  /** $OpDocImageDecodeJpeg
    *
    * @group ImageOps
    * @param contents              0-D [[Output]] of type [[org.platanios.tensorflow.api.types.STRING]].
    *                              The JPEG-encoded image.
    * @param channels              Number of color channels for the decoded image. Defaults to 0.
    * @param ratio                 Downscaling ratio. Defaults to 1.
    * @param fancy_upscaling       If true use a slower but nicer upscaling of the chroma planes (yuv420/422 only).
    *                              Defaults to true.
    * @param try_recover_truncated If true try to recover an image from truncated input. Defaults to false.
    * @param acceptable_fraction   The minimum required fraction of lines before a truncated input is accepted.
    *                              Defaults to 1.
    * @param dct_method            String specifying a hint about the algorithm used for decompression.
    *                              Defaults to "" which maps to a system-specific default.
    *                              Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"].
    *                              The hint may be ignored (e.g., the internal jpeg library changes to a version that
    *                              does not have that specific option.)
    * @param name                  Name for the created op.

    * @return 3-D [[Output]] of type [[org.platanios.tensorflow.api.types.UINT8]]
    *         with shape `[height, width, channels]`.
    */
  def decodeJpeg(
      contents: Output, channels: Int = 0, ratio: Int = 1, fancy_upscaling: Boolean = true,
      try_recover_truncated: Boolean = false, acceptable_fraction: Float = 1, dct_method: String = "",
      name: String = "DecodeJpeg"): Output = {
    Op.Builder(opType = "DecodeJpeg", name = name)
      .addInput(contents)
      .setAttribute("channels", channels)
      .setAttribute("ratio", ratio)
      .setAttribute("fancy_upscaling", fancy_upscaling)
      .setAttribute("try_recover_truncated", try_recover_truncated)
      .setAttribute("acceptable_fraction", acceptable_fraction)
      .setAttribute("dct_method", dct_method)
      .build().outputs(0)
  }

  /** $OpDocImageEncodeJpeg
    *
    * @group ImageOps
    * @param image              3-D [[Output]] of type [[org.platanios.tensorflow.api.types.UINT8]]
    *                           with shape `[height, width, channels]`.
    * @param format             Per pixel image format. One of `"", "grayscale", "rgb"`. Defaults to `""`.
    * @param quality            Quality of the compression from 0 to 100 (higher is better and slower).
    *                           Defaults to `95`.
    * @param progressive        If True, create a JPEG that loads progressively (coarse to fine). Defaults to `False`.
    * @param optimizeSize       If True, spend CPU/RAM to reduce size with no quality change. Defaults to `false`.
    * @param chromaDownsampling See [[http://en.wikipedia.org/wiki/Chroma_subsampling]]. Defaults to `true`.
    * @param densityUnit        Unit used to specify `x_density` and `y_density`:
    *                           pixels per inch (`"in"`) or centimeter (`"cm"`). Defaults to `"in"`.
    * @param xDensity           Horizontal pixels per density unit. Defaults to `300`.
    * @param yDensity           Vertical pixels per density unit. Defaults to `300`.
    * @param xmpMetadata        If not empty, embed this XMP metadata in the image header. Defaults to `""`.
    * @param name               Name for the created op.

    * @return 0-D [[Output]] of type [[org.platanios.tensorflow.api.types.STRING]] containing the JPEG-encoded image.
    */
  def encodeJpeg(
      image: Output, format: String = "", quality: Int = 95, progressive: Boolean = false,
      optimizeSize: Boolean = false, chromaDownsampling: Boolean = true, densityUnit: String = "in",
      xDensity: Int = 300, yDensity: Int = 300, xmpMetadata: String = "",
      name: String = "EncodeJpeg"): Output = {
    Op.Builder(opType = "EncodeJpeg", name = name)
      .addInput(image)
      .setAttribute("format", format)
      .setAttribute("quality", quality)
      .setAttribute("progressive", progressive)
      .setAttribute("optimize_size", optimizeSize)
      .setAttribute("chroma_downsampling", chromaDownsampling)
      .setAttribute("density_unit", densityUnit)
      .setAttribute("x_density", xDensity)
      .setAttribute("y_density", yDensity)
      .setAttribute("xmp_metadata", xmpMetadata)
      .build().outputs(0)
  }
}

private[ops] object Image extends Image {
  /** @define OpDocImageExtractImagePatches
    *   The `extractImagePatches` op extracts `patches` from `images` and puts them in the `depth` output dimension.
    * @define OpDocDecodeJpeg
    *   Decodes a JPEG-encoded image to a [[org.platanios.tensorflow.api.types.UINT8]] tensor.
    *
    *   The attr `channels` indicates the desired number of color channels for the decoded image.
    *
    *   Accepted values are:
    *     0: Use the number of channels in the JPEG-encoded image.
    *     1: output a grayscale image.
    *     3: output an RGB image.
    *
    *   If needed, the JPEG-encoded image is transformed to match the requested number of color channels.
    *
    *   The attr `ratio` allows downscaling the image by an integer factor during decoding.
    *   Allowed values are: 1, 2, 4, and 8.  This is much faster than downscaling the image later.
    * @define OpDocEncodeJpeg
    *   JPEG-encode an image.
    *
    *   `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
    *
    *   The attr `format` can be used to override the color format of the encoded output.  Values can be:
    *     `''`: Use a default format based on the number of channels in the image.
    *     `grayscale`: Output a grayscale JPEG image.  The `channels` dimension of `image` must be 1.
    *     `rgb`: Output an RGB JPEG image. The `channels` dimension of `image` must be 3.
    *
    *   If `format` is not specified or is the empty string, a default format is picked
    *   in function of the number of channels in `image`:
    *     1: Output a grayscale image.
    *     3: Output an RGB image.
    */
  private[ops] trait Documentation
}
