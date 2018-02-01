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

  /** Dct method used for JPEG decoding. **/
  sealed trait DctMethod

  object DctMethod {
    case object IntegerFast extends DctMethod
    case object IntegerAccurate extends DctMethod
    case object SystemDefault extends DctMethod
  }

  /** $OpDocImageDecodeJpeg
    *
    * @group ImageOps
    * @param contents            0-D [[Output]] of type [[org.platanios.tensorflow.api.types.STRING]].
    *                            The JPEG-encoded image.
    * @param channels            Number of color channels for the decoded image. Defaults to 0.
    * @param ratio               Downscaling ratio. Defaults to 1.
    * @param fancyUpscaling      If true use a slower but nicer upscaling of the chroma planes (yuv420/422 only).
    *                            Defaults to true.
    * @param tryRecoverTruncated If true try to recover an image from truncated input. Defaults to false.
    * @param acceptableFraction  The minimum required fraction of lines before a truncated input is accepted.
    *                            Defaults to 1.
    * @param dctMethod           Specifies a hint about the algorithm used for decompression.
    *                            Defaults to [[DctMethod.SystemDefault]] which maps to a system-specific default.
    *                            The hint may be ignored (e.g., the internal jpeg library changes to a version that
    *                            does not have that specific option.)
    * @param name                Name for the created op.

    * @return 3-D [[Output]] of type [[org.platanios.tensorflow.api.types.UINT8]]
    *         with shape `[height, width, channels]`.
    */
  def decodeJpeg(
      contents: Output, channels: Int = 0, ratio: Int = 1, fancyUpscaling: Boolean = true,
      tryRecoverTruncated: Boolean = false, acceptableFraction: Float = 1,
      dctMethod: DctMethod = DctMethod.SystemDefault,
      name: String = "DecodeJpeg"): Output = {
    val dctMethodString = dctMethod match {
      case DctMethod.IntegerFast => "INTEGER_FAST"
      case DctMethod.IntegerAccurate => "INTEGER_ACCURATE"
      case DctMethod.SystemDefault => ""
    }
    Op.Builder(opType = "DecodeJpeg", name = name)
      .addInput(contents)
      .setAttribute("channels", channels)
      .setAttribute("ratio", ratio)
      .setAttribute("fancy_upscaling", fancyUpscaling)
      .setAttribute("try_recover_truncated", tryRecoverTruncated)
      .setAttribute("acceptable_fraction", acceptableFraction)
      .setAttribute("dct_method", dctMethodString)
      .build().outputs(0)
  }

  /** Format of the image tensor. **/
  sealed trait ImageFormat

  object ImageFormat {
    case object Grayscale extends ImageFormat
    case object RGB extends ImageFormat
    case object Default extends ImageFormat
  }

  /** Per pixel density unit. */
  sealed trait DensityUnit

  object DensityUnit {
    case object Inch extends DensityUnit
    case object Centimeter extends DensityUnit
  }

  /** $OpDocImageEncodeJpeg
    *
    * @group ImageOps
    * @param image              3-D [[Output]] of type [[org.platanios.tensorflow.api.types.UINT8]]
    *                           with shape `[height, width, channels]`.
    * @param format             Per pixel image format. One of [[ImageFormat.Default]], [[ImageFormat.Grayscale]],
    *                           [[ImageFormat.RGB]].
    * @param quality            Quality of the compression from 0 to 100 (higher is better and slower).
    *                           Defaults to `95`.
    * @param progressive        If true, create a JPEG that loads progressively (coarse to fine). Defaults to `false`.
    * @param optimizeSize       If true, spend CPU/RAM to reduce size with no quality change. Defaults to `false`.
    * @param chromaDownsampling See [[http://en.wikipedia.org/wiki/Chroma_subsampling]]. Defaults to `true`.
    * @param densityUnit        Unit used to specify `xDensity` and `yDensity`:
    *                           Pixels per inch [[DensityUnit.Inch]] or centimeter [[DensityUnit.Centimeter]].
    *                           Defaults to [[DensityUnit.Inch]].
    * @param xDensity           Horizontal pixels per density unit. Defaults to `300`.
    * @param yDensity           Vertical pixels per density unit. Defaults to `300`.
    * @param xmpMetadata        If not empty, embed this XMP metadata in the image header. Defaults to `""`.
    * @param name               Name for the created op.

    * @return 0-D [[Output]] of type [[org.platanios.tensorflow.api.types.STRING]] containing the JPEG-encoded image.
    */
  def encodeJpeg(
      image: Output, format: ImageFormat = ImageFormat.Default, quality: Int = 95, progressive: Boolean = false,
      optimizeSize: Boolean = false, chromaDownsampling: Boolean = true, densityUnit: DensityUnit = DensityUnit.Inch,
      xDensity: Int = 300, yDensity: Int = 300, xmpMetadata: String = "",
      name: String = "EncodeJpeg"): Output = {
    val formatAttribute = format match {
      case ImageFormat.Grayscale => "grayscale"
      case ImageFormat.RGB => "rgb"
      case ImageFormat.Default => ""
    }
    val densityUnitAttribute = densityUnit match {
      case DensityUnit.Inch => "in"
      case DensityUnit.Centimeter => "cm"
    }
    Op.Builder(opType = "EncodeJpeg", name = name)
      .addInput(image)
      .setAttribute("format", formatAttribute)
      .setAttribute("quality", quality)
      .setAttribute("progressive", progressive)
      .setAttribute("optimize_size", optimizeSize)
      .setAttribute("chroma_downsampling", chromaDownsampling)
      .setAttribute("density_unit", densityUnitAttribute)
      .setAttribute("x_density", xDensity)
      .setAttribute("y_density", yDensity)
      .setAttribute("xmp_metadata", xmpMetadata)
      .build().outputs(0)
  }
}

private[ops] object Image extends Image {
  /** @define OpDocImageExtractImagePatches
    *   The `extractImagePatches` op extracts `patches` from `images` and puts them in the `depth` output dimension.
    *
    * @define OpDocImageDecodeJpeg
    *   The `decodeJpeg` op decodes a JPEG-encoded image to a [[org.platanios.tensorflow.api.types.UINT8]] tensor.
    *
    *   The attribute `channels` indicates the desired number of color channels for the decoded image.
    *
    *   Accepted values are:
    *     0: Use the number of channels in the JPEG-encoded image.
    *     1: output a grayscale image.
    *     3: output an RGB image.
    *
    *   If needed, the JPEG-encoded image is transformed to match the requested number of color channels.
    *
    *   The attribute `ratio` allows downscaling the image by an integer factor during decoding.
    *   Allowed values are: 1, 2, 4, and 8.  This is much faster than downscaling the image later.
    *
    * @define OpDocImageEncodeJpeg
    *   The `encodeJpeg` op encodes an image tensor as JPEG.
    *
    *   `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
    *
    *   The attribute `format` can be used to override the color format of the encoded output.  Values can be:
    *     [[ImageFormat.Default]]: Use a default format based on the number of channels in the image.
    *     [[ImageFormat.Grayscale]]: Output a grayscale JPEG image.  The `channels` dimension of `image` must be 1.
    *     [[ImageFormat.RGB]]: Output an RGB JPEG image. The `channels` dimension of `image` must be 3.
    *
    *   If `format` is set to [[ImageFormat.Default]], a default format is picked in function of the number of channels
    *   in `image`:
    *     1: Output a grayscale image.
    *     3: Output an RGB image.
    */
  private[ops] trait Documentation
}
