/* Copyright 2017-18, Emmanouil Antonios Platanios. All Rights Reserved.
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

import org.platanios.tensorflow.api.ops.Image._
import org.platanios.tensorflow.api.types._

/** Contains functions for constructing ops related to processing image data.
  *
  * @author Emmanouil Antonios Platanios, Sören Brunk
  */
trait Image {
  // TODO: [OPS] Add missing ops.
  // TODO: [OPS] Add implicits.
  // TODO: [GRADIENTS] Add gradients.

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
  def extractImagePatches[T: IsReal](
      images: Output[T],
      slidingWindowSizes: Seq[Int],
      strides: Seq[Int],
      rates: Seq[Int],
      name: String = "ExtractImagePatches"
  ): Output[T] = {
    Op.Builder[Output[T], Output[T]](
      opType = "ExtractImagePatches",
      name = name,
      input = images
    ).setAttribute("ksizes", slidingWindowSizes.map(_.toLong).toArray)
        .setAttribute("strides", strides.map(_.toLong).toArray)
        .setAttribute("rates", rates.map(_.toLong).toArray)
        .build().output
  }

  /** $OpDocImageDecodeJpeg
    *
    * @group ImageOps
    * @param  contents            0-D [[Output]] of containing the JPEG-encoded image.
    * @param  numChannels         Number of color channels for the decoded image. Defaults to 0.
    * @param  ratio               Downscaling ratio.
    * @param  fancyUpscaling      If `true` use a slower but nicer upscaling of the chroma planes (yuv420/422 only).
    * @param  tryRecoverTruncated If true try to recover an image from truncated input. Defaults to false.
    * @param  acceptableFraction  The minimum required fraction of lines before a truncated input is accepted.
    * @param  dctMethod           Specifies a hint about the algorithm to use for decompression. Defaults to
    *                             [[DCTMethod.SystemDefault]] which maps to a system-specific default. The hint may be
    *                             ignored (e.g., the internal JPEG library changes to a version that does not have that
    *                             specific option).
    * @param  name                Name for the created op.
    * @return 3-D tensor with shape `[height, width, numChannels]`.
    */
  def decodeJpeg(
      contents: Output[String],
      numChannels: Int = 0,
      ratio: Int = 1,
      fancyUpscaling: Boolean = true,
      tryRecoverTruncated: Boolean = false,
      acceptableFraction: Float = 1,
      dctMethod: DCTMethod = DCTMethod.SystemDefault,
      name: String = "DecodeJpeg"
  ): Output[UByte] = {
    Op.Builder[Output[String], Output[UByte]](
      opType = "DecodeJpeg",
      name = name,
      input = contents
    ).setAttribute("channels", numChannels)
        .setAttribute("ratio", ratio)
        .setAttribute("fancy_upscaling", fancyUpscaling)
        .setAttribute("try_recover_truncated", tryRecoverTruncated)
        .setAttribute("acceptable_fraction", acceptableFraction)
        .setAttribute("dct_method", dctMethod.toTFString)
        .build().output
  }

  /** $OpDocImageEncodeJpeg
    *
    * @group ImageOps
    * @param  image              3-D tensor with shape `[height, width, numChannels]`.
    * @param  format             Per-pixel image format.
    * @param  quality            Quality of the compression from 0 to 100 (higher is better and slower).
    * @param  progressive        If `true`, create a JPEG that loads progressively (coarse to fine).
    * @param  optimizeSize       If `true`, spend CPU/RAM to reduce size with no quality change.
    * @param  chromaDownsampling Boolean value indicating whether or not to perform chroma downsampling. For details on
    *                            this method, please refer to
    *                            [http://en.wikipedia.org/wiki/Chroma_subsampling](http://en.wikipedia.org/wiki/Chroma_subsampling).
    * @param  densityUnit        Unit used to specify `xDensity` and `yDensity`:
    *                            Pixels per inch [[DensityUnit.Inch]] or centimeter [[DensityUnit.Centimeter]].
    * @param  xDensity           Horizontal pixels per density unit.
    * @param  yDensity           Vertical pixels per density unit.
    * @param  xmpMetadata        If not empty, embed this XMP metadata in the image header. Defaults to `""`.
    * @param  name               Name for the created op.
    * @return 0-D tensor containing the JPEG-encoded image.
    */
  def encodeJpeg(
      image: Output[UByte],
      format: ImageFormat = ImageFormat.Default,
      quality: Int = 95,
      progressive: Boolean = false,
      optimizeSize: Boolean = false,
      chromaDownsampling: Boolean = true,
      densityUnit: DensityUnit = DensityUnit.Inch,
      xDensity: Int = 300,
      yDensity: Int = 300,
      xmpMetadata: String = "",
      name: String = "EncodeJpeg"
  ): Output[String] = {
    Op.Builder[Output[UByte], Output[String]](
      opType = "EncodeJpeg",
      name = name,
      input = image
    ).setAttribute("format", format.toTFString)
        .setAttribute("quality", quality)
        .setAttribute("progressive", progressive)
        .setAttribute("optimize_size", optimizeSize)
        .setAttribute("chroma_downsampling", chromaDownsampling)
        .setAttribute("density_unit", densityUnit.toTFString)
        .setAttribute("x_density", xDensity)
        .setAttribute("y_density", yDensity)
        .setAttribute("xmp_metadata", xmpMetadata)
        .build().output
  }

  /** $OpDocImageResizeArea
    *
    * @group ImageOps
    * @param  images       4-D tensor with shape `[batch, height, width, channels]`.
    * @param  size         1-D tensor of 2 elements: `newHeight, newWidth`. The new size for the images.
    * @param  alignCorners If true, rescale input by (new_height - 1) / (height - 1), which exactly aligns the 4 corners
    *                      of images and resized images. If false, rescale by new_height / height.
    *                      Treat the width dimension similarly. Defaults to `false`.
    * @param name          Name for the created op.
    * @return 4-D tensor with shape `[batch, newHeight, newWidth, channels]`.
    */
  def resizeArea[T: IsReal](
      images: Output[T],
      size: Output[Int],
      alignCorners: Boolean = false,
      name: String = "ResizeArea"
  ): Output[Float] = {
    Op.Builder[(Output[T], Output[Int]), Output[Float]](
      opType = "ResizeArea",
      name = name,
      input = (images, size)
    ).setAttribute("align_corners", alignCorners)
        .build().output
  }

  /** $OpDocImageResizeBilinear
    *
    * @group ImageOps
    * @param  images       4-D tensor with shape `[batch, height, width, channels]`.
    * @param  size         1-D tensor of 2 elements: `newHeight, newWidth`. The new size for the images.
    * @param  alignCorners If true, rescale input by (new_height - 1) / (height - 1), which exactly aligns the 4 corners
    *                      of images and resized images. If false, rescale by new_height / height.
    *                      Treat the width dimension similarly. Defaults to `false`.
    * @param name          Name for the created op.
    * @return 4-D tensor with shape `[batch, newHeight, newWidth, channels]`.
    */
  def resizeBilinear[T: IsReal](
      images: Output[T],
      size: Output[Int],
      alignCorners: Boolean = false,
      name: String = "ResizeBilinear"
  ): Output[Float] = {
    Op.Builder[(Output[T], Output[Int]), Output[Float]](
      opType = "ResizeBilinear",
      name = name,
      input = (images, size)
    ).setAttribute("align_corners", alignCorners)
        .build().output
  }

  /** $OpDocImageResizeBicubic
    *
    * @group ImageOps
    * @param  images       4-D tensor with shape `[batch, height, width, channels]`.
    * @param  size         1-D tensor of 2 elements: `newHeight, newWidth`. The new size for the images.
    * @param  alignCorners If true, rescale input by (new_height - 1) / (height - 1), which exactly aligns the 4 corners
    *                      of images and resized images. If false, rescale by new_height / height.
    *                      Treat the width dimension similarly. Defaults to `false`.
    * @param name          Name for the created op.
    * @return 4-D tensor with shape `[batch, newHeight, newWidth, channels]`.
    */
  def resizeBicubic[T: IsReal](
      images: Output[T],
      size: Output[Int],
      alignCorners: Boolean = false,
      name: String = "ResizeBicubic"
  ): Output[Float] = {
    Op.Builder[(Output[T], Output[Int]), Output[Float]](
      opType = "ResizeBicubic",
      name = name,
      input = (images, size)
    ).setAttribute("align_corners", alignCorners)
        .build().output
  }

  /** $OpDocImageResizeNearestNeighbor
    *
    * @group ImageOps
    * @param  images       4-D tensor with shape `[batch, height, width, channels]`.
    * @param  size         1-D tensor of 2 elements: `newHeight, newWidth`. The new size for the images.
    * @param  alignCorners If true, rescale input by (new_height - 1) / (height - 1), which exactly aligns the 4 corners
    *                      of images and resized images. If false, rescale by new_height / height.
    *                      Treat the width dimension similarly. Defaults to `false`.
    * @param name          Name for the created op.
    * @return 4-D tensor with shape `[batch, newHeight, newWidth, channels]`.
    */
  def resizeNearestNeighbor[T: IsReal](
      images: Output[T],
      size: Output[Int],
      alignCorners: Boolean = false,
      name: String = "ResizeNearestNeighbor"
  ): Output[T] = {
    Op.Builder[(Output[T], Output[Int]), Output[T]](
      opType = "ResizeNearestNeighbor",
      name = name,
      input = (images, size)
    ).setAttribute("align_corners", alignCorners)
        .build().output
  }
}

object Image extends Image {
  /** Image tensor format. **/
  sealed trait ImageFormat {
    def toTFString: String
  }

  object ImageFormat {
    case object Grayscale extends ImageFormat {
      override def toTFString: String = "grayscale"
    }

    case object RGB extends ImageFormat {
      override def toTFString: String = "rgb"
    }

    case object Default extends ImageFormat {
      override def toTFString: String = ""
    }
  }

  /** Per-pixel density unit. */
  sealed trait DensityUnit {
    def toTFString: String
  }

  object DensityUnit {
    case object Inch extends DensityUnit {
      override def toTFString: String = "in"
    }

    case object Centimeter extends DensityUnit {
      override def toTFString: String = "cm"
    }
  }

  /** DCT method used for JPEG decoding. **/
  sealed trait DCTMethod {
    def toTFString: String
  }

  object DCTMethod {
    case object IntegerFast extends DCTMethod {
      override def toTFString: String = "INTEGER_FAST"
    }

    case object IntegerAccurate extends DCTMethod {
      override def toTFString: String = "INTEGER_ACCURATE"
    }

    case object SystemDefault extends DCTMethod {
      override def toTFString: String = ""
    }
  }

  /** @define OpDocImageExtractImagePatches
    *   The `extractImagePatches` op extracts `patches` from `images` and puts them in the `depth` output dimension.
    *
    * @define OpDocImageDecodeJpeg
    *   The `decodeJpeg` op decodes a JPEG-encoded image to a [[UINT8]] tensor.
    *
    *   The attribute `numChannels` indicates the desired number of color channels for the decoded image. Accepted
    *   values are:
    *     `0`: Use the number of channels in the JPEG-encoded image.
    *     `1`: output a grayscale image.
    *     `3`: output an RGB image.
    *
    *   If needed, the JPEG-encoded image is transformed to match the requested number of color channels.
    *
    *   The attribute `ratio` allows downscaling the image by an integer factor during decoding.
    *   Allowed values are: `1`, `2`, `4`, and `8`. This is much faster than downscaling the image later.
    *
    * @define OpDocImageEncodeJpeg
    *   The `encodeJpeg` op encodes an image tensor as a JPEG image.
    *
    *   `image` is a 3-D [[UINT8]] Tensor of shape `[height, width, numChannels]`.
    *
    *   The attribute `format` can be used to override the color format of the encoded output. Accepted values are:
    *     [[ImageFormat.Default]]: Use a default format based on the number of channels in the image.
    *     [[ImageFormat.Grayscale]]: Output a grayscale JPEG image.  The `channels` dimension of `image` must be 1.
    *     [[ImageFormat.RGB]]: Output an RGB JPEG image. The `channels` dimension of `image` must be 3.
    *
    *   If `format` is set to [[ImageFormat.Default]], a default format is picked as a function of the number of
    *   channels in the `image` tensor:
    *     1: Output a grayscale image.
    *     3: Output an RGB image.
    *
    * @define OpDocImageResizeArea
    *   The `resizeArea` op resizes `images` to `size` using area interpolation.
    *
    *   Input images can be of different types but output images are always float.
    *
    *   Each output pixel is computed by first transforming the pixel's footprint into the input tensor and then
    *   averaging the pixels that intersect the footprint. An input pixel's contribution to the average is weighted by
    *   the fraction of its area that intersects the footprint. This is the same as OpenCV's INTER_AREA.
    *
    * @define OpDocImageResizeBilinear
    *   The `resizeBilinear` op resizes `images` to `size` using bilinear interpolation.
    *
    *   Input images can be of different types but output images are always float.
    *
    * @define OpDocImageResizeBicubic
    *   The `resizeBicubic` op resizes `images` to `size` using bicubic interpolation.
    *
    *   Input images can be of different types but output images are always float.
    *
    * @define OpDocImageResizeNearestNeighbor
    *   The `resizeNearestNeighbor` op resizes `images` to `size` using nearest neighbor interpolation.
    */
  private[ops] trait Documentation
}
