package org.platanios.tensorflow.data

import java.io.IOException
import java.net.URL
import java.nio.file.{Files, Path}

import com.typesafe.scalalogging.Logger

/**
  * @author Emmanouil Antonios Platanios
  */
trait Loader {
  protected val logger: Logger

  def maybeDownload(path: Path, url: String, bufferSize: Int = 8192): Boolean = {
    if (Files.exists(path)) {
      false
    } else {
      try {
        logger.info(s"Downloading file '$url'.")
        Files.createDirectories(path.getParent)
        // TODO: [DATA] Add progress bar.
        val inputStream = new URL(url).openStream()
        val outputStream = Files.newOutputStream(path)
        val buffer = new Array[Byte](bufferSize)
        Stream.continually(inputStream.read(buffer)).takeWhile(_ != -1).foreach(outputStream.write(buffer, 0, _))
        outputStream.close()
        logger.info(s"Downloaded file '$url'.")
        true
      } catch {
        case e: IOException =>
          logger.error(s"Could not download file '$url'", e)
          throw e
      }
    }
  }
}
