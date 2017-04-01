package org.platanios.tensorflow.jni

import ch.jodersky.jni.nativeLoader

/**
  * @author Emmanouil Antonios Platanios
  */
@nativeLoader("tensorflow_jni")
object TensorFlow {
  @native def version: String
}
