package org.platanios.tensorflow.api.core.client

import org.platanios.tensorflow.api.core.client

/**
  * @author Emmanouil Antonios Platanios
  */
trait ClientAPI {
  type Session = client.Session
  val Session = client.Session

  type Executable = client.Executable
  type Feedable[T] = client.Feedable[T]
  type Fetchable[+T] = client.Fetchable[T]
}
