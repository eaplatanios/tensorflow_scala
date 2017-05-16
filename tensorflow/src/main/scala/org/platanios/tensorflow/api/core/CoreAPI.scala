package org.platanios.tensorflow.api.core

import org.platanios.tensorflow.api
import org.platanios.tensorflow.api.core

/**
  * @author Emmanouil Antonios Platanios
  */
private[api] trait CoreAPI {
  type Graph = core.Graph
  val Graph = core.Graph

  val defaultGraph: core.Graph = api.defaultGraph

  type Indexer = core.Indexer
  type Index = core.Index
  type Slice = core.Slice

  val Indexer  = core.Indexer
  val Index    = core.Index
  val Slice    = core.Slice
  val NewAxis  = core.NewAxis
  val Ellipsis = core.Ellipsis

  type Session = core.Session
  val Session = core.Session

  type Shape = core.Shape
  val Shape = core.Shape

  type DeviceSpecification = core.DeviceSpecification
  val DeviceSpecification = core.DeviceSpecification

  type ShapeMismatchException = core.exception.ShapeMismatchException
  type GraphMismatchException = core.exception.GraphMismatchException
  type IllegalNameException = core.exception.IllegalNameException
  type InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
  type InvalidGraphElementException = core.exception.InvalidGraphElementException
  type InvalidShapeException = core.exception.InvalidShapeException
  type InvalidIndexerException = core.exception.InvalidIndexerException
  type InvalidDataTypeException = core.exception.InvalidDataTypeException
  type OpBuilderUsedException = core.exception.OpBuilderUsedException

  val ShapeMismatchException              = core.exception.ShapeMismatchException
  val GraphMismatchException              = core.exception.GraphMismatchException
  val IllegalNameException                = core.exception.IllegalNameException
  val InvalidDeviceSpecificationException = core.exception.InvalidDeviceSpecificationException
  val InvalidGraphElementException        = core.exception.InvalidGraphElementException
  val InvalidShapeException               = core.exception.InvalidShapeException
  val InvalidIndexerException             = core.exception.InvalidIndexerException
  val InvalidDataTypeException            = core.exception.InvalidDataTypeException
  val OpBuilderUsedException              = core.exception.OpBuilderUsedException
}
