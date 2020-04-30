/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool LoopLikeOpInterface::isDefinedOutsideOfLoop(Value  value) {
      return getImpl()->isDefinedOutsideOfLoop(getOperation(), value);
  }
Region & LoopLikeOpInterface::getLoopBody() {
      return getImpl()->getLoopBody(getOperation());
  }
LogicalResult LoopLikeOpInterface::moveOutOfLoop(ArrayRef<Operation *> ops) {
      return getImpl()->moveOutOfLoop(getOperation(), ops);
  }
