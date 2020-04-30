/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

CallInterfaceCallable CallOpInterface::getCallableForCallee() {
      return getImpl()->getCallableForCallee(getOperation());
  }
Operation::operand_range CallOpInterface::getArgOperands() {
      return getImpl()->getArgOperands(getOperation());
  }
Operation * CallOpInterface::resolveCallable() {
      return getImpl()->resolveCallable(getOperation());
  }
Region * CallableOpInterface::getCallableRegion() {
      return getImpl()->getCallableRegion(getOperation());
  }
ArrayRef<Type> CallableOpInterface::getCallableResults() {
      return getImpl()->getCallableResults(getOperation());
  }
