/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool DerivedAttributeOpInterface::isDerivedAttribute(StringRef name) {
      return getImpl()->isDerivedAttribute(name);
  }
DictionaryAttr DerivedAttributeOpInterface::materializeDerivedAttributes() {
      return getImpl()->materializeDerivedAttributes(getOperation());
  }
