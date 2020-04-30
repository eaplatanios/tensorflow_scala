/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Definitions                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

LogicalResult InferShapedTypeOpInterface::inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
      return getImpl()->inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
  }
LogicalResult InferShapedTypeOpInterface::reifyReturnTypeShapes(OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {
      return getImpl()->reifyReturnTypeShapes(getOperation(), builder, reifiedReturnShapes);
  }
LogicalResult InferTypeOpInterface::inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
      return getImpl()->inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
  }
bool InferTypeOpInterface::isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {
      return getImpl()->isCompatibleReturnTypes(lhs, rhs);
  }
