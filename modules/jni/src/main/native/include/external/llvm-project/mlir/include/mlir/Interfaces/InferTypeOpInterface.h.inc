/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct InferShapedTypeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual LogicalResult inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) = 0;
    virtual LogicalResult reifyReturnTypeShapes(Operation *tablegen_opaque_op, OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    LogicalResult inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) final {
      return ConcreteOp::inferReturnTypeComponents(context, location, operands, attributes, regions, inferredReturnShapes);
    }
    LogicalResult reifyReturnTypeShapes(Operation *tablegen_opaque_op, OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.reifyReturnTypeShapes(builder, reifiedReturnShapes);
    }
  };
};
} // end namespace detail
class InferShapedTypeOpInterface : public OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits> {
public:
  using OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<InferShapedTypeOpInterface, detail::InferShapedTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  LogicalResult reifyReturnTypeShapes(OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes) {
 return failure();   }
  };
  LogicalResult inferReturnTypeComponents(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);
  LogicalResult reifyReturnTypeShapes(OpBuilder& builder, SmallVectorImpl<Value>& reifiedReturnShapes);
};
namespace detail {
struct InferTypeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) = 0;
    virtual bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) final {
      return ConcreteOp::inferReturnTypes(context, location, operands, attributes, regions, inferredReturnTypes);
    }
    bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) final {

        return ConcreteOp::isCompatibleReturnTypes(lhs, rhs);
      
    }
  };
};
} // end namespace detail
class InferTypeOpInterface : public OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits> {
public:
  using OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<InferTypeOpInterface, detail::InferTypeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  static bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs) {

        /// Returns whether two arrays are equal as strongest check for
        /// compatibility by default.
        return lhs == rhs;
        }
  static LogicalResult verifyTrait(Operation* op) {

    return detail::verifyInferredResultTypes(op);
  
  }
  };
  LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes);
  bool isCompatibleReturnTypes(ArrayRef<Type> lhs, ArrayRef<Type> rhs);
};
