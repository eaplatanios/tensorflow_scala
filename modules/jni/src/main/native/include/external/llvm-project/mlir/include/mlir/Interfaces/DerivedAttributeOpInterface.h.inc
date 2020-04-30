/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct DerivedAttributeOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual bool isDerivedAttribute(StringRef name) = 0;
    virtual DictionaryAttr materializeDerivedAttributes(Operation *tablegen_opaque_op) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    bool isDerivedAttribute(StringRef name) final {
      return ConcreteOp::isDerivedAttribute(name);
    }
    DictionaryAttr materializeDerivedAttributes(Operation *tablegen_opaque_op) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.materializeDerivedAttributes();
    }
  };
};
} // end namespace detail
class DerivedAttributeOpInterface : public OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits> {
public:
  using OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<DerivedAttributeOpInterface, detail::DerivedAttributeOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  bool isDerivedAttribute(StringRef name);
  DictionaryAttr materializeDerivedAttributes();
};
