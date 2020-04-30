/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Operation Interface Declarations                                           *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace detail {
struct OpAsmOpInterfaceInterfaceTraits {
  class Concept {
  public:
    virtual ~Concept() = default;
    virtual void getAsmResultNames(Operation *tablegen_opaque_op, OpAsmSetValueNameFn setNameFn) = 0;
  };
  template<typename ConcreteOp>
  class Model : public Concept {
public:
    void getAsmResultNames(Operation *tablegen_opaque_op, OpAsmSetValueNameFn setNameFn) final {
      auto op = llvm::cast<ConcreteOp>(tablegen_opaque_op);
      (void)op;
      return op.getAsmResultNames(setNameFn);
    }
  };
};
} // end namespace detail
class OpAsmOpInterface : public OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits> {
public:
  using OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
  };
  void getAsmResultNames(OpAsmSetValueNameFn setNameFn);
};
