/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

class AddOp;
class BroadcastOp;
class ConcatOp;
class ConstantOp;
class DebugPrintOp;
class FromExtentTensorOp;
class JoinOp;
class MulOp;
class ReduceOp;
class ShapeOfOp;
class SplitAtOp;
class ToExtentTensorOp;
class YieldOp;

//===----------------------------------------------------------------------===//
// shape::AddOp declarations
//===----------------------------------------------------------------------===//

class AddOpOperandAdaptor {
public:
  AddOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lhs();
  Value rhs();

private:
  ArrayRef<Value> tblgen_operands;
};
class AddOp : public Op<AddOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl, OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  using OperandAdaptor = AddOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value lhs();
  Value rhs();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs);
  static void build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::BroadcastOp declarations
//===----------------------------------------------------------------------===//

class BroadcastOpOperandAdaptor {
public:
  BroadcastOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lhs();
  Value rhs();

private:
  ArrayRef<Value> tblgen_operands;
};
class BroadcastOp : public Op<BroadcastOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = BroadcastOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value lhs();
  Value rhs();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  StringAttr errorAttr();
  Optional< StringRef > error();
  void errorAttr(StringAttr attr);
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs, /*optional*/StringAttr error);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs, /*optional*/StringAttr error);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::ConcatOp declarations
//===----------------------------------------------------------------------===//

class ConcatOpOperandAdaptor {
public:
  ConcatOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lhs();
  Value rhs();

private:
  ArrayRef<Value> tblgen_operands;
};
class ConcatOp : public Op<ConcatOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl, InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = ConcatOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value lhs();
  Value rhs();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs);
  static void build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes);
};

//===----------------------------------------------------------------------===//
// shape::ConstantOp declarations
//===----------------------------------------------------------------------===//

class ConstantOpOperandAdaptor {
public:
  ConstantOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);

private:
  ArrayRef<Value> tblgen_operands;
};
class ConstantOp : public Op<ConstantOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using OperandAdaptor = ConstantOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::result_range getODSResults(unsigned index);
  Value result();
  Attribute valueAttr();
  Attribute value();
  void valueAttr(Attribute attr);
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Attribute value);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Attribute value);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::DebugPrintOp declarations
//===----------------------------------------------------------------------===//

class DebugPrintOpOperandAdaptor {
public:
  DebugPrintOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class DebugPrintOp : public Op<DebugPrintOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = DebugPrintOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  Operation::result_range getODSResults(unsigned index);
  Value output();
  static void build(Builder *odsBuilder, OperationState &odsState, Type output, Value input);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::FromExtentTensorOp declarations
//===----------------------------------------------------------------------===//

class FromExtentTensorOpOperandAdaptor {
public:
  FromExtentTensorOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class FromExtentTensorOp : public Op<FromExtentTensorOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = FromExtentTensorOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value input);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::JoinOp declarations
//===----------------------------------------------------------------------===//

class JoinOpOperandAdaptor {
public:
  JoinOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value arg0();
  Value arg1();

private:
  ArrayRef<Value> tblgen_operands;
};
class JoinOp : public Op<JoinOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = JoinOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value arg0();
  Value arg1();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  StringAttr errorAttr();
  Optional< StringRef > error();
  void errorAttr(StringAttr attr);
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value arg0, Value arg1, /*optional*/StringAttr error);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value arg0, Value arg1, /*optional*/StringAttr error);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::MulOp declarations
//===----------------------------------------------------------------------===//

class MulOpOperandAdaptor {
public:
  MulOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lhs();
  Value rhs();

private:
  ArrayRef<Value> tblgen_operands;
};
class MulOp : public Op<MulOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl, OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;
  using OperandAdaptor = MulOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value lhs();
  Value rhs();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs);
  static void build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::ReduceOp declarations
//===----------------------------------------------------------------------===//

class ReduceOpOperandAdaptor {
public:
  ReduceOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value shape();
  ArrayRef<Value> args();

private:
  ArrayRef<Value> tblgen_operands;
};
class ReduceOp : public Op<ReduceOp, OpTrait::OneRegion, OpTrait::VariadicResults, OpTrait::ZeroSuccessor, OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = ReduceOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value shape();
  Operation::operand_range args();
  Operation::result_range getODSResults(unsigned index);
  Operation::result_range result();
  Region &body();
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> result, Value shape, ValueRange args);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::ShapeOfOp declarations
//===----------------------------------------------------------------------===//

class ShapeOfOpOperandAdaptor {
public:
  ShapeOfOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value arg();

private:
  ArrayRef<Value> tblgen_operands;
};
class ShapeOfOp : public Op<ShapeOfOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = ShapeOfOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value arg();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value arg);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value arg);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::SplitAtOp declarations
//===----------------------------------------------------------------------===//

class SplitAtOpOperandAdaptor {
public:
  SplitAtOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value operand();
  Value index();

private:
  ArrayRef<Value> tblgen_operands;
};
class SplitAtOp : public Op<SplitAtOp, OpTrait::ZeroRegion, OpTrait::NResults<2>::Impl, OpTrait::ZeroSuccessor, OpTrait::NOperands<2>::Impl, InferTypeOpInterface::Trait, OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using OperandAdaptor = SplitAtOpOperandAdaptor;
  void getAsmResultNames(OpAsmSetValueNameFn setNameFn);
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value operand();
  Value index();
  Operation::result_range getODSResults(unsigned index);
  Value head();
  Value tail();
  static void build(Builder *odsBuilder, OperationState &odsState, Type head, Type tail, Value operand, Value index);
  static void build(Builder *odsBuilder, OperationState &odsState, Value operand, Value index);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand, Value index);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static void build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static LogicalResult inferReturnTypes(MLIRContext* context, Optional<Location> location, ValueRange operands, ArrayRef<NamedAttribute> attributes, RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes);
};

//===----------------------------------------------------------------------===//
// shape::ToExtentTensorOp declarations
//===----------------------------------------------------------------------===//

class ToExtentTensorOpOperandAdaptor {
public:
  ToExtentTensorOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value input();

private:
  ArrayRef<Value> tblgen_operands;
};
class ToExtentTensorOp : public Op<ToExtentTensorOp, OpTrait::ZeroRegion, OpTrait::OneResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand> {
public:
  using Op::Op;
  using OperandAdaptor = ToExtentTensorOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value input();
  Operation::result_range getODSResults(unsigned index);
  Value result();
  static void build(Builder *odsBuilder, OperationState &odsState, Type result, Value input);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// shape::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpOperandAdaptor {
public:
  YieldOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  ArrayRef<Value> operands();

private:
  ArrayRef<Value> tblgen_operands;
};
class YieldOp : public Op<YieldOp, OpTrait::ZeroRegion, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::VariadicOperands, MemoryEffectOpInterface::Trait, OpTrait::IsTerminator> {
public:
  using Op::Op;
  using OperandAdaptor = YieldOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range operands();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *b, OperationState &result);
  static void build(Builder *odsBuilder, OperationState &odsState, ValueRange operands);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

#endif  // GET_OP_CLASSES

