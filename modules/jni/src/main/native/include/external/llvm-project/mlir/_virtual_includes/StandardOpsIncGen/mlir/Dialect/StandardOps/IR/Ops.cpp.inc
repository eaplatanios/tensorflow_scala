/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

AbsFOp,
AddFOp,
AddIOp,
AllocOp,
AllocaOp,
AndOp,
AssumeAlignmentOp,
AtomicRMWOp,
AtomicYieldOp,
BranchOp,
CallIndirectOp,
CallOp,
CeilFOp,
CmpFOp,
CmpIOp,
CondBranchOp,
ConstantOp,
CopySignOp,
CosOp,
DeallocOp,
DimOp,
DivFOp,
Exp2Op,
ExpOp,
ExtractElementOp,
FPExtOp,
FPTruncOp,
GenericAtomicRMWOp,
IndexCastOp,
LoadOp,
Log10Op,
Log2Op,
LogOp,
MemRefCastOp,
MulFOp,
MulIOp,
NegFOp,
OrOp,
PrefetchOp,
RankOp,
RemFOp,
ReturnOp,
RsqrtOp,
SIToFPOp,
SelectOp,
ShiftLeftOp,
SignExtendIOp,
SignedDivIOp,
SignedRemIOp,
SignedShiftRightOp,
SplatOp,
SqrtOp,
StoreOp,
SubFOp,
SubIOp,
SubViewOp,
TanhOp,
TensorCastOp,
TensorLoadOp,
TensorStoreOp,
TruncateIOp,
UnsignedDivIOp,
UnsignedRemIOp,
UnsignedShiftRightOp,
ViewOp,
XOrOp,
ZeroExtendIOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// AbsFOp definitions
//===----------------------------------------------------------------------===//

AbsFOpOperandAdaptor::AbsFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AbsFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AbsFOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef AbsFOp::getOperationName() {
  return "std.absf";
}

Operation::operand_range AbsFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AbsFOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range AbsFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void AbsFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void AbsFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AbsFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AbsFOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void AbsFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult AbsFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AbsFOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult AbsFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void AbsFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AddFOp definitions
//===----------------------------------------------------------------------===//

AddFOpOperandAdaptor::AddFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AddFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AddFOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value AddFOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef AddFOp::getOperationName() {
  return "std.addf";
}

Operation::operand_range AddFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AddFOp::lhs() {
  return *getODSOperands(0).begin();
}

Value AddFOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range AddFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void AddFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddFOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult AddFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AddFOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult AddFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void AddFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AddIOp definitions
//===----------------------------------------------------------------------===//

AddIOpOperandAdaptor::AddIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AddIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AddIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value AddIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef AddIOp::getOperationName() {
  return "std.addi";
}

Operation::operand_range AddIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AddIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value AddIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range AddIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void AddIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult AddIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AddIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult AddIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void AddIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AllocOp definitions
//===----------------------------------------------------------------------===//

AllocOpOperandAdaptor::AllocOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AllocOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> AllocOpOperandAdaptor::value() {
  return getODSOperands(0);
}

StringRef AllocOp::getOperationName() {
  return "std.alloc";
}

Operation::operand_range AllocOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range AllocOp::value() {
  return getODSOperands(0);
}

Operation::result_range AllocOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

IntegerAttr AllocOp::alignmentAttr() {
  return this->getAttr("alignment").dyn_cast_or_null<IntegerAttr>();
}

Optional< APInt > AllocOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? Optional< APInt >(attr.getValue()) : (llvm::None);
}

void AllocOp::alignmentAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("alignment", attr);
}

void AllocOp::build(Builder *builder, OperationState &result, MemRefType memrefType) {
       result.types.push_back(memrefType);
     
}

void AllocOp::build(Builder *builder, OperationState &result, MemRefType memrefType, ValueRange operands, IntegerAttr alignment ) {
       result.addOperands(operands);
       result.types.push_back(memrefType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
     
}

void AllocOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, ValueRange value, /*optional*/IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  odsState.addTypes(resultType0);
}

void AllocOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange value, /*optional*/IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AllocOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAllocLikeOp(parser, result);
}

void AllocOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AllocOp::verify() {
  auto tblgen_alignment = this->getAttr("alignment");
  if (tblgen_alignment) {
    if (!((((tblgen_alignment.isa<IntegerAttr>())) && ((tblgen_alignment.cast<IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<IntegerAttr>().getInt() >= 0)))) return emitOpError("attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}




//===----------------------------------------------------------------------===//
// AllocaOp definitions
//===----------------------------------------------------------------------===//

AllocaOpOperandAdaptor::AllocaOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AllocaOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> AllocaOpOperandAdaptor::value() {
  return getODSOperands(0);
}

StringRef AllocaOp::getOperationName() {
  return "std.alloca";
}

Operation::operand_range AllocaOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range AllocaOp::value() {
  return getODSOperands(0);
}

Operation::result_range AllocaOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

IntegerAttr AllocaOp::alignmentAttr() {
  return this->getAttr("alignment").dyn_cast_or_null<IntegerAttr>();
}

Optional< APInt > AllocaOp::alignment() {
  auto attr = alignmentAttr();
  return attr ? Optional< APInt >(attr.getValue()) : (llvm::None);
}

void AllocaOp::alignmentAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("alignment", attr);
}

void AllocaOp::build(Builder *builder, OperationState &result, MemRefType memrefType) {
       result.types.push_back(memrefType);
     
}

void AllocaOp::build(Builder *builder, OperationState &result, MemRefType memrefType, ValueRange operands, IntegerAttr alignment ) {
       result.addOperands(operands);
       result.types.push_back(memrefType);
       if (alignment)
         result.addAttribute(getAlignmentAttrName(), alignment);
     
}

void AllocaOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, ValueRange value, /*optional*/IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  odsState.addTypes(resultType0);
}

void AllocaOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange value, /*optional*/IntegerAttr alignment) {
  odsState.addOperands(value);
  if (alignment) {
  odsState.addAttribute("alignment", alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult AllocaOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseAllocLikeOp(parser, result);
}

void AllocaOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult AllocaOp::verify() {
  auto tblgen_alignment = this->getAttr("alignment");
  if (tblgen_alignment) {
    if (!((((tblgen_alignment.isa<IntegerAttr>())) && ((tblgen_alignment.cast<IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_alignment.cast<IntegerAttr>().getInt() >= 0)))) return emitOpError("attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}




//===----------------------------------------------------------------------===//
// AndOp definitions
//===----------------------------------------------------------------------===//

AndOpOperandAdaptor::AndOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AndOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AndOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value AndOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef AndOp::getOperationName() {
  return "std.and";
}

Operation::operand_range AndOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AndOp::lhs() {
  return *getODSOperands(0).begin();
}

Value AndOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range AndOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void AndOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AndOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AndOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AndOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AndOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult AndOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void AndOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult AndOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void AndOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// AssumeAlignmentOp definitions
//===----------------------------------------------------------------------===//

AssumeAlignmentOpOperandAdaptor::AssumeAlignmentOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AssumeAlignmentOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AssumeAlignmentOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

StringRef AssumeAlignmentOp::getOperationName() {
  return "std.assume_alignment";
}

Operation::operand_range AssumeAlignmentOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AssumeAlignmentOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::result_range AssumeAlignmentOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

IntegerAttr AssumeAlignmentOp::alignmentAttr() {
  return this->getAttr("alignment").cast<IntegerAttr>();
}

APInt AssumeAlignmentOp::alignment() {
  auto attr = alignmentAttr();
  return attr.getValue();
}

void AssumeAlignmentOp::alignmentAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("alignment", attr);
}

void AssumeAlignmentOp::build(Builder *odsBuilder, OperationState &odsState, Value memref, IntegerAttr alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", alignment);
}

void AssumeAlignmentOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, IntegerAttr alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeAlignmentOp::build(Builder *odsBuilder, OperationState &odsState, Value memref, APInt alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", (*odsBuilder).getIntegerAttr((*odsBuilder).getIntegerType(32), alignment));
}

void AssumeAlignmentOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, APInt alignment) {
  odsState.addOperands(memref);
  odsState.addAttribute("alignment", (*odsBuilder).getIntegerAttr((*odsBuilder).getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeAlignmentOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult AssumeAlignmentOp::verify() {
  auto tblgen_alignment = this->getAttr("alignment");
  if (!tblgen_alignment) return emitOpError("requires attribute 'alignment'");
  {
    if (!((((tblgen_alignment.isa<IntegerAttr>())) && ((tblgen_alignment.cast<IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_alignment.cast<IntegerAttr>().getValue().isStrictlyPositive())))) return emitOpError("attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute whose value is positive");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

ParseResult AssumeAlignmentOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  IntegerAttr alignmentAttr;
  if (parser.parseAttribute(alignmentAttr, parser.getBuilder().getIntegerType(32), "alignment", result.attributes))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  return success();
}

void AssumeAlignmentOp::print(OpAsmPrinter &p) {
  p << "assume_alignment";
  p << " ";
  p << memref();
  p << ",";
  p << " ";
  p.printAttributeWithoutType(alignmentAttr());
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"alignment"});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
}


//===----------------------------------------------------------------------===//
// AtomicRMWOp definitions
//===----------------------------------------------------------------------===//

AtomicRMWOpOperandAdaptor::AtomicRMWOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AtomicRMWOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value AtomicRMWOpOperandAdaptor::value() {
  return *getODSOperands(0).begin();
}

Value AtomicRMWOpOperandAdaptor::memref() {
  return *getODSOperands(1).begin();
}

ArrayRef<Value> AtomicRMWOpOperandAdaptor::indices() {
  return getODSOperands(2);
}

StringRef AtomicRMWOp::getOperationName() {
  return "std.atomic_rmw";
}

Operation::operand_range AtomicRMWOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value AtomicRMWOp::value() {
  return *getODSOperands(0).begin();
}

Value AtomicRMWOp::memref() {
  return *getODSOperands(1).begin();
}

Operation::operand_range AtomicRMWOp::indices() {
  return getODSOperands(2);
}

Operation::result_range AtomicRMWOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value AtomicRMWOp::result() {
  return *getODSResults(0).begin();
}

IntegerAttr AtomicRMWOp::kindAttr() {
  return this->getAttr("kind").cast<IntegerAttr>();
}

::mlir::AtomicRMWKind AtomicRMWOp::kind() {
  auto attr = kindAttr();
  return static_cast<::mlir::AtomicRMWKind>(attr.getInt());
}

void AtomicRMWOp::kindAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("kind", attr);
}

void AtomicRMWOp::build(Builder *odsBuilder, OperationState &odsState, Type result, IntegerAttr kind, Value value, Value memref, ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", kind);
  odsState.addTypes(result);
}

void AtomicRMWOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, IntegerAttr kind, Value value, Value memref, ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", kind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(Builder *odsBuilder, OperationState &odsState, Type result, ::mlir::AtomicRMWKind kind, Value value, Value memref, ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", (*odsBuilder).getI64IntegerAttr(static_cast<int64_t>(kind)));
  odsState.addTypes(result);
}

void AtomicRMWOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ::mlir::AtomicRMWKind kind, Value value, Value memref, ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("kind", (*odsBuilder).getI64IntegerAttr(static_cast<int64_t>(kind)));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult AtomicRMWOp::verify() {
  auto tblgen_kind = this->getAttr("kind");
  if (!tblgen_kind) return emitOpError("requires attribute 'kind'");
  {
    if (!((((tblgen_kind.isa<IntegerAttr>())) && ((tblgen_kind.cast<IntegerAttr>().getType().isSignlessInteger(64)))) && (((tblgen_kind.cast<IntegerAttr>().getInt() == 0)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 1)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 2)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 3)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 4)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 5)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 6)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 7)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 8)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 9)) || ((tblgen_kind.cast<IntegerAttr>().getInt() == 10))))) return emitOpError("attribute 'kind' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<FloatType>())))) {
        return emitOpError("operand #") << index << " must be signless integer or floating-point, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && (((v.getType().cast<ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be memref of signless integer or floating-point values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<FloatType>())))) {
        return emitOpError("result #") << index << " must be signless integer or floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((llvm::is_splat(llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSResults(0).begin()).getType()}))))) {
    return emitOpError("failed to verify that all of {value, result} have same type");
  }
  if (!(((*this->getODSOperands(1).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSOperands(0).begin()).getType()))) {
    return emitOpError("failed to verify that value type matches element type of memref");
  }
  return ::verify(*this);
}

ParseResult AtomicRMWOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType valueRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> valueOperands(valueRawOperands);  llvm::SMLoc valueOperandsLoc = parser.getCurrentLocation();
  (void)valueOperandsLoc;
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> indicesOperands;
  llvm::SMLoc indicesOperandsLoc = parser.getCurrentLocation();
  (void)indicesOperandsLoc;
  Type valueRawTypes[1];
  ArrayRef<Type> valueTypes(valueRawTypes);
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);
  Type resultRawTypes[1];
  ArrayRef<Type> resultTypes(resultRawTypes);

  {
    StringAttr attrVal;
    SmallVector<NamedAttribute, 1> attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "kind", attrStorage))
      return failure();

    auto attrOptional = ::mlir::symbolizeAtomicRMWKind(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "kind attribute specification: " << attrVal;

    result.addAttribute("kind", parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue())));
  }

  if (parser.parseOperand(valueRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(indicesOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();
  if (parser.parseLParen())
    return failure();

  if (parser.parseType(valueRawTypes[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseArrow())
    return failure();

  if (parser.parseType(resultRawTypes[0]))
    return failure();
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return failure();
  return success();
}

void AtomicRMWOp::print(OpAsmPrinter &p) {
  p << "atomic_rmw";
  p << " ";
  p << "\"" << stringifyAtomicRMWKind(kind()) << "\"";
  p << " ";
  p << value();
  p << ",";
  p << " ";
  p << memref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"kind"});
  p << " " << ":";
  p << " " << "(";
  p << ArrayRef<Type>(value().getType());
  p << ",";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
  p << ")";
  p << " " << "->";
  p << " ";
  p << ArrayRef<Type>(result().getType());
}


//===----------------------------------------------------------------------===//
// AtomicYieldOp definitions
//===----------------------------------------------------------------------===//

AtomicYieldOpOperandAdaptor::AtomicYieldOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AtomicYieldOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AtomicYieldOpOperandAdaptor::result() {
  return *getODSOperands(0).begin();
}

StringRef AtomicYieldOp::getOperationName() {
  return "std.atomic_yield";
}

Operation::operand_range AtomicYieldOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AtomicYieldOp::result() {
  return *getODSOperands(0).begin();
}

Operation::result_range AtomicYieldOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void AtomicYieldOp::build(Builder *odsBuilder, OperationState &odsState, Value result) {
  odsState.addOperands(result);
}

void AtomicYieldOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value result) {
  odsState.addOperands(result);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicYieldOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult AtomicYieldOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

ParseResult AtomicYieldOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType resultRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> resultOperands(resultRawOperands);  llvm::SMLoc resultOperandsLoc = parser.getCurrentLocation();
  (void)resultOperandsLoc;
  Type resultRawTypes[1];
  ArrayRef<Type> resultTypes(resultRawTypes);

  if (parser.parseOperand(resultRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(resultRawTypes[0]))
    return failure();
  if (parser.resolveOperands(resultOperands, resultTypes, resultOperandsLoc, result.operands))
    return failure();
  return success();
}

void AtomicYieldOp::print(OpAsmPrinter &p) {
  p << "atomic_yield";
  p << " ";
  p << result();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(result().getType());
}

void AtomicYieldOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// BranchOp definitions
//===----------------------------------------------------------------------===//

BranchOpOperandAdaptor::BranchOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> BranchOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> BranchOpOperandAdaptor::destOperands() {
  return getODSOperands(0);
}

StringRef BranchOp::getOperationName() {
  return "std.br";
}

Operation::operand_range BranchOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range BranchOp::destOperands() {
  return getODSOperands(0);
}

Operation::result_range BranchOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Block *BranchOp::dest() {
  return this->getOperation()->getSuccessor(0);
}

void BranchOp::build(Builder *, OperationState &result, Block *dest, ValueRange destOperands ) {
    result.addSuccessors(dest);
    result.addOperands(destOperands);
  
}

void BranchOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange destOperands, Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
}

void BranchOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange destOperands, Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult BranchOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return mlir::success();
}







ParseResult BranchOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> destOperandsOperands;
  llvm::SMLoc destOperandsOperandsLoc = parser.getCurrentLocation();
  (void)destOperandsOperandsLoc;
  SmallVector<Type, 1> destOperandsTypes;

  Block *destSuccessor = nullptr;
  if (parser.parseSuccessor(destSuccessor))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {

  if (parser.parseOperandList(destOperandsOperands))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(destOperandsTypes))
    return failure();
  if (parser.parseRParen())
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return failure();
  result.addSuccessors(destSuccessor);
  return success();
}

void BranchOp::print(OpAsmPrinter &p) {
  p << "br";
  p << " ";
  p << dest();
  if (!destOperands().empty()) {
  p << "(";
  p << destOperands();
  p << " " << ":";
  p << " ";
  p << destOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
}

void BranchOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CallIndirectOp definitions
//===----------------------------------------------------------------------===//

CallIndirectOpOperandAdaptor::CallIndirectOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CallIndirectOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value CallIndirectOpOperandAdaptor::callee() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> CallIndirectOpOperandAdaptor::operands() {
  return getODSOperands(1);
}

StringRef CallIndirectOp::getOperationName() {
  return "std.call_indirect";
}

Operation::operand_range CallIndirectOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value CallIndirectOp::callee() {
  return *getODSOperands(0).begin();
}

Operation::operand_range CallIndirectOp::operands() {
  return getODSOperands(1);
}

Operation::result_range CallIndirectOp::getODSResults(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic result corresponds to.
  // This assumes all static variadic results have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // result (variadic or not) as size 1. So here for each previous static variadic
  // result, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static result starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->result_begin(), offset), std::next(getOperation()->result_begin(), offset + size)};
}

Operation::result_range CallIndirectOp::results() {
  return getODSResults(0);
}

void CallIndirectOp::build(Builder *, OperationState &result, Value callee,ValueRange operands ) {
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addTypes(callee.getType().cast<FunctionType>().getResults());
  
}

void CallIndirectOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> results, Value callee, ValueRange operands) {
  odsState.addOperands(callee);
  odsState.addOperands(operands);
  odsState.addTypes(results);
}

void CallIndirectOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

LogicalResult CallIndirectOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<FunctionType>()))) {
        return emitOpError("operand #") << index << " must be function type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<FunctionType>().getInputs() == this->getODSOperands(1).getType()))) {
    return emitOpError("failed to verify that callee input types match argument types");
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<FunctionType>().getResults() == this->getODSResults(0).getType()))) {
    return emitOpError("failed to verify that callee result types match result types");
  }
  return mlir::success();
}



ParseResult CallIndirectOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType calleeRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> calleeOperands(calleeRawOperands);  llvm::SMLoc calleeOperandsLoc = parser.getCurrentLocation();
  (void)calleeOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> operandsOperands;
  llvm::SMLoc operandsOperandsLoc = parser.getCurrentLocation();
  (void)operandsOperandsLoc;
  Type calleeRawTypes[1];
  ArrayRef<Type> calleeTypes(calleeRawTypes);

  if (parser.parseOperand(calleeRawOperands[0]))
    return failure();
  if (parser.parseLParen())
    return failure();

  if (parser.parseOperandList(operandsOperands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(calleeRawTypes[0]))
    return failure();
  for (Type type : calleeTypes) {
    (void)type;
    if (!((type.isa<FunctionType>()))) {
      return parser.emitError(parser.getNameLoc()) << "'callee' must be function type, but got " << type;
    }
  }
  result.addTypes(calleeTypes[0].cast<FunctionType>().getResults());
  if (parser.resolveOperands(calleeOperands, calleeTypes, calleeOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(operandsOperands, calleeTypes[0].cast<FunctionType>().getInputs(), operandsOperandsLoc, result.operands))
    return failure();
  return success();
}

void CallIndirectOp::print(OpAsmPrinter &p) {
  p << "call_indirect";
  p << " ";
  p << callee();
  p << "(";
  p << operands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(callee().getType());
}


//===----------------------------------------------------------------------===//
// CallOp definitions
//===----------------------------------------------------------------------===//

CallOpOperandAdaptor::CallOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CallOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> CallOpOperandAdaptor::operands() {
  return getODSOperands(0);
}

StringRef CallOp::getOperationName() {
  return "std.call";
}

Operation::operand_range CallOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range CallOp::operands() {
  return getODSOperands(0);
}

Operation::result_range CallOp::getODSResults(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic result corresponds to.
  // This assumes all static variadic results have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // result (variadic or not) as size 1. So here for each previous static variadic
  // result, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static result starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->result_begin(), offset), std::next(getOperation()->result_begin(), offset + size)};
}

FlatSymbolRefAttr CallOp::calleeAttr() {
  return this->getAttr("callee").cast<FlatSymbolRefAttr>();
}

StringRef CallOp::callee() {
  auto attr = calleeAttr();
  return attr.getValue();
}

void CallOp::calleeAttr(FlatSymbolRefAttr attr) {
  this->getOperation()->setAttr("callee", attr);
}

void CallOp::build(Builder *builder, OperationState &result, FuncOp callee,ValueRange operands ) {
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  
}

void CallOp::build(Builder *builder, OperationState &result, SymbolRefAttr callee,ArrayRef<Type> results, ValueRange operands ) {
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addTypes(results);
  
}

void CallOp::build(Builder *builder, OperationState &result, StringRef callee,ArrayRef<Type> results, ValueRange operands ) {
      build(builder, result, builder->getSymbolRefAttr(callee), results,
            operands);
  
}

void CallOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultType0, FlatSymbolRefAttr callee, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("callee", callee);
  odsState.addTypes(resultType0);
}

void CallOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultType0, StringRef callee, ValueRange operands) {
  odsState.addOperands(operands);
  odsState.addAttribute("callee", (*odsBuilder).getSymbolRefAttr(callee));
  odsState.addTypes(resultType0);
}

void CallOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

LogicalResult CallOp::verify() {
  auto tblgen_callee = this->getAttr("callee");
  if (!tblgen_callee) return emitOpError("requires attribute 'callee'");
  {
    if (!((tblgen_callee.isa<FlatSymbolRefAttr>()))) return emitOpError("attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operandsOperands;
  llvm::SMLoc operandsOperandsLoc = parser.getCurrentLocation();
  (void)operandsOperandsLoc;
  ArrayRef<Type> operandsTypes;
  ArrayRef<Type> allResultTypes;

  FlatSymbolRefAttr calleeAttr;
  if (parser.parseAttribute(calleeAttr, parser.getBuilder().getType<NoneType>(), "callee", result.attributes))
    return failure();
  if (parser.parseLParen())
    return failure();

  if (parser.parseOperandList(operandsOperands))
    return failure();
  if (parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  FunctionType operands__allResult_functionType;
  if (parser.parseType(operands__allResult_functionType))
    return failure();
  operandsTypes = operands__allResult_functionType.getInputs();
  allResultTypes = operands__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return failure();
  return success();
}

void CallOp::print(OpAsmPrinter &p) {
  p << "call";
  p << " ";
  p.printAttributeWithoutType(calleeAttr());
  p << "(";
  p << operands();
  p << ")";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"callee"});
  p << " " << ":";
  p << " ";
  p.printFunctionalType(operands().getTypes(), getOperation()->getResultTypes());
}


//===----------------------------------------------------------------------===//
// CeilFOp definitions
//===----------------------------------------------------------------------===//

CeilFOpOperandAdaptor::CeilFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CeilFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value CeilFOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef CeilFOp::getOperationName() {
  return "std.ceilf";
}

Operation::operand_range CeilFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value CeilFOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range CeilFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void CeilFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void CeilFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CeilFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CeilFOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void CeilFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult CeilFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void CeilFOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult CeilFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void CeilFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CmpFOp definitions
//===----------------------------------------------------------------------===//

CmpFOpOperandAdaptor::CmpFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CmpFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value CmpFOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value CmpFOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef CmpFOp::getOperationName() {
  return "std.cmpf";
}

Operation::operand_range CmpFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value CmpFOp::lhs() {
  return *getODSOperands(0).begin();
}

Value CmpFOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range CmpFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value CmpFOp::result() {
  return *getODSResults(0).begin();
}

IntegerAttr CmpFOp::predicateAttr() {
  return this->getAttr("predicate").cast<IntegerAttr>();
}

::mlir::CmpFPredicate CmpFOp::predicate() {
  auto attr = predicateAttr();
  return static_cast<::mlir::CmpFPredicate>(attr.getInt());
}

void CmpFOp::predicateAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("predicate", attr);
}

void CmpFOp::build(Builder *builder, OperationState &result, CmpFPredicate predicate,Value lhs, Value rhs) {
      ::buildCmpFOp(builder, result, predicate, lhs, rhs);
  
}

void CmpFOp::build(Builder *odsBuilder, OperationState &odsState, Type result, IntegerAttr predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addTypes(result);
}

void CmpFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, IntegerAttr predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpFOp::build(Builder *odsBuilder, OperationState &odsState, Type result, ::mlir::CmpFPredicate predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", (*odsBuilder).getI64IntegerAttr(static_cast<int64_t>(predicate)));
  odsState.addTypes(result);
}

void CmpFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ::mlir::CmpFPredicate predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", (*odsBuilder).getI64IntegerAttr(static_cast<int64_t>(predicate)));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult CmpFOp::verify() {
  auto tblgen_predicate = this->getAttr("predicate");
  if (!tblgen_predicate) return emitOpError("requires attribute 'predicate'");
  {
    if (!((((tblgen_predicate.isa<IntegerAttr>())) && ((tblgen_predicate.cast<IntegerAttr>().getType().isSignlessInteger(64)))) && (((tblgen_predicate.cast<IntegerAttr>().getInt() == 0)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 1)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 2)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 3)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 4)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 5)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 6)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 7)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 8)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 9)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 10)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 11)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 12)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 13)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 14)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 15))))) return emitOpError("attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(1))) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(1)))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(1)))))) {
        return emitOpError("result #") << index << " must be bool-like, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((getI1SameShape((*this->getODSOperands(0).begin()).getType()) == (*this->getODSResults(0).begin()).getType()))) {
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  }
  return success();
}



ParseResult CmpFOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType lhsRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  llvm::SMLoc lhsOperandsLoc = parser.getCurrentLocation();
  (void)lhsOperandsLoc;
  OpAsmParser::OperandType rhsRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  llvm::SMLoc rhsOperandsLoc = parser.getCurrentLocation();
  (void)rhsOperandsLoc;
  Type lhsRawTypes[1];
  ArrayRef<Type> lhsTypes(lhsRawTypes);

  {
    StringAttr attrVal;
    SmallVector<NamedAttribute, 1> attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "predicate", attrStorage))
      return failure();

    auto attrOptional = ::mlir::symbolizeCmpFPredicate(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "predicate attribute specification: " << attrVal;

    result.addAttribute("predicate", parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue())));
  }
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(lhsRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(rhsRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(lhsRawTypes[0]))
    return failure();
  for (Type type : lhsTypes) {
    (void)type;
    if (!(((type.isa<FloatType>())) || (((type.isa<VectorType>())) && ((type.cast<ShapedType>().getElementType().isa<FloatType>()))) || (((type.isa<TensorType>())) && ((type.cast<ShapedType>().getElementType().isa<FloatType>()))))) {
      return parser.emitError(parser.getNameLoc()) << "'lhs' must be floating-point-like, but got " << type;
    }
  }
  result.addTypes(getI1SameShape(lhsTypes[0]));
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes, rhsOperandsLoc, result.operands))
    return failure();
  return success();
}

void CmpFOp::print(OpAsmPrinter &p) {
  p << "cmpf";
  p << " ";
  p << "\"" << stringifyCmpFPredicate(predicate()) << "\"";
  p << ",";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"predicate"});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(lhs().getType());
}

void CmpFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CmpIOp definitions
//===----------------------------------------------------------------------===//

CmpIOpOperandAdaptor::CmpIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CmpIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value CmpIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value CmpIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef CmpIOp::getOperationName() {
  return "std.cmpi";
}

Operation::operand_range CmpIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value CmpIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value CmpIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range CmpIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value CmpIOp::result() {
  return *getODSResults(0).begin();
}

IntegerAttr CmpIOp::predicateAttr() {
  return this->getAttr("predicate").cast<IntegerAttr>();
}

::mlir::CmpIPredicate CmpIOp::predicate() {
  auto attr = predicateAttr();
  return static_cast<::mlir::CmpIPredicate>(attr.getInt());
}

void CmpIOp::predicateAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("predicate", attr);
}

void CmpIOp::build(Builder *builder, OperationState &result, CmpIPredicate predicate,Value lhs, Value rhs) {
      ::buildCmpIOp(builder, result, predicate, lhs, rhs);
  
}

void CmpIOp::build(Builder *odsBuilder, OperationState &odsState, Type result, IntegerAttr predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  odsState.addTypes(result);
}

void CmpIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, IntegerAttr predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpIOp::build(Builder *odsBuilder, OperationState &odsState, Type result, ::mlir::CmpIPredicate predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", (*odsBuilder).getI64IntegerAttr(static_cast<int64_t>(predicate)));
  odsState.addTypes(result);
}

void CmpIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ::mlir::CmpIPredicate predicate, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute("predicate", (*odsBuilder).getI64IntegerAttr(static_cast<int64_t>(predicate)));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CmpIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult CmpIOp::verify() {
  auto tblgen_predicate = this->getAttr("predicate");
  if (!tblgen_predicate) return emitOpError("requires attribute 'predicate'");
  {
    if (!((((tblgen_predicate.isa<IntegerAttr>())) && ((tblgen_predicate.cast<IntegerAttr>().getType().isSignlessInteger(64)))) && (((tblgen_predicate.cast<IntegerAttr>().getInt() == 0)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 1)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 2)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 3)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 4)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 5)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 6)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 7)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 8)) || ((tblgen_predicate.cast<IntegerAttr>().getInt() == 9))))) return emitOpError("attribute 'predicate' failed to satisfy constraint: allowed 64-bit signless integer cases: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(1))) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(1)))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(1)))))) {
        return emitOpError("result #") << index << " must be bool-like, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((getI1SameShape((*this->getODSOperands(0).begin()).getType()) == (*this->getODSResults(0).begin()).getType()))) {
    return emitOpError("failed to verify that result type has i1 element type and same shape as operands");
  }
  return success();
}



ParseResult CmpIOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType lhsRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> lhsOperands(lhsRawOperands);  llvm::SMLoc lhsOperandsLoc = parser.getCurrentLocation();
  (void)lhsOperandsLoc;
  OpAsmParser::OperandType rhsRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> rhsOperands(rhsRawOperands);  llvm::SMLoc rhsOperandsLoc = parser.getCurrentLocation();
  (void)rhsOperandsLoc;
  Type lhsRawTypes[1];
  ArrayRef<Type> lhsTypes(lhsRawTypes);

  {
    StringAttr attrVal;
    SmallVector<NamedAttribute, 1> attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(attrVal, parser.getBuilder().getNoneType(),
                              "predicate", attrStorage))
      return failure();

    auto attrOptional = ::mlir::symbolizeCmpIPredicate(attrVal.getValue());
    if (!attrOptional)
      return parser.emitError(loc, "invalid ")
             << "predicate attribute specification: " << attrVal;

    result.addAttribute("predicate", parser.getBuilder().getI64IntegerAttr(static_cast<int64_t>(attrOptional.getValue())));
  }
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(lhsRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(rhsRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(lhsRawTypes[0]))
    return failure();
  for (Type type : lhsTypes) {
    (void)type;
    if (!(((type.isSignlessInteger())) || ((type.isa<IndexType>())) || (((type.isa<VectorType>())) && ((type.cast<ShapedType>().getElementType().isSignlessInteger()))) || (((type.isa<TensorType>())) && ((type.cast<ShapedType>().getElementType().isSignlessInteger()))))) {
      return parser.emitError(parser.getNameLoc()) << "'lhs' must be signless-integer-like, but got " << type;
    }
  }
  result.addTypes(getI1SameShape(lhsTypes[0]));
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(rhsOperands, lhsTypes, rhsOperandsLoc, result.operands))
    return failure();
  return success();
}

void CmpIOp::print(OpAsmPrinter &p) {
  p << "cmpi";
  p << " ";
  p << "\"" << stringifyCmpIPredicate(predicate()) << "\"";
  p << ",";
  p << " ";
  p << lhs();
  p << ",";
  p << " ";
  p << rhs();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"predicate"});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(lhs().getType());
}

void CmpIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CondBranchOp definitions
//===----------------------------------------------------------------------===//

StringRef CondBranchOp::getOperationName() {
  return "std.cond_br";
}

Operation::operand_range CondBranchOp::getODSOperands(unsigned index) {
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned end = start + (*(sizeAttr.begin() + index)).getZExtValue();
  return {std::next(getOperation()->operand_begin(), start), std::next(getOperation()->operand_begin(), end)};
}

Value CondBranchOp::condition() {
  return *getODSOperands(0).begin();
}

Operation::operand_range CondBranchOp::trueDestOperands() {
  return getODSOperands(1);
}

Operation::operand_range CondBranchOp::falseDestOperands() {
  return getODSOperands(2);
}

Operation::result_range CondBranchOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Block *CondBranchOp::trueDest() {
  return this->getOperation()->getSuccessor(0);
}

Block *CondBranchOp::falseDest() {
  return this->getOperation()->getSuccessor(1);
}

void CondBranchOp::build(Builder *builder, OperationState &result, Value condition,Block *trueDest, ValueRange trueOperands,Block *falseDest, ValueRange falseOperands) {
      build(builder, result, condition, trueOperands, falseOperands, trueDest,
            falseDest);
  
}

void CondBranchOp::build(Builder *builder, OperationState &result, Value condition,Block *trueDest, Block *falseDest, ValueRange falseOperands ) {
      build(builder, result, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  
}

void CondBranchOp::build(Builder *odsBuilder, OperationState &odsState, Value condition, ValueRange trueDestOperands, ValueRange falseDestOperands, Block *trueDest, Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder->getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBranchOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value condition, ValueRange trueDestOperands, ValueRange falseDestOperands, Block *trueDest, Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute("operand_segment_sizes", odsBuilder->getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBranchOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult CondBranchOp::verify() {
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
  auto numElements = sizeAttr.getType().cast<ShapedType>().getNumElements();
  if (numElements != 3) {
    return emitOpError("'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 3 elements");
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isSignlessInteger(1)))) {
        return emitOpError("operand #") << index << " must be 1-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
{
    unsigned index = 0; (void)index;
  }
  return mlir::success();
}







ParseResult CondBranchOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType conditionRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> conditionOperands(conditionRawOperands);  llvm::SMLoc conditionOperandsLoc = parser.getCurrentLocation();
  (void)conditionOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> trueDestOperandsOperands;
  llvm::SMLoc trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  (void)trueDestOperandsOperandsLoc;
  SmallVector<Type, 1> trueDestOperandsTypes;
  SmallVector<OpAsmParser::OperandType, 4> falseDestOperandsOperands;
  llvm::SMLoc falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  (void)falseDestOperandsOperandsLoc;
  SmallVector<Type, 1> falseDestOperandsTypes;

  if (parser.parseOperand(conditionRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  Block *trueDestSuccessor = nullptr;
  if (parser.parseSuccessor(trueDestSuccessor))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {

  if (parser.parseOperandList(trueDestOperandsOperands))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return failure();
  if (parser.parseRParen())
    return failure();
  }
  if (parser.parseComma())
    return failure();

  Block *falseDestSuccessor = nullptr;
  if (parser.parseSuccessor(falseDestSuccessor))
    return failure();
  if (succeeded(parser.parseOptionalLParen())) {

  if (parser.parseOperandList(falseDestOperandsOperands))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return failure();
  if (parser.parseRParen())
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, result.operands))
    return failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return failure();
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
  result.addAttribute("operand_segment_sizes", builder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}));
  return success();
}

void CondBranchOp::print(OpAsmPrinter &p) {
  p << "cond_br";
  p << " ";
  p << condition();
  p << ",";
  p << " ";
  p << trueDest();
  if (!trueDestOperands().empty()) {
  p << "(";
  p << trueDestOperands();
  p << " " << ":";
  p << " ";
  p << trueDestOperands().getTypes();
  p << ")";
  }
  p << ",";
  p << " ";
  p << falseDest();
  if (!falseDestOperands().empty()) {
  p << "(";
  p << falseDestOperands();
  p << " " << ":";
  p << " ";
  p << falseDestOperands().getTypes();
  p << ")";
  }
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
}

void CondBranchOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpOperandAdaptor::ConstantOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ConstantOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

StringRef ConstantOp::getOperationName() {
  return "std.constant";
}

Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Operation::result_range ConstantOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Attribute ConstantOp::valueAttr() {
  return this->getAttr("value").cast<Attribute>();
}

Attribute ConstantOp::value() {
  auto attr = valueAttr();
  return attr;
}

void ConstantOp::valueAttr(Attribute attr) {
  this->getOperation()->setAttr("value", attr);
}

void ConstantOp::build(Builder *builder, OperationState &result, Attribute value) {
 build(builder, result, value.getType(), value); 
}

void ConstantOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(resultType0);
}

void ConstantOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Attribute value) {
  odsState.addAttribute("value", value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseConstantOp(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult ConstantOp::verify() {
  auto tblgen_value = this->getAttr("value");
  if (!tblgen_value) return emitOpError("requires attribute 'value'");
  {
    if (!((true))) return emitOpError("attribute 'value' failed to satisfy constraint: any attribute");
  }
  {
    unsigned index = 0; (void)index;
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





void ConstantOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CopySignOp definitions
//===----------------------------------------------------------------------===//

CopySignOpOperandAdaptor::CopySignOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CopySignOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value CopySignOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value CopySignOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef CopySignOp::getOperationName() {
  return "std.copysign";
}

Operation::operand_range CopySignOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value CopySignOp::lhs() {
  return *getODSOperands(0).begin();
}

Value CopySignOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range CopySignOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void CopySignOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void CopySignOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void CopySignOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult CopySignOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void CopySignOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult CopySignOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void CopySignOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// CosOp definitions
//===----------------------------------------------------------------------===//

CosOpOperandAdaptor::CosOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> CosOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value CosOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef CosOp::getOperationName() {
  return "std.cos";
}

Operation::operand_range CosOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value CosOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range CosOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void CosOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void CosOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CosOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CosOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void CosOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult CosOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void CosOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult CosOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void CosOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// DeallocOp definitions
//===----------------------------------------------------------------------===//

DeallocOpOperandAdaptor::DeallocOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> DeallocOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value DeallocOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

StringRef DeallocOp::getOperationName() {
  return "std.dealloc";
}

Operation::operand_range DeallocOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value DeallocOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::result_range DeallocOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void DeallocOp::build(Builder *odsBuilder, OperationState &odsState, Value memref) {
  odsState.addOperands(memref);
}

void DeallocOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DeallocOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult DeallocOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}





ParseResult DeallocOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  return success();
}

void DeallocOp::print(OpAsmPrinter &p) {
  p << "dealloc";
  p << " ";
  p << memref();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
}


//===----------------------------------------------------------------------===//
// DimOp definitions
//===----------------------------------------------------------------------===//

DimOpOperandAdaptor::DimOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> DimOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value DimOpOperandAdaptor::memrefOrTensor() {
  return *getODSOperands(0).begin();
}

StringRef DimOp::getOperationName() {
  return "std.dim";
}

Operation::operand_range DimOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value DimOp::memrefOrTensor() {
  return *getODSOperands(0).begin();
}

Operation::result_range DimOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

IntegerAttr DimOp::indexAttr() {
  return this->getAttr("index").cast<IntegerAttr>();
}

APInt DimOp::index() {
  auto attr = indexAttr();
  return attr.getValue();
}

void DimOp::indexAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("index", attr);
}

void DimOp::build(Builder *builder, OperationState &result, Value memrefOrTensor,unsigned index) {
      auto indexType = builder->getIndexType();
      auto indexAttr = builder->getIntegerAttr(indexType, index);
      build(builder, result, indexType, memrefOrTensor, indexAttr);
    
}

void DimOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value memrefOrTensor, IntegerAttr index) {
  odsState.addOperands(memrefOrTensor);
  odsState.addAttribute("index", index);
  odsState.addTypes(resultType0);
}

void DimOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memrefOrTensor, IntegerAttr index) {
  odsState.addOperands(memrefOrTensor);
  odsState.addAttribute("index", index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DimOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult DimOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseDimOp(parser, result);
}

void DimOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult DimOp::verify() {
  auto tblgen_index = this->getAttr("index");
  if (!tblgen_index) return emitOpError("requires attribute 'index'");
  {
    if (!((tblgen_index.isa<IntegerAttr>()))) return emitOpError("attribute 'index' failed to satisfy constraint: arbitrary integer attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<MemRefType>())) && ((true))) || (((v.getType().isa<TensorType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be any tensor or memref type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}



void DimOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// DivFOp definitions
//===----------------------------------------------------------------------===//

DivFOpOperandAdaptor::DivFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> DivFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value DivFOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value DivFOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef DivFOp::getOperationName() {
  return "std.divf";
}

Operation::operand_range DivFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value DivFOp::lhs() {
  return *getODSOperands(0).begin();
}

Value DivFOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range DivFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void DivFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void DivFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DivFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void DivFOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void DivFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult DivFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void DivFOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult DivFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void DivFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// Exp2Op definitions
//===----------------------------------------------------------------------===//

Exp2OpOperandAdaptor::Exp2OpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> Exp2OpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value Exp2OpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef Exp2Op::getOperationName() {
  return "std.exp2";
}

Operation::operand_range Exp2Op::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value Exp2Op::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range Exp2Op::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void Exp2Op::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void Exp2Op::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void Exp2Op::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult Exp2Op::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Exp2Op::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult Exp2Op::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void Exp2Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ExpOp definitions
//===----------------------------------------------------------------------===//

ExpOpOperandAdaptor::ExpOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ExpOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ExpOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef ExpOp::getOperationName() {
  return "std.exp";
}

Operation::operand_range ExpOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ExpOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range ExpOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void ExpOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void ExpOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void ExpOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult ExpOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void ExpOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult ExpOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void ExpOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ExtractElementOp definitions
//===----------------------------------------------------------------------===//

ExtractElementOpOperandAdaptor::ExtractElementOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ExtractElementOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value ExtractElementOpOperandAdaptor::aggregate() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> ExtractElementOpOperandAdaptor::indices() {
  return getODSOperands(1);
}

StringRef ExtractElementOp::getOperationName() {
  return "std.extract_element";
}

Operation::operand_range ExtractElementOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value ExtractElementOp::aggregate() {
  return *getODSOperands(0).begin();
}

Operation::operand_range ExtractElementOp::indices() {
  return getODSOperands(1);
}

Operation::result_range ExtractElementOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value ExtractElementOp::result() {
  return *getODSResults(0).begin();
}

void ExtractElementOp::build(Builder *builder, OperationState &result, Value aggregate,ValueRange indices ) {
      auto resType = aggregate.getType().cast<ShapedType>()
                                         .getElementType();
      build(builder, result, resType, aggregate, indices);
    
}

void ExtractElementOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value aggregate, ValueRange indices) {
  odsState.addOperands(aggregate);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void ExtractElementOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value aggregate, ValueRange indices) {
  odsState.addOperands(aggregate);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ExtractElementOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<VectorType>())) && ((true))) || (((v.getType().isa<TensorType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be vector of any type values or tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<ShapedType>().getElementType() == (*this->getODSResults(0).begin()).getType()))) {
    return emitOpError("failed to verify that result type matches element type of aggregate");
  }
  return ::verify(*this);
}



ParseResult ExtractElementOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType aggregateRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> aggregateOperands(aggregateRawOperands);  llvm::SMLoc aggregateOperandsLoc = parser.getCurrentLocation();
  (void)aggregateOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> indicesOperands;
  llvm::SMLoc indicesOperandsLoc = parser.getCurrentLocation();
  (void)indicesOperandsLoc;
  Type aggregateRawTypes[1];
  ArrayRef<Type> aggregateTypes(aggregateRawTypes);

  if (parser.parseOperand(aggregateRawOperands[0]))
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(indicesOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(aggregateRawTypes[0]))
    return failure();
  for (Type type : aggregateTypes) {
    (void)type;
    if (!((((type.isa<VectorType>())) && ((true))) || (((type.isa<TensorType>())) && ((true))))) {
      return parser.emitError(parser.getNameLoc()) << "'aggregate' must be vector of any type values or tensor of any type values, but got " << type;
    }
  }
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIndexType();
  result.addTypes(aggregateTypes[0].cast<ShapedType>().getElementType());
  if (parser.resolveOperands(aggregateOperands, aggregateTypes, aggregateOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return failure();
  return success();
}

void ExtractElementOp::print(OpAsmPrinter &p) {
  p << "extract_element";
  p << " ";
  p << aggregate();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(aggregate().getType());
}

void ExtractElementOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FPExtOp definitions
//===----------------------------------------------------------------------===//

FPExtOpOperandAdaptor::FPExtOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> FPExtOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value FPExtOpOperandAdaptor::in() {
  return *getODSOperands(0).begin();
}

StringRef FPExtOp::getOperationName() {
  return "std.fpext";
}

Operation::operand_range FPExtOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value FPExtOp::in() {
  return *getODSOperands(0).begin();
}

Operation::result_range FPExtOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void FPExtOp::build(Builder *builder, OperationState &result, Value source, Type destType) {
       impl::buildCastOp(builder, result, source, destType);
  
}

void FPExtOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void FPExtOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPExtOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult FPExtOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void FPExtOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult FPExtOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void FPExtOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// FPTruncOp definitions
//===----------------------------------------------------------------------===//

FPTruncOpOperandAdaptor::FPTruncOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> FPTruncOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value FPTruncOpOperandAdaptor::in() {
  return *getODSOperands(0).begin();
}

StringRef FPTruncOp::getOperationName() {
  return "std.fptrunc";
}

Operation::operand_range FPTruncOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value FPTruncOp::in() {
  return *getODSOperands(0).begin();
}

Operation::result_range FPTruncOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void FPTruncOp::build(Builder *builder, OperationState &result, Value source, Type destType) {
       impl::buildCastOp(builder, result, source, destType);
  
}

void FPTruncOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void FPTruncOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPTruncOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult FPTruncOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void FPTruncOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult FPTruncOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void FPTruncOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// GenericAtomicRMWOp definitions
//===----------------------------------------------------------------------===//

GenericAtomicRMWOpOperandAdaptor::GenericAtomicRMWOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> GenericAtomicRMWOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value GenericAtomicRMWOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> GenericAtomicRMWOpOperandAdaptor::indices() {
  return getODSOperands(1);
}

StringRef GenericAtomicRMWOp::getOperationName() {
  return "std.generic_atomic_rmw";
}

Operation::operand_range GenericAtomicRMWOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value GenericAtomicRMWOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::operand_range GenericAtomicRMWOp::indices() {
  return getODSOperands(1);
}

Operation::result_range GenericAtomicRMWOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value GenericAtomicRMWOp::result() {
  return *getODSResults(0).begin();
}

Region &GenericAtomicRMWOp::body() {
  return this->getOperation()->getRegion(0);
}



ParseResult GenericAtomicRMWOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseGenericAtomicRMWOp(parser, result);
}

void GenericAtomicRMWOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult GenericAtomicRMWOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && (((v.getType().cast<ShapedType>().getElementType().isSignlessInteger())) || ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be memref of signless integer or floating-point values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<FloatType>())))) {
        return emitOpError("result #") << index << " must be signless integer or floating-point, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSResults(0).begin()).getType()))) {
    return emitOpError("failed to verify that result type matches element type of memref");
  }
{
    unsigned index = 0; (void)index;
    for (Region &region : MutableArrayRef<Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((true))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: any region";
      }
      ++index;
    }
  }
  return ::verify(*this);
}


//===----------------------------------------------------------------------===//
// IndexCastOp definitions
//===----------------------------------------------------------------------===//

IndexCastOpOperandAdaptor::IndexCastOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> IndexCastOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value IndexCastOpOperandAdaptor::in() {
  return *getODSOperands(0).begin();
}

StringRef IndexCastOp::getOperationName() {
  return "std.index_cast";
}

Operation::operand_range IndexCastOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value IndexCastOp::in() {
  return *getODSOperands(0).begin();
}

Operation::result_range IndexCastOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void IndexCastOp::build(Builder *builder, OperationState &result, Value source, Type destType) {
       impl::buildCastOp(builder, result, source, destType);
  
}

void IndexCastOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void IndexCastOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IndexCastOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult IndexCastOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void IndexCastOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult IndexCastOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}



void IndexCastOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// LoadOp definitions
//===----------------------------------------------------------------------===//

LoadOpOperandAdaptor::LoadOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> LoadOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value LoadOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> LoadOpOperandAdaptor::indices() {
  return getODSOperands(1);
}

StringRef LoadOp::getOperationName() {
  return "std.load";
}

Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value LoadOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::operand_range LoadOp::indices() {
  return getODSOperands(1);
}

Operation::result_range LoadOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value LoadOp::result() {
  return *getODSResults(0).begin();
}

void LoadOp::build(Builder *, OperationState &result, Value memref,ValueRange indices ) {
      auto memrefType = memref.getType().cast<MemRefType>();
      result.addOperands(memref);
      result.addOperands(indices);
      result.types.push_back(memrefType.getElementType());
  
}

void LoadOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value memref, ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addTypes(result);
}

void LoadOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, ValueRange indices) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult LoadOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSOperands(0).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSResults(0).begin()).getType()))) {
    return emitOpError("failed to verify that result type matches element type of 'memref'");
  }
  return ::verify(*this);
}



ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> indicesOperands;
  llvm::SMLoc indicesOperandsLoc = parser.getCurrentLocation();
  (void)indicesOperandsLoc;
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(indicesOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  for (Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<MemRefType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIndexType();
  result.addTypes(memrefTypes[0].cast<MemRefType>().getElementType());
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return failure();
  return success();
}

void LoadOp::print(OpAsmPrinter &p) {
  p << "load";
  p << " ";
  p << memref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
}

void LoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// Log10Op definitions
//===----------------------------------------------------------------------===//

Log10OpOperandAdaptor::Log10OpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> Log10OpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value Log10OpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef Log10Op::getOperationName() {
  return "std.log10";
}

Operation::operand_range Log10Op::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value Log10Op::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range Log10Op::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void Log10Op::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void Log10Op::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void Log10Op::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult Log10Op::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Log10Op::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult Log10Op::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void Log10Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// Log2Op definitions
//===----------------------------------------------------------------------===//

Log2OpOperandAdaptor::Log2OpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> Log2OpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value Log2OpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef Log2Op::getOperationName() {
  return "std.log2";
}

Operation::operand_range Log2Op::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value Log2Op::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range Log2Op::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void Log2Op::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void Log2Op::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void Log2Op::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult Log2Op::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void Log2Op::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult Log2Op::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void Log2Op::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// LogOp definitions
//===----------------------------------------------------------------------===//

LogOpOperandAdaptor::LogOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> LogOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value LogOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef LogOp::getOperationName() {
  return "std.log";
}

Operation::operand_range LogOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value LogOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range LogOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void LogOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void LogOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LogOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void LogOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult LogOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void LogOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult LogOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void LogOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// MemRefCastOp definitions
//===----------------------------------------------------------------------===//

MemRefCastOpOperandAdaptor::MemRefCastOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> MemRefCastOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value MemRefCastOpOperandAdaptor::source() {
  return *getODSOperands(0).begin();
}

StringRef MemRefCastOp::getOperationName() {
  return "std.memref_cast";
}

Operation::operand_range MemRefCastOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value MemRefCastOp::source() {
  return *getODSOperands(0).begin();
}

Operation::result_range MemRefCastOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void MemRefCastOp::build(Builder *builder, OperationState &result, Value source, Type destType) {
       impl::buildCastOp(builder, result, source, destType);
  
}

void MemRefCastOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value source) {
  odsState.addOperands(source);
  odsState.addTypes(resultType0);
}

void MemRefCastOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value source) {
  odsState.addOperands(source);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemRefCastOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult MemRefCastOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void MemRefCastOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult MemRefCastOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<MemRefType>())) && ((true))))) {
        return emitOpError("operand #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<UnrankedMemRefType>())) && ((true))) || (((v.getType().isa<MemRefType>())) && ((true))))) {
        return emitOpError("result #") << index << " must be unranked.memref of any type values or memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}



void MemRefCastOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// MulFOp definitions
//===----------------------------------------------------------------------===//

MulFOpOperandAdaptor::MulFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> MulFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value MulFOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value MulFOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef MulFOp::getOperationName() {
  return "std.mulf";
}

Operation::operand_range MulFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value MulFOp::lhs() {
  return *getODSOperands(0).begin();
}

Value MulFOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range MulFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void MulFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulFOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void MulFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult MulFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void MulFOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult MulFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void MulFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// MulIOp definitions
//===----------------------------------------------------------------------===//

MulIOpOperandAdaptor::MulIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> MulIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value MulIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value MulIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef MulIOp::getOperationName() {
  return "std.muli";
}

Operation::operand_range MulIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value MulIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value MulIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range MulIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void MulIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void MulIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult MulIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void MulIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult MulIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void MulIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// NegFOp definitions
//===----------------------------------------------------------------------===//

NegFOpOperandAdaptor::NegFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> NegFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value NegFOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef NegFOp::getOperationName() {
  return "std.negf";
}

Operation::operand_range NegFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value NegFOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range NegFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void NegFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void NegFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NegFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void NegFOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void NegFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult NegFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void NegFOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult NegFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void NegFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// OrOp definitions
//===----------------------------------------------------------------------===//

OrOpOperandAdaptor::OrOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> OrOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value OrOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value OrOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef OrOp::getOperationName() {
  return "std.or";
}

Operation::operand_range OrOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value OrOp::lhs() {
  return *getODSOperands(0).begin();
}

Value OrOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range OrOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void OrOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void OrOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void OrOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void OrOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult OrOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void OrOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult OrOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void OrOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// PrefetchOp definitions
//===----------------------------------------------------------------------===//

PrefetchOpOperandAdaptor::PrefetchOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> PrefetchOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value PrefetchOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> PrefetchOpOperandAdaptor::indices() {
  return getODSOperands(1);
}

StringRef PrefetchOp::getOperationName() {
  return "std.prefetch";
}

Operation::operand_range PrefetchOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value PrefetchOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::operand_range PrefetchOp::indices() {
  return getODSOperands(1);
}

Operation::result_range PrefetchOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

BoolAttr PrefetchOp::isWriteAttr() {
  return this->getAttr("isWrite").cast<BoolAttr>();
}

bool PrefetchOp::isWrite() {
  auto attr = isWriteAttr();
  return attr.getValue();
}

IntegerAttr PrefetchOp::localityHintAttr() {
  return this->getAttr("localityHint").cast<IntegerAttr>();
}

APInt PrefetchOp::localityHint() {
  auto attr = localityHintAttr();
  return attr.getValue();
}

BoolAttr PrefetchOp::isDataCacheAttr() {
  return this->getAttr("isDataCache").cast<BoolAttr>();
}

bool PrefetchOp::isDataCache() {
  auto attr = isDataCacheAttr();
  return attr.getValue();
}

void PrefetchOp::isWriteAttr(BoolAttr attr) {
  this->getOperation()->setAttr("isWrite", attr);
}

void PrefetchOp::localityHintAttr(IntegerAttr attr) {
  this->getOperation()->setAttr("localityHint", attr);
}

void PrefetchOp::isDataCacheAttr(BoolAttr attr) {
  this->getOperation()->setAttr("isDataCache", attr);
}

void PrefetchOp::build(Builder *builder, OperationState &result, Value memref,ArrayRef<Value> indices, bool isWrite, unsigned hint, bool isData) {
      auto hintAttr = builder->getI32IntegerAttr(hint);
      auto isWriteAttr = builder->getBoolAttr(isWrite);
      auto isDataCacheAttr = builder->getBoolAttr(isData);
      result.addOperands(memref);
      result.addOperands(indices);
      result.addAttribute("localityHint", hintAttr);
      result.addAttribute("isWrite", isWriteAttr);
      result.addAttribute("isDataCache", isDataCacheAttr);
    
}

void PrefetchOp::build(Builder *odsBuilder, OperationState &odsState, Value memref, ValueRange indices, BoolAttr isWrite, IntegerAttr localityHint, BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", isWrite);
  odsState.addAttribute("localityHint", localityHint);
  odsState.addAttribute("isDataCache", isDataCache);
}

void PrefetchOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, ValueRange indices, BoolAttr isWrite, IntegerAttr localityHint, BoolAttr isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", isWrite);
  odsState.addAttribute("localityHint", localityHint);
  odsState.addAttribute("isDataCache", isDataCache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrefetchOp::build(Builder *odsBuilder, OperationState &odsState, Value memref, ValueRange indices, bool isWrite, APInt localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", (*odsBuilder).getBoolAttr(isWrite));
  odsState.addAttribute("localityHint", (*odsBuilder).getIntegerAttr((*odsBuilder).getIntegerType(32), localityHint));
  odsState.addAttribute("isDataCache", (*odsBuilder).getBoolAttr(isDataCache));
}

void PrefetchOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref, ValueRange indices, bool isWrite, APInt localityHint, bool isDataCache) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  odsState.addAttribute("isWrite", (*odsBuilder).getBoolAttr(isWrite));
  odsState.addAttribute("localityHint", (*odsBuilder).getIntegerAttr((*odsBuilder).getIntegerType(32), localityHint));
  odsState.addAttribute("isDataCache", (*odsBuilder).getBoolAttr(isDataCache));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrefetchOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult PrefetchOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parsePrefetchOp(parser, result);
}

void PrefetchOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult PrefetchOp::verify() {
  auto tblgen_isWrite = this->getAttr("isWrite");
  if (!tblgen_isWrite) return emitOpError("requires attribute 'isWrite'");
  {
    if (!((tblgen_isWrite.isa<BoolAttr>()))) return emitOpError("attribute 'isWrite' failed to satisfy constraint: bool attribute");
  }
  auto tblgen_localityHint = this->getAttr("localityHint");
  if (!tblgen_localityHint) return emitOpError("requires attribute 'localityHint'");
  {
    if (!((((tblgen_localityHint.isa<IntegerAttr>())) && ((tblgen_localityHint.cast<IntegerAttr>().getType().isSignlessInteger(32)))) && ((tblgen_localityHint.cast<IntegerAttr>().getInt() >= 0)) && ((tblgen_localityHint.cast<IntegerAttr>().getInt() <= 3)))) return emitOpError("attribute 'localityHint' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 3");
  }
  auto tblgen_isDataCache = this->getAttr("isDataCache");
  if (!tblgen_isDataCache) return emitOpError("requires attribute 'isDataCache'");
  {
    if (!((tblgen_isDataCache.isa<BoolAttr>()))) return emitOpError("attribute 'isDataCache' failed to satisfy constraint: bool attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}




//===----------------------------------------------------------------------===//
// RankOp definitions
//===----------------------------------------------------------------------===//

RankOpOperandAdaptor::RankOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> RankOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

StringRef RankOp::getOperationName() {
  return "std.rank";
}

Operation::operand_range RankOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Operation::result_range RankOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void RankOp::build(Builder *builder, OperationState &result, Value tensor) {
      auto indexType = builder->getIndexType();
      build(builder, result, indexType, tensor);
    
}

void RankOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(resultType0);
}

void RankOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RankOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult RankOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("result #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



ParseResult RankOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<Type, 1> allOperandTypes;
  llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::OperandType, 4> allOperands;
  if (parser.parseOperandList(allOperands))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(allOperandTypes))
    return failure();
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return failure();
  return success();
}

void RankOp::print(OpAsmPrinter &p) {
  p << "rank";
  p << " ";
  p << getOperation()->getOperands();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << getOperation()->getOperandTypes();
}

void RankOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// RemFOp definitions
//===----------------------------------------------------------------------===//

RemFOpOperandAdaptor::RemFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> RemFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value RemFOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value RemFOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef RemFOp::getOperationName() {
  return "std.remf";
}

Operation::operand_range RemFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value RemFOp::lhs() {
  return *getODSOperands(0).begin();
}

Value RemFOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range RemFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void RemFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void RemFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RemFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RemFOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void RemFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult RemFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void RemFOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult RemFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void RemFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpOperandAdaptor::ReturnOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ReturnOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

ArrayRef<Value> ReturnOpOperandAdaptor::operands() {
  return getODSOperands(0);
}

StringRef ReturnOp::getOperationName() {
  return "std.return";
}

Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Operation::operand_range ReturnOp::operands() {
  return getODSOperands(0);
}

Operation::result_range ReturnOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void ReturnOp::build(Builder *b, OperationState &result) {
 build(b, result, llvm::None); 
}

void ReturnOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands) {
  odsState.addOperands(operands);
}

void ReturnOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands) {
  odsState.addOperands(operands);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReturnOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ReturnOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::verify(*this);
}

ParseResult ReturnOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 4> operandsOperands;
  llvm::SMLoc operandsOperandsLoc = parser.getCurrentLocation();
  (void)operandsOperandsLoc;
  SmallVector<Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseOperandList(operandsOperands))
    return failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(operandsTypes))
    return failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return failure();
  return success();
}

void ReturnOp::print(OpAsmPrinter &p) {
  p << "return";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  p << " ";
  p << operands();
  p << " " << ":";
  p << " ";
  p << operands().getTypes();
  }
}

void ReturnOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// RsqrtOp definitions
//===----------------------------------------------------------------------===//

RsqrtOpOperandAdaptor::RsqrtOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> RsqrtOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value RsqrtOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef RsqrtOp::getOperationName() {
  return "std.rsqrt";
}

Operation::operand_range RsqrtOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value RsqrtOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range RsqrtOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void RsqrtOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void RsqrtOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RsqrtOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void RsqrtOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult RsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void RsqrtOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult RsqrtOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void RsqrtOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SIToFPOp definitions
//===----------------------------------------------------------------------===//

SIToFPOpOperandAdaptor::SIToFPOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SIToFPOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SIToFPOpOperandAdaptor::in() {
  return *getODSOperands(0).begin();
}

StringRef SIToFPOp::getOperationName() {
  return "std.sitofp";
}

Operation::operand_range SIToFPOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SIToFPOp::in() {
  return *getODSOperands(0).begin();
}

Operation::result_range SIToFPOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SIToFPOp::build(Builder *builder, OperationState &result, Value source, Type destType) {
       impl::buildCastOp(builder, result, source, destType);
  
}

void SIToFPOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value in) {
  odsState.addOperands(in);
  odsState.addTypes(resultType0);
}

void SIToFPOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SIToFPOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult SIToFPOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void SIToFPOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult SIToFPOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}

void SIToFPOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SelectOp definitions
//===----------------------------------------------------------------------===//

SelectOpOperandAdaptor::SelectOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SelectOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SelectOpOperandAdaptor::condition() {
  return *getODSOperands(0).begin();
}

Value SelectOpOperandAdaptor::true_value() {
  return *getODSOperands(1).begin();
}

Value SelectOpOperandAdaptor::false_value() {
  return *getODSOperands(2).begin();
}

StringRef SelectOp::getOperationName() {
  return "std.select";
}

Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SelectOp::condition() {
  return *getODSOperands(0).begin();
}

Value SelectOp::true_value() {
  return *getODSOperands(1).begin();
}

Value SelectOp::false_value() {
  return *getODSOperands(2).begin();
}

Operation::result_range SelectOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value SelectOp::result() {
  return *getODSResults(0).begin();
}

void SelectOp::build(Builder *builder, OperationState &result, Value condition,Value trueValue, Value falseValue) {
      result.addOperands({condition, trueValue, falseValue});
      result.addTypes(trueValue.getType());
  
}

void SelectOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value condition, Value true_value, Value false_value) {
  odsState.addOperands(condition);
  odsState.addOperands(true_value);
  odsState.addOperands(false_value);
  odsState.addTypes(result);
}

void SelectOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value condition, Value true_value, Value false_value) {
  odsState.addOperands(condition);
  odsState.addOperands(true_value);
  odsState.addOperands(false_value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult SelectOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseSelectOp(parser, result);
}

void SelectOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult SelectOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger(1))) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(1)))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(1)))))) {
        return emitOpError("operand #") << index << " must be bool-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((llvm::is_splat(llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()}))))) {
    return emitOpError("failed to verify that all of {true_value, false_value, result} have same type");
  }
  return ::verify(*this);
}



void SelectOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ShiftLeftOp definitions
//===----------------------------------------------------------------------===//

ShiftLeftOpOperandAdaptor::ShiftLeftOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ShiftLeftOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ShiftLeftOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value ShiftLeftOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef ShiftLeftOp::getOperationName() {
  return "std.shift_left";
}

Operation::operand_range ShiftLeftOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ShiftLeftOp::lhs() {
  return *getODSOperands(0).begin();
}

Value ShiftLeftOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range ShiftLeftOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void ShiftLeftOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void ShiftLeftOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShiftLeftOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShiftLeftOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void ShiftLeftOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult ShiftLeftOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void ShiftLeftOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult ShiftLeftOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void ShiftLeftOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignExtendIOp definitions
//===----------------------------------------------------------------------===//

SignExtendIOpOperandAdaptor::SignExtendIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SignExtendIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SignExtendIOpOperandAdaptor::value() {
  return *getODSOperands(0).begin();
}

StringRef SignExtendIOp::getOperationName() {
  return "std.sexti";
}

Operation::operand_range SignExtendIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SignExtendIOp::value() {
  return *getODSOperands(0).begin();
}

Operation::result_range SignExtendIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SignExtendIOp::build(Builder *builder, OperationState &result, Value value, Type destType) {
      result.addOperands(value);
      result.addTypes(destType);
  
}

void SignExtendIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value value) {
  odsState.addOperands(value);
  odsState.addTypes(resultType0);
}

void SignExtendIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignExtendIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult SignExtendIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void SignExtendIOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult SignExtendIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("result #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void SignExtendIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignedDivIOp definitions
//===----------------------------------------------------------------------===//

SignedDivIOpOperandAdaptor::SignedDivIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SignedDivIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SignedDivIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value SignedDivIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef SignedDivIOp::getOperationName() {
  return "std.divi_signed";
}

Operation::operand_range SignedDivIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SignedDivIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value SignedDivIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range SignedDivIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignedDivIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult SignedDivIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SignedDivIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult SignedDivIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void SignedDivIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignedRemIOp definitions
//===----------------------------------------------------------------------===//

SignedRemIOpOperandAdaptor::SignedRemIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SignedRemIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SignedRemIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value SignedRemIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef SignedRemIOp::getOperationName() {
  return "std.remi_signed";
}

Operation::operand_range SignedRemIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SignedRemIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value SignedRemIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range SignedRemIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignedRemIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult SignedRemIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SignedRemIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult SignedRemIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void SignedRemIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SignedShiftRightOp definitions
//===----------------------------------------------------------------------===//

SignedShiftRightOpOperandAdaptor::SignedShiftRightOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SignedShiftRightOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SignedShiftRightOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value SignedShiftRightOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef SignedShiftRightOp::getOperationName() {
  return "std.shift_right_signed";
}

Operation::operand_range SignedShiftRightOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SignedShiftRightOp::lhs() {
  return *getODSOperands(0).begin();
}

Value SignedShiftRightOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range SignedShiftRightOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SignedShiftRightOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult SignedShiftRightOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SignedShiftRightOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult SignedShiftRightOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void SignedShiftRightOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SplatOp definitions
//===----------------------------------------------------------------------===//

SplatOpOperandAdaptor::SplatOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SplatOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SplatOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef SplatOp::getOperationName() {
  return "std.splat";
}

Operation::operand_range SplatOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SplatOp::input() {
  return *getODSOperands(0).begin();
}

Operation::result_range SplatOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value SplatOp::aggregate() {
  return *getODSResults(0).begin();
}

void SplatOp::build(Builder *builder, OperationState &result, Value element, Type aggregateType) {
 build(builder, result, aggregateType, element); 
}

void SplatOp::build(Builder *odsBuilder, OperationState &odsState, Type aggregate, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(aggregate);
}

void SplatOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplatOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult SplatOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<FloatType>())))) {
        return emitOpError("operand #") << index << " must be integer or float type, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<VectorType>())) && ((true))) || ((((v.getType().isa<TensorType>())) && ((true))) && ((v.getType().cast<ShapedType>().hasStaticShape()))))) {
        return emitOpError("result #") << index << " must be vector of any type values or statically shaped tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!(((*this->getODSResults(0).begin()).getType().cast<ShapedType>().getElementType() == (*this->getODSOperands(0).begin()).getType()))) {
    return emitOpError("failed to verify that operand type matches element type of result");
  }
  return ::verify(*this);
}



ParseResult SplatOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType inputRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> inputOperands(inputRawOperands);  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  (void)inputOperandsLoc;
  Type aggregateRawTypes[1];
  ArrayRef<Type> aggregateTypes(aggregateRawTypes);

  if (parser.parseOperand(inputRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(aggregateRawTypes[0]))
    return failure();
  for (Type type : aggregateTypes) {
    (void)type;
    if (!((((type.isa<VectorType>())) && ((true))) || ((((type.isa<TensorType>())) && ((true))) && ((type.cast<ShapedType>().hasStaticShape()))))) {
      return parser.emitError(parser.getNameLoc()) << "'aggregate' must be vector of any type values or statically shaped tensor of any type values, but got " << type;
    }
  }
  result.addTypes(aggregateTypes);
  if (parser.resolveOperands(inputOperands, aggregateTypes[0].cast<ShapedType>().getElementType(), inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void SplatOp::print(OpAsmPrinter &p) {
  p << "splat";
  p << " ";
  p << input();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(aggregate().getType());
}

void SplatOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SqrtOp definitions
//===----------------------------------------------------------------------===//

SqrtOpOperandAdaptor::SqrtOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SqrtOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SqrtOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef SqrtOp::getOperationName() {
  return "std.sqrt";
}

Operation::operand_range SqrtOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SqrtOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range SqrtOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SqrtOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void SqrtOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void SqrtOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult SqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SqrtOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult SqrtOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void SqrtOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// StoreOp definitions
//===----------------------------------------------------------------------===//

StoreOpOperandAdaptor::StoreOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> StoreOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value StoreOpOperandAdaptor::value() {
  return *getODSOperands(0).begin();
}

Value StoreOpOperandAdaptor::memref() {
  return *getODSOperands(1).begin();
}

ArrayRef<Value> StoreOpOperandAdaptor::indices() {
  return getODSOperands(2);
}

StringRef StoreOp::getOperationName() {
  return "std.store";
}

Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value StoreOp::value() {
  return *getODSOperands(0).begin();
}

Value StoreOp::memref() {
  return *getODSOperands(1).begin();
}

Operation::operand_range StoreOp::indices() {
  return getODSOperands(2);
}

Operation::result_range StoreOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void StoreOp::build(Builder *, OperationState &result, Value valueToStore, Value memref) {
      result.addOperands(valueToStore);
      result.addOperands(memref);
  
}

void StoreOp::build(Builder *odsBuilder, OperationState &odsState, Value value, Value memref, ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
}

void StoreOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value value, Value memref, ValueRange indices) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult StoreOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("operand #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!(((*this->getODSOperands(1).begin()).getType().cast<MemRefType>().getElementType() == (*this->getODSOperands(0).begin()).getType()))) {
    return emitOpError("failed to verify that type of 'value' matches element type of 'memref'");
  }
  return ::verify(*this);
}



ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType valueRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> valueOperands(valueRawOperands);  llvm::SMLoc valueOperandsLoc = parser.getCurrentLocation();
  (void)valueOperandsLoc;
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> indicesOperands;
  llvm::SMLoc indicesOperandsLoc = parser.getCurrentLocation();
  (void)indicesOperandsLoc;
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);

  if (parser.parseOperand(valueRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(indicesOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  for (Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<MemRefType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIndexType();
  if (parser.resolveOperands(valueOperands, memrefTypes[0].cast<MemRefType>().getElementType(), valueOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(indicesOperands, odsBuildableType0, result.operands))
    return failure();
  return success();
}

void StoreOp::print(OpAsmPrinter &p) {
  p << "store";
  p << " ";
  p << value();
  p << ",";
  p << " ";
  p << memref();
  p << "[";
  p << indices();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
}

void StoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// SubFOp definitions
//===----------------------------------------------------------------------===//

SubFOpOperandAdaptor::SubFOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SubFOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SubFOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value SubFOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef SubFOp::getOperationName() {
  return "std.subf";
}

Operation::operand_range SubFOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SubFOp::lhs() {
  return *getODSOperands(0).begin();
}

Value SubFOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range SubFOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SubFOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SubFOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubFOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubFOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SubFOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult SubFOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SubFOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult SubFOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void SubFOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubIOp definitions
//===----------------------------------------------------------------------===//

SubIOpOperandAdaptor::SubIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SubIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SubIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value SubIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef SubIOp::getOperationName() {
  return "std.subi";
}

Operation::operand_range SubIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SubIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value SubIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range SubIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void SubIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void SubIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void SubIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult SubIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void SubIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult SubIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void SubIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// SubViewOp definitions
//===----------------------------------------------------------------------===//

StringRef SubViewOp::getOperationName() {
  return "std.subview";
}

Operation::operand_range SubViewOp::getODSOperands(unsigned index) {
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += (*(sizeAttr.begin() + i)).getZExtValue();
  unsigned end = start + (*(sizeAttr.begin() + index)).getZExtValue();
  return {std::next(getOperation()->operand_begin(), start), std::next(getOperation()->operand_begin(), end)};
}

Value SubViewOp::source() {
  return *getODSOperands(0).begin();
}

Operation::operand_range SubViewOp::offsets() {
  return getODSOperands(1);
}

Operation::operand_range SubViewOp::sizes() {
  return getODSOperands(2);
}

Operation::operand_range SubViewOp::strides() {
  return getODSOperands(3);
}

Operation::result_range SubViewOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value SubViewOp::result() {
  return *getODSResults(0).begin();
}





void SubViewOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder->getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  odsState.addTypes(result);
}

void SubViewOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides) {
  odsState.addOperands(source);
  odsState.addOperands(offsets);
  odsState.addOperands(sizes);
  odsState.addOperands(strides);
  odsState.addAttribute("operand_segment_sizes", odsBuilder->getI32VectorAttr({1, static_cast<int32_t>(offsets.size()), static_cast<int32_t>(sizes.size()), static_cast<int32_t>(strides.size())}));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubViewOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult SubViewOp::verify() {
  auto sizeAttr = getAttrOfType<DenseIntElementsAttr>("operand_segment_sizes");
  auto numElements = sizeAttr.getType().cast<ShapedType>().getNumElements();
  if (numElements != 4) {
    return emitOpError("'operand_segment_sizes' attribute for specifying operand segments "
                       "must have 4 elements");
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup2 = getODSOperands(2);
    for (Value v : valueGroup2) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup3 = getODSOperands(3);
    for (Value v : valueGroup3) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





ParseResult SubViewOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType sourceRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> sourceOperands(sourceRawOperands);  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  (void)sourceOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> offsetsOperands;
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  (void)offsetsOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> sizesOperands;
  llvm::SMLoc sizesOperandsLoc = parser.getCurrentLocation();
  (void)sizesOperandsLoc;
  SmallVector<OpAsmParser::OperandType, 4> stridesOperands;
  llvm::SMLoc stridesOperandsLoc = parser.getCurrentLocation();
  (void)stridesOperandsLoc;
  Type sourceRawTypes[1];
  ArrayRef<Type> sourceTypes(sourceRawTypes);
  Type resultRawTypes[1];
  ArrayRef<Type> resultTypes(resultRawTypes);

  if (parser.parseOperand(sourceRawOperands[0]))
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(offsetsOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(sizesOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseLSquare())
    return failure();

  if (parser.parseOperandList(stridesOperands))
    return failure();
  if (parser.parseRSquare())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(sourceRawTypes[0]))
    return failure();
  if (parser.parseKeyword("to"))
    return failure();

  if (parser.parseType(resultRawTypes[0]))
    return failure();
  Builder &builder = parser.getBuilder();
  Type odsBuildableType0 = builder.getIndexType();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(offsetsOperands, odsBuildableType0, result.operands))
    return failure();
  if (parser.resolveOperands(sizesOperands, odsBuildableType0, result.operands))
    return failure();
  if (parser.resolveOperands(stridesOperands, odsBuildableType0, result.operands))
    return failure();
  result.addAttribute("operand_segment_sizes", builder.getI32VectorAttr({1, static_cast<int32_t>(offsetsOperands.size()), static_cast<int32_t>(sizesOperands.size()), static_cast<int32_t>(stridesOperands.size())}));
  return success();
}

void SubViewOp::print(OpAsmPrinter &p) {
  p << "subview";
  p << " ";
  p << source();
  p << "[";
  p << offsets();
  p << "]";
  p << " " << "[";
  p << sizes();
  p << "]";
  p << " " << "[";
  p << strides();
  p << "]";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(source().getType());
  p << " " << "to";
  p << " ";
  p << ArrayRef<Type>(result().getType());
}

void SubViewOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TanhOp definitions
//===----------------------------------------------------------------------===//

TanhOpOperandAdaptor::TanhOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> TanhOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value TanhOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

StringRef TanhOp::getOperationName() {
  return "std.tanh";
}

Operation::operand_range TanhOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value TanhOp::operand() {
  return *getODSOperands(0).begin();
}

Operation::result_range TanhOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void TanhOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes(resultType0);
}

void TanhOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand) {
  odsState.addOperands(operand);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TanhOp::build(Builder *odsBuilder, OperationState &odsState, Value operand) {
  odsState.addOperands(operand);
  odsState.addTypes({operand.getType()});

}

void TanhOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult TanhOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void TanhOp::print(OpAsmPrinter &p) {
  return printStandardUnaryOp(this->getOperation(), p);
}

LogicalResult TanhOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<FloatType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<FloatType>()))))) {
        return emitOpError("operand #") << index << " must be floating-point-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void TanhOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TensorCastOp definitions
//===----------------------------------------------------------------------===//

TensorCastOpOperandAdaptor::TensorCastOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> TensorCastOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

StringRef TensorCastOp::getOperationName() {
  return "std.tensor_cast";
}

Operation::operand_range TensorCastOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Operation::result_range TensorCastOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void TensorCastOp::build(Builder *builder, OperationState &result, Value source, Type destType) {
       impl::buildCastOp(builder, result, source, destType);
  
}

void TensorCastOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(resultType0);
}

void TensorCastOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorCastOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult TensorCastOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void TensorCastOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult TensorCastOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verifyCastOp(*this);
}



void TensorCastOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// TensorLoadOp definitions
//===----------------------------------------------------------------------===//

TensorLoadOpOperandAdaptor::TensorLoadOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> TensorLoadOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value TensorLoadOpOperandAdaptor::memref() {
  return *getODSOperands(0).begin();
}

StringRef TensorLoadOp::getOperationName() {
  return "std.tensor_load";
}

Operation::operand_range TensorLoadOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value TensorLoadOp::memref() {
  return *getODSOperands(0).begin();
}

Operation::result_range TensorLoadOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value TensorLoadOp::result() {
  return *getODSResults(0).begin();
}

void TensorLoadOp::build(Builder *builder, OperationState &result, Value memref) {
      auto memrefType = memref.getType().cast<MemRefType>();
      auto resultType = RankedTensorType::get(memrefType.getShape(),
                                              memrefType.getElementType());
      result.addOperands(memref);
      result.addTypes(resultType);
  
}

void TensorLoadOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value memref) {
  odsState.addOperands(memref);
  odsState.addTypes(result);
}

void TensorLoadOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value memref) {
  odsState.addOperands(memref);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorLoadOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult TensorLoadOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  if (!((getTensorTypeFromMemRefType((*this->getODSOperands(0).begin()).getType()) == (*this->getODSResults(0).begin()).getType()))) {
    return emitOpError("failed to verify that result type matches tensor equivalent of 'memref'");
  }
  return mlir::success();
}

ParseResult TensorLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  for (Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<MemRefType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  result.addTypes(getTensorTypeFromMemRefType(memrefTypes[0]));
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  return success();
}

void TensorLoadOp::print(OpAsmPrinter &p) {
  p << "tensor_load";
  p << " ";
  p << memref();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
}

void TensorLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (Value value : getODSOperands(0))
    effects.emplace_back(MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// TensorStoreOp definitions
//===----------------------------------------------------------------------===//

TensorStoreOpOperandAdaptor::TensorStoreOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> TensorStoreOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value TensorStoreOpOperandAdaptor::tensor() {
  return *getODSOperands(0).begin();
}

Value TensorStoreOpOperandAdaptor::memref() {
  return *getODSOperands(1).begin();
}

StringRef TensorStoreOp::getOperationName() {
  return "std.tensor_store";
}

Operation::operand_range TensorStoreOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value TensorStoreOp::tensor() {
  return *getODSOperands(0).begin();
}

Value TensorStoreOp::memref() {
  return *getODSOperands(1).begin();
}

Operation::result_range TensorStoreOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void TensorStoreOp::build(Builder *odsBuilder, OperationState &odsState, Value tensor, Value memref) {
  odsState.addOperands(tensor);
  odsState.addOperands(memref);
}

void TensorStoreOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value tensor, Value memref) {
  odsState.addOperands(tensor);
  odsState.addOperands(memref);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TensorStoreOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult TensorStoreOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be tensor of any type values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("operand #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  if (!((getTensorTypeFromMemRefType((*this->getODSOperands(1).begin()).getType()) == (*this->getODSOperands(0).begin()).getType()))) {
    return emitOpError("failed to verify that type of 'value' matches tensor equivalent of 'memref'");
  }
  return mlir::success();
}

ParseResult TensorStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType tensorRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> tensorOperands(tensorRawOperands);  llvm::SMLoc tensorOperandsLoc = parser.getCurrentLocation();
  (void)tensorOperandsLoc;
  OpAsmParser::OperandType memrefRawOperands[1];
  ArrayRef<OpAsmParser::OperandType> memrefOperands(memrefRawOperands);  llvm::SMLoc memrefOperandsLoc = parser.getCurrentLocation();
  (void)memrefOperandsLoc;
  Type memrefRawTypes[1];
  ArrayRef<Type> memrefTypes(memrefRawTypes);

  if (parser.parseOperand(tensorRawOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(memrefRawOperands[0]))
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseType(memrefRawTypes[0]))
    return failure();
  for (Type type : memrefTypes) {
    (void)type;
    if (!(((type.isa<MemRefType>())) && ((true)))) {
      return parser.emitError(parser.getNameLoc()) << "'memref' must be memref of any type values, but got " << type;
    }
  }
  if (parser.resolveOperands(tensorOperands, getTensorTypeFromMemRefType(memrefTypes[0]), tensorOperandsLoc, result.operands))
    return failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return failure();
  return success();
}

void TensorStoreOp::print(OpAsmPrinter &p) {
  p << "tensor_store";
  p << " ";
  p << tensor();
  p << ",";
  p << " ";
  p << memref();
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  p << " " << ":";
  p << " ";
  p << ArrayRef<Type>(memref().getType());
}

void TensorStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (Value value : getODSOperands(1))
    effects.emplace_back(MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}


//===----------------------------------------------------------------------===//
// TruncateIOp definitions
//===----------------------------------------------------------------------===//

TruncateIOpOperandAdaptor::TruncateIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> TruncateIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value TruncateIOpOperandAdaptor::value() {
  return *getODSOperands(0).begin();
}

StringRef TruncateIOp::getOperationName() {
  return "std.trunci";
}

Operation::operand_range TruncateIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value TruncateIOp::value() {
  return *getODSOperands(0).begin();
}

Operation::result_range TruncateIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void TruncateIOp::build(Builder *builder, OperationState &result, Value value, Type destType) {
      result.addOperands(value);
      result.addTypes(destType);
  
}

void TruncateIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value value) {
  odsState.addOperands(value);
  odsState.addTypes(resultType0);
}

void TruncateIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TruncateIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult TruncateIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void TruncateIOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult TruncateIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("result #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void TruncateIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UnsignedDivIOp definitions
//===----------------------------------------------------------------------===//

UnsignedDivIOpOperandAdaptor::UnsignedDivIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> UnsignedDivIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value UnsignedDivIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value UnsignedDivIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef UnsignedDivIOp::getOperationName() {
  return "std.divi_unsigned";
}

Operation::operand_range UnsignedDivIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value UnsignedDivIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value UnsignedDivIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range UnsignedDivIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void UnsignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void UnsignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnsignedDivIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnsignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UnsignedDivIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult UnsignedDivIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void UnsignedDivIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult UnsignedDivIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void UnsignedDivIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UnsignedRemIOp definitions
//===----------------------------------------------------------------------===//

UnsignedRemIOpOperandAdaptor::UnsignedRemIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> UnsignedRemIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value UnsignedRemIOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value UnsignedRemIOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef UnsignedRemIOp::getOperationName() {
  return "std.remi_unsigned";
}

Operation::operand_range UnsignedRemIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value UnsignedRemIOp::lhs() {
  return *getODSOperands(0).begin();
}

Value UnsignedRemIOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range UnsignedRemIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void UnsignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void UnsignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnsignedRemIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnsignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UnsignedRemIOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult UnsignedRemIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void UnsignedRemIOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult UnsignedRemIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void UnsignedRemIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// UnsignedShiftRightOp definitions
//===----------------------------------------------------------------------===//

UnsignedShiftRightOpOperandAdaptor::UnsignedShiftRightOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> UnsignedShiftRightOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value UnsignedShiftRightOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value UnsignedShiftRightOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef UnsignedShiftRightOp::getOperationName() {
  return "std.shift_right_unsigned";
}

Operation::operand_range UnsignedShiftRightOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value UnsignedShiftRightOp::lhs() {
  return *getODSOperands(0).begin();
}

Value UnsignedShiftRightOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range UnsignedShiftRightOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void UnsignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void UnsignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnsignedShiftRightOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UnsignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void UnsignedShiftRightOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult UnsignedShiftRightOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void UnsignedShiftRightOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult UnsignedShiftRightOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}

void UnsignedShiftRightOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ViewOp definitions
//===----------------------------------------------------------------------===//

ViewOpOperandAdaptor::ViewOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ViewOpOperandAdaptor::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (tblgen_operands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(tblgen_operands.begin(), offset), std::next(tblgen_operands.begin(), offset + size)};
}

Value ViewOpOperandAdaptor::source() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> ViewOpOperandAdaptor::operands() {
  return getODSOperands(1);
}

StringRef ViewOp::getOperationName() {
  return "std.view";
}

Operation::operand_range ViewOp::getODSOperands(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int offset = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;

  return {std::next(getOperation()->operand_begin(), offset), std::next(getOperation()->operand_begin(), offset + size)};
}

Value ViewOp::source() {
  return *getODSOperands(0).begin();
}

Operation::operand_range ViewOp::operands() {
  return getODSOperands(1);
}

Operation::result_range ViewOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void ViewOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value source, ValueRange operands) {
  odsState.addOperands(source);
  odsState.addOperands(operands);
  odsState.addTypes(resultType0);
}

void ViewOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value source, ValueRange operands) {
  odsState.addOperands(source);
  odsState.addOperands(operands);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ViewOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult ViewOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseViewOp(parser, result);
}

void ViewOp::print(OpAsmPrinter &p) {
  return ::print(p, *this);
}

LogicalResult ViewOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<MemRefType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(8)))) && (((v.getType().cast<ShapedType>().hasRank())) && ((v.getType().cast<ShapedType>().getRank() == 1))))) {
        return emitOpError("operand #") << index << " must be 1D memref of 8-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<IndexType>()))) {
        return emitOpError("operand #") << index << " must be index, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<MemRefType>())) && ((true)))) {
        return emitOpError("result #") << index << " must be memref of any type values, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}





void ViewOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// XOrOp definitions
//===----------------------------------------------------------------------===//

XOrOpOperandAdaptor::XOrOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> XOrOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value XOrOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value XOrOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef XOrOp::getOperationName() {
  return "std.xor";
}

Operation::operand_range XOrOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value XOrOp::lhs() {
  return *getODSOperands(0).begin();
}

Value XOrOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range XOrOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void XOrOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void XOrOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void XOrOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

ParseResult XOrOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseOneResultSameOperandTypeOp(parser, result);
}

void XOrOp::print(OpAsmPrinter &p) {
  return printStandardBinaryOp(this->getOperation(), p);
}

LogicalResult XOrOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((true))) {
        return emitOpError("result #") << index << " must be any type, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}



void XOrOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


//===----------------------------------------------------------------------===//
// ZeroExtendIOp definitions
//===----------------------------------------------------------------------===//

ZeroExtendIOpOperandAdaptor::ZeroExtendIOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ZeroExtendIOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ZeroExtendIOpOperandAdaptor::value() {
  return *getODSOperands(0).begin();
}

StringRef ZeroExtendIOp::getOperationName() {
  return "std.zexti";
}

Operation::operand_range ZeroExtendIOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ZeroExtendIOp::value() {
  return *getODSOperands(0).begin();
}

Operation::result_range ZeroExtendIOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void ZeroExtendIOp::build(Builder *builder, OperationState &result, Value value, Type destType) {
      result.addOperands(value);
      result.addTypes(destType);
  
}

void ZeroExtendIOp::build(Builder *odsBuilder, OperationState &odsState, Type resultType0, Value value) {
  odsState.addOperands(value);
  odsState.addTypes(resultType0);
}

void ZeroExtendIOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ZeroExtendIOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

ParseResult ZeroExtendIOp::parse(OpAsmParser &parser, OperationState &result) {
  return impl::parseCastOp(parser, result);
}

void ZeroExtendIOp::print(OpAsmPrinter &p) {
  return printStandardCastOp(this->getOperation(), p);
}

LogicalResult ZeroExtendIOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("operand #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isSignlessInteger())) || ((v.getType().isa<IndexType>())) || (((v.getType().isa<VectorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))) || (((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger()))))) {
        return emitOpError("result #") << index << " must be signless-integer-like, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}

void ZeroExtendIOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

