/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

shape::AddOp,
shape::BroadcastOp,
shape::ConcatOp,
shape::ConstantOp,
shape::DebugPrintOp,
shape::FromExtentTensorOp,
shape::JoinOp,
shape::MulOp,
shape::ReduceOp,
shape::ShapeOfOp,
shape::SplitAtOp,
shape::ToExtentTensorOp,
shape::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// shape::AddOp definitions
//===----------------------------------------------------------------------===//

AddOpOperandAdaptor::AddOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> AddOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value AddOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value AddOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef AddOp::getOperationName() {
  return "shape.add";
}

Operation::operand_range AddOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value AddOp::lhs() {
  return *getODSOperands(0).begin();
}

Value AddOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range AddOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value AddOp::result() {
  return *getODSResults(0).begin();
}

void AddOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(result);
}

void AddOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void AddOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

LogicalResult AddOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::SizeType>()))) {
        return emitOpError("operand #") << index << " must be size, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::SizeType>()))) {
        return emitOpError("operand #") << index << " must be size, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::SizeType>()))) {
        return emitOpError("result #") << index << " must be size, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::BroadcastOp definitions
//===----------------------------------------------------------------------===//

BroadcastOpOperandAdaptor::BroadcastOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> BroadcastOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value BroadcastOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value BroadcastOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef BroadcastOp::getOperationName() {
  return "shape.broadcast";
}

Operation::operand_range BroadcastOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value BroadcastOp::lhs() {
  return *getODSOperands(0).begin();
}

Value BroadcastOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range BroadcastOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value BroadcastOp::result() {
  return *getODSResults(0).begin();
}

StringAttr BroadcastOp::errorAttr() {
  return this->getAttr("error").dyn_cast_or_null<StringAttr>();
}

Optional< StringRef > BroadcastOp::error() {
  auto attr = errorAttr();
  return attr ? Optional< StringRef >(attr.getValue()) : (llvm::None);
}

void BroadcastOp::errorAttr(StringAttr attr) {
  this->getOperation()->setAttr("error", attr);
}

void BroadcastOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs, /*optional*/StringAttr error) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (error) {
  odsState.addAttribute("error", error);
  }
  odsState.addTypes(result);
}

void BroadcastOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs, /*optional*/StringAttr error) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (error) {
  odsState.addAttribute("error", error);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BroadcastOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult BroadcastOp::verify() {
  auto tblgen_error = this->getAttr("error");
  if (tblgen_error) {
    if (!((tblgen_error.isa<StringAttr>()))) return emitOpError("attribute 'error' failed to satisfy constraint: string attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::ConcatOp definitions
//===----------------------------------------------------------------------===//

ConcatOpOperandAdaptor::ConcatOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ConcatOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ConcatOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value ConcatOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef ConcatOp::getOperationName() {
  return "shape.concat";
}

Operation::operand_range ConcatOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ConcatOp::lhs() {
  return *getODSOperands(0).begin();
}

Value ConcatOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range ConcatOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value ConcatOp::result() {
  return *getODSResults(0).begin();
}

void ConcatOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(result);
}

void ConcatOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        SmallVector<Type, 2> inferredReturnTypes;
        if (succeeded(ConcatOp::inferReturnTypes(odsBuilder->getContext(),
                      odsState.location, odsState.operands, odsState.attributes,
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConcatOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConcatOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    SmallVector<Type, 2> inferredReturnTypes;
    if (succeeded(ConcatOp::inferReturnTypes(odsBuilder->getContext(),
                  odsState.location, operands, attributes,
                  /*regions=*/{}, inferredReturnTypes)))
      build(odsBuilder, odsState, inferredReturnTypes, operands, attributes);
    else
      llvm::report_fatal_error("Failed to infer result type(s).");
}

LogicalResult ConcatOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}




//===----------------------------------------------------------------------===//
// shape::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpOperandAdaptor::ConstantOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ConstantOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

StringRef ConstantOp::getOperationName() {
  return "shape.constant";
}

Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Operation::result_range ConstantOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value ConstantOp::result() {
  return *getODSResults(0).begin();
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

void ConstantOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Attribute value) {
  odsState.addAttribute("value", value);
  odsState.addTypes(result);
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
      if (!(((v.getType().isa<::mlir::shape::SizeType>())) || ((v.getType().isa<::mlir::shape::ShapeType>())))) {
        return emitOpError("result #") << index << " must be shape or size, but got " << v.getType();
      }
      ++index;
    }
  }
  return ::verify(*this);
}


//===----------------------------------------------------------------------===//
// shape::DebugPrintOp definitions
//===----------------------------------------------------------------------===//

DebugPrintOpOperandAdaptor::DebugPrintOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> DebugPrintOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value DebugPrintOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef DebugPrintOp::getOperationName() {
  return "shape.debug_print";
}

Operation::operand_range DebugPrintOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value DebugPrintOp::input() {
  return *getODSOperands(0).begin();
}

Operation::result_range DebugPrintOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value DebugPrintOp::output() {
  return *getODSResults(0).begin();
}

void DebugPrintOp::build(Builder *odsBuilder, OperationState &odsState, Type output, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(output);
}

void DebugPrintOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DebugPrintOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult DebugPrintOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::shape::SizeType>())) || ((v.getType().isa<::mlir::shape::ShapeType>())))) {
        return emitOpError("operand #") << index << " must be shape or size, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::shape::SizeType>())) || ((v.getType().isa<::mlir::shape::ShapeType>())))) {
        return emitOpError("result #") << index << " must be shape or size, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::FromExtentTensorOp definitions
//===----------------------------------------------------------------------===//

FromExtentTensorOpOperandAdaptor::FromExtentTensorOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> FromExtentTensorOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value FromExtentTensorOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef FromExtentTensorOp::getOperationName() {
  return "shape.from_extent_tensor";
}

Operation::operand_range FromExtentTensorOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value FromExtentTensorOp::input() {
  return *getODSOperands(0).begin();
}

Operation::result_range FromExtentTensorOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value FromExtentTensorOp::result() {
  return *getODSResults(0).begin();
}

void FromExtentTensorOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(result);
}

void FromExtentTensorOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FromExtentTensorOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult FromExtentTensorOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isSignlessInteger(32))))) {
        return emitOpError("operand #") << index << " must be tensor of 32-bit signless integer values, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::JoinOp definitions
//===----------------------------------------------------------------------===//

JoinOpOperandAdaptor::JoinOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> JoinOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value JoinOpOperandAdaptor::arg0() {
  return *getODSOperands(0).begin();
}

Value JoinOpOperandAdaptor::arg1() {
  return *getODSOperands(1).begin();
}

StringRef JoinOp::getOperationName() {
  return "shape.join";
}

Operation::operand_range JoinOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value JoinOp::arg0() {
  return *getODSOperands(0).begin();
}

Value JoinOp::arg1() {
  return *getODSOperands(1).begin();
}

Operation::result_range JoinOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value JoinOp::result() {
  return *getODSResults(0).begin();
}

StringAttr JoinOp::errorAttr() {
  return this->getAttr("error").dyn_cast_or_null<StringAttr>();
}

Optional< StringRef > JoinOp::error() {
  auto attr = errorAttr();
  return attr ? Optional< StringRef >(attr.getValue()) : (llvm::None);
}

void JoinOp::errorAttr(StringAttr attr) {
  this->getOperation()->setAttr("error", attr);
}

void JoinOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value arg0, Value arg1, /*optional*/StringAttr error) {
  odsState.addOperands(arg0);
  odsState.addOperands(arg1);
  if (error) {
  odsState.addAttribute("error", error);
  }
  odsState.addTypes(result);
}

void JoinOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value arg0, Value arg1, /*optional*/StringAttr error) {
  odsState.addOperands(arg0);
  odsState.addOperands(arg1);
  if (error) {
  odsState.addAttribute("error", error);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void JoinOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult JoinOp::verify() {
  auto tblgen_error = this->getAttr("error");
  if (tblgen_error) {
    if (!((tblgen_error.isa<StringAttr>()))) return emitOpError("attribute 'error' failed to satisfy constraint: string attribute");
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::shape::SizeType>())) || ((v.getType().isa<::mlir::shape::ShapeType>())))) {
        return emitOpError("operand #") << index << " must be shape or size, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!(((v.getType().isa<::mlir::shape::SizeType>())) || ((v.getType().isa<::mlir::shape::ShapeType>())))) {
        return emitOpError("operand #") << index << " must be shape or size, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<::mlir::shape::SizeType>())) || ((v.getType().isa<::mlir::shape::ShapeType>())))) {
        return emitOpError("result #") << index << " must be shape or size, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::MulOp definitions
//===----------------------------------------------------------------------===//

MulOpOperandAdaptor::MulOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> MulOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value MulOpOperandAdaptor::lhs() {
  return *getODSOperands(0).begin();
}

Value MulOpOperandAdaptor::rhs() {
  return *getODSOperands(1).begin();
}

StringRef MulOp::getOperationName() {
  return "shape.mul";
}

Operation::operand_range MulOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value MulOp::lhs() {
  return *getODSOperands(0).begin();
}

Value MulOp::rhs() {
  return *getODSOperands(1).begin();
}

Operation::result_range MulOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value MulOp::result() {
  return *getODSResults(0).begin();
}

void MulOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(result);
}

void MulOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulOp::build(Builder *odsBuilder, OperationState &odsState, Value lhs, Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes({lhs.getType()});

}

void MulOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes({operands[0].getType()});

}

LogicalResult MulOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::SizeType>()))) {
        return emitOpError("operand #") << index << " must be size, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::SizeType>()))) {
        return emitOpError("operand #") << index << " must be size, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::SizeType>()))) {
        return emitOpError("result #") << index << " must be size, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::ReduceOp definitions
//===----------------------------------------------------------------------===//

ReduceOpOperandAdaptor::ReduceOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ReduceOpOperandAdaptor::getODSOperands(unsigned index) {
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

Value ReduceOpOperandAdaptor::shape() {
  return *getODSOperands(0).begin();
}

ArrayRef<Value> ReduceOpOperandAdaptor::args() {
  return getODSOperands(1);
}

StringRef ReduceOp::getOperationName() {
  return "shape.reduce";
}

Operation::operand_range ReduceOp::getODSOperands(unsigned index) {
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

Value ReduceOp::shape() {
  return *getODSOperands(0).begin();
}

Operation::operand_range ReduceOp::args() {
  return getODSOperands(1);
}

Operation::result_range ReduceOp::getODSResults(unsigned index) {
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

Operation::result_range ReduceOp::result() {
  return getODSResults(0);
}

Region &ReduceOp::body() {
  return this->getOperation()->getRegion(0);
}

void ReduceOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> result, Value shape, ValueRange args) {
  odsState.addOperands(shape);
  odsState.addOperands(args);
  (void)odsState.addRegion();
  odsState.addTypes(result);
}

void ReduceOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

LogicalResult ReduceOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
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
{
    unsigned index = 0; (void)index;
    for (Region &region : MutableArrayRef<Region>(this->getOperation()->getRegion(0))) {
      (void)region;
      if (!((region.getBlocks().size() == 1))) {
        return emitOpError("region #") << index << " ('body') failed to verify constraint: region with 1 blocks";
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::ShapeOfOp definitions
//===----------------------------------------------------------------------===//

ShapeOfOpOperandAdaptor::ShapeOfOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ShapeOfOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ShapeOfOpOperandAdaptor::arg() {
  return *getODSOperands(0).begin();
}

StringRef ShapeOfOp::getOperationName() {
  return "shape.shape_of";
}

Operation::operand_range ShapeOfOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ShapeOfOp::arg() {
  return *getODSOperands(0).begin();
}

Operation::result_range ShapeOfOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value ShapeOfOp::result() {
  return *getODSResults(0).begin();
}

void ShapeOfOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(result);
}

void ShapeOfOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShapeOfOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ShapeOfOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((((v.getType().isa<ShapedType>())) && ((true))) || ((v.getType().isa<::mlir::shape::ValueShapeType>())))) {
        return emitOpError("operand #") << index << " must be shaped of any type values or value shape, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::SplitAtOp definitions
//===----------------------------------------------------------------------===//

SplitAtOpOperandAdaptor::SplitAtOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> SplitAtOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value SplitAtOpOperandAdaptor::operand() {
  return *getODSOperands(0).begin();
}

Value SplitAtOpOperandAdaptor::index() {
  return *getODSOperands(1).begin();
}

void SplitAtOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "head");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "tail");
}

StringRef SplitAtOp::getOperationName() {
  return "shape.split_at";
}

Operation::operand_range SplitAtOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value SplitAtOp::operand() {
  return *getODSOperands(0).begin();
}

Value SplitAtOp::index() {
  return *getODSOperands(1).begin();
}

Operation::result_range SplitAtOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value SplitAtOp::head() {
  return *getODSResults(0).begin();
}

Value SplitAtOp::tail() {
  return *getODSResults(1).begin();
}

void SplitAtOp::build(Builder *odsBuilder, OperationState &odsState, Type head, Type tail, Value operand, Value index) {
  odsState.addOperands(operand);
  odsState.addOperands(index);
  odsState.addTypes(head);
  odsState.addTypes(tail);
}

void SplitAtOp::build(Builder *odsBuilder, OperationState &odsState, Value operand, Value index) {
  odsState.addOperands(operand);
  odsState.addOperands(index);

        SmallVector<Type, 2> inferredReturnTypes;
        if (succeeded(SplitAtOp::inferReturnTypes(odsBuilder->getContext(),
                      odsState.location, odsState.operands, odsState.attributes,
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          llvm::report_fatal_error("Failed to infer result type(s).");
}

void SplitAtOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value operand, Value index) {
  odsState.addOperands(operand);
  odsState.addOperands(index);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplitAtOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SplitAtOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    SmallVector<Type, 2> inferredReturnTypes;
    if (succeeded(SplitAtOp::inferReturnTypes(odsBuilder->getContext(),
                  odsState.location, operands, attributes,
                  /*regions=*/{}, inferredReturnTypes)))
      build(odsBuilder, odsState, inferredReturnTypes, operands, attributes);
    else
      llvm::report_fatal_error("Failed to infer result type(s).");
}

LogicalResult SplitAtOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSOperands(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isSignlessInteger(32)))) {
        return emitOpError("operand #") << index << " must be 32-bit signless integer, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
    auto valueGroup1 = getODSResults(1);
    for (Value v : valueGroup1) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("result #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}




//===----------------------------------------------------------------------===//
// shape::ToExtentTensorOp definitions
//===----------------------------------------------------------------------===//

ToExtentTensorOpOperandAdaptor::ToExtentTensorOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> ToExtentTensorOpOperandAdaptor::getODSOperands(unsigned index) {
  return {std::next(tblgen_operands.begin(), index), std::next(tblgen_operands.begin(), index + 1)};
}

Value ToExtentTensorOpOperandAdaptor::input() {
  return *getODSOperands(0).begin();
}

StringRef ToExtentTensorOp::getOperationName() {
  return "shape.to_extent_tensor";
}

Operation::operand_range ToExtentTensorOp::getODSOperands(unsigned index) {
  return {std::next(getOperation()->operand_begin(), index), std::next(getOperation()->operand_begin(), index + 1)};
}

Value ToExtentTensorOp::input() {
  return *getODSOperands(0).begin();
}

Operation::result_range ToExtentTensorOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

Value ToExtentTensorOp::result() {
  return *getODSResults(0).begin();
}

void ToExtentTensorOp::build(Builder *odsBuilder, OperationState &odsState, Type result, Value input) {
  odsState.addOperands(input);
  odsState.addTypes(result);
}

void ToExtentTensorOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ToExtentTensorOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult ToExtentTensorOp::verify() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!((v.getType().isa<::mlir::shape::ShapeType>()))) {
        return emitOpError("operand #") << index << " must be shape, but got " << v.getType();
      }
      ++index;
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);
    for (Value v : valueGroup0) {
      (void)v;
      if (!(((v.getType().isa<TensorType>())) && ((v.getType().cast<ShapedType>().getElementType().isa<IndexType>())))) {
        return emitOpError("result #") << index << " must be tensor of index values, but got " << v.getType();
      }
      ++index;
    }
  }
  return mlir::success();
}


//===----------------------------------------------------------------------===//
// shape::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpOperandAdaptor::YieldOpOperandAdaptor(ArrayRef<Value> values) {
  tblgen_operands = values;
}

ArrayRef<Value> YieldOpOperandAdaptor::getODSOperands(unsigned index) {
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

ArrayRef<Value> YieldOpOperandAdaptor::operands() {
  return getODSOperands(0);
}

StringRef YieldOp::getOperationName() {
  return "shape.yield";
}

Operation::operand_range YieldOp::getODSOperands(unsigned index) {
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

Operation::operand_range YieldOp::operands() {
  return getODSOperands(0);
}

Operation::result_range YieldOp::getODSResults(unsigned index) {
  return {std::next(getOperation()->result_begin(), index), std::next(getOperation()->result_begin(), index + 1)};
}

void YieldOp::build(Builder *b, OperationState &result) {
 build(b, result, llvm::None); 
}

void YieldOp::build(Builder *odsBuilder, OperationState &odsState, ValueRange operands) {
  odsState.addOperands(operands);
}

void YieldOp::build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands) {
  odsState.addOperands(operands);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

LogicalResult YieldOp::verify() {
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
  return mlir::success();
}

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
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

void YieldOp::print(OpAsmPrinter &p) {
  p << "shape.yield";
  p.printOptionalAttrDict(getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  p << " ";
  p << operands();
  p << " " << ":";
  p << " ";
  p << operands().getTypes();
  }
}

void YieldOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {

}


#endif  // GET_OP_CLASSES

