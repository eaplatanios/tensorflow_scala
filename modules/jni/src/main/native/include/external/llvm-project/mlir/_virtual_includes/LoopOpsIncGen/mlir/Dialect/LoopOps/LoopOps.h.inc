/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES

class ForOp;
class IfOp;
class ParallelOp;
class ReduceOp;
class ReduceReturnOp;
class YieldOp;

//===----------------------------------------------------------------------===//
// ForOp declarations
//===----------------------------------------------------------------------===//

class ForOpOperandAdaptor {
public:
  ForOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value lowerBound();
  Value upperBound();
  Value step();
  ArrayRef<Value> initArgs();

private:
  ArrayRef<Value> tblgen_operands;
};
class ForOp : public Op<ForOp, OpTrait::OneRegion, OpTrait::VariadicResults, OpTrait::ZeroSuccessor, OpTrait::AtLeastNOperands<3>::Impl, LoopLikeOpInterface::Trait, RegionBranchOpInterface::Trait, OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using OperandAdaptor = ForOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value lowerBound();
  Value upperBound();
  Value step();
  Operation::operand_range initArgs();
  Operation::result_range getODSResults(unsigned index);
  Operation::result_range results();
  Region &region();
  static void build(Builder *builder, OperationState &result, Value lowerBound, Value upperBound, Value step, ValueRange iterArgs = llvm::None);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  bool isDefinedOutsideOfLoop(Value  value);
  Region &getLoopBody();
  LogicalResult moveOutOfLoop(ArrayRef<Operation *> ops);
  void getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions);

    Block *getBody() { return &region().front(); }
    Value getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }
    Operation::operand_range getIterOperands() {
      return getOperands().drop_front(getNumControlOperands());
    }

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - 1;
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }
    /// Does the operation hold operands for loop-carried values
    bool hasIterOperands() {
      return getOperation()->getNumOperands() > getNumControlOperands();
    }
    /// Get Number of loop-carried values
    unsigned getNumIterOperands() {
      return getOperation()->getNumOperands() - getNumControlOperands();
    }

    /// Return operands used when entering the region at 'index'. These operands
    /// correspond to the loop iterator operands, i.e., those exclusing the
    /// induction variable. LoopOp only has one region, so 0 is the only valid
    /// value for `index`.
    OperandRange getSuccessorEntryOperands(unsigned index);
  
};

//===----------------------------------------------------------------------===//
// IfOp declarations
//===----------------------------------------------------------------------===//

class IfOpOperandAdaptor {
public:
  IfOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value condition();

private:
  ArrayRef<Value> tblgen_operands;
};
class IfOp : public Op<IfOp, OpTrait::NRegions<2>::Impl, OpTrait::VariadicResults, OpTrait::ZeroSuccessor, OpTrait::OneOperand, RegionBranchOpInterface::Trait, OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using OperandAdaptor = IfOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value condition();
  Operation::result_range getODSResults(unsigned index);
  Operation::result_range results();
  Region &thenRegion();
  Region &elseRegion();
  static void build(Builder *builder, OperationState &result, Value cond, bool withElseRegion);
  static void build(Builder *builder, OperationState &result, TypeRange resultTypes, Value cond, bool withElseRegion);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  void getSuccessorRegions(Optional<unsigned> index, ArrayRef<Attribute> operands, SmallVectorImpl<RegionSuccessor> & regions);

    OpBuilder getThenBodyBuilder() {
      assert(!thenRegion().empty() && "Unexpected empty 'then' region.");
      Block &body = thenRegion().front();
      return OpBuilder(&body,
                       results().empty() ? std::prev(body.end()) : body.end());
    }
    OpBuilder getElseBodyBuilder() {
      assert(!elseRegion().empty() && "Unexpected empty 'else' region.");
      Block &body = elseRegion().front();
      return OpBuilder(&body,
                       results().empty() ? std::prev(body.end()) : body.end());
    }
  
};

//===----------------------------------------------------------------------===//
// ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOp : public Op<ParallelOp, OpTrait::OneRegion, OpTrait::VariadicResults, OpTrait::ZeroSuccessor, OpTrait::VariadicOperands, OpTrait::AttrSizedOperandSegments, LoopLikeOpInterface::Trait, OpTrait::HasRecursiveSideEffects, OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl> {
public:
  using Op::Op;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range lowerBound();
  Operation::operand_range upperBound();
  Operation::operand_range step();
  Operation::operand_range initVals();
  Operation::result_range getODSResults(unsigned index);
  Operation::result_range results();
  Region &region();
  static void build(Builder *builder, OperationState &result, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange initVals = {});
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  bool isDefinedOutsideOfLoop(Value  value);
  Region &getLoopBody();
  LogicalResult moveOutOfLoop(ArrayRef<Operation *> ops);

    Block *getBody() { return &region().front(); }
    ValueRange getInductionVars() {
      return getBody()->getArguments();
    }
    unsigned getNumLoops() { return step().size(); }
    unsigned getNumReductions() { return initVals().size(); }
  
};

//===----------------------------------------------------------------------===//
// ReduceOp declarations
//===----------------------------------------------------------------------===//

class ReduceOpOperandAdaptor {
public:
  ReduceOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value operand();

private:
  ArrayRef<Value> tblgen_operands;
};
class ReduceOp : public Op<ReduceOp, OpTrait::OneRegion, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, OpTrait::HasParent<ParallelOp>::Impl> {
public:
  using Op::Op;
  using OperandAdaptor = ReduceOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value operand();
  Operation::result_range getODSResults(unsigned index);
  Region &reductionOperator();
  static void build(Builder *builder, OperationState &result, Value operand);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
};

//===----------------------------------------------------------------------===//
// ReduceReturnOp declarations
//===----------------------------------------------------------------------===//

class ReduceReturnOpOperandAdaptor {
public:
  ReduceReturnOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  Value result();

private:
  ArrayRef<Value> tblgen_operands;
};
class ReduceReturnOp : public Op<ReduceReturnOp, OpTrait::ZeroRegion, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::OneOperand, OpTrait::HasParent<ReduceOp>::Impl, MemoryEffectOpInterface::Trait, OpTrait::IsTerminator> {
public:
  using Op::Op;
  using OperandAdaptor = ReduceReturnOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Value result();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *odsBuilder, OperationState &odsState, Value result);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, Value result);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  LogicalResult verify();
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

//===----------------------------------------------------------------------===//
// YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpOperandAdaptor {
public:
  YieldOpOperandAdaptor(ArrayRef<Value> values);
  ArrayRef<Value> getODSOperands(unsigned index);
  ArrayRef<Value> results();

private:
  ArrayRef<Value> tblgen_operands;
};
class YieldOp : public Op<YieldOp, OpTrait::ZeroRegion, OpTrait::ZeroResult, OpTrait::ZeroSuccessor, OpTrait::VariadicOperands, MemoryEffectOpInterface::Trait, OpTrait::ReturnLike, OpTrait::IsTerminator> {
public:
  using Op::Op;
  using OperandAdaptor = YieldOpOperandAdaptor;
  static StringRef getOperationName();
  Operation::operand_range getODSOperands(unsigned index);
  Operation::operand_range results();
  Operation::result_range getODSResults(unsigned index);
  static void build(Builder *builder, OperationState &result);
  static void build(Builder *odsBuilder, OperationState &odsState, ValueRange results);
  static void build(Builder *odsBuilder, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange results);
  static void build(Builder *, OperationState &odsState, ArrayRef<Type> resultTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();
  void getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects);
};

#endif  // GET_OP_CLASSES

