#include "PTXAsmFormat.h"
#include "Passes.h"
#include "TargetInfo.h"
#include "intellifusion/lib/TargetCapabilities.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace ttn = mlir::triton::nvgpu;
using Constraints = std::vector<std::string>;
using OperandsAndConstraints = std::vector<std::pair<mlir::Value, std::string>>;

namespace mlir {
namespace triton {
namespace Intellifusion {

static LogicalResult
rewriteAsPtxAsm(Operation *op, PatternRewriter &rewriter, std::string ptxAsm,
                const OperandsAndConstraints &operandsAndConstraints = {},
                const Constraints &outputConstraints = {});

namespace {

bool isNumber(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

Type getTypeFromConstraint(char constraint, PatternRewriter &rewriter) {
  Type ty;
  if (constraint == 'b')
    ty = IntegerType::get(rewriter.getContext(), 1);
  else if (constraint == 'h')
    ty = IntegerType::get(rewriter.getContext(), 16);
  else if (constraint == 'r')
    ty = IntegerType::get(rewriter.getContext(), 32);
  else if (constraint == 'l')
    ty = IntegerType::get(rewriter.getContext(), 64);
  else if (constraint == 'f')
    ty = Float32Type::get(rewriter.getContext());
  else if (constraint == 'd')
    ty = Float64Type::get(rewriter.getContext());
  else {
    assert(false && "Unsupported constraint");
  }
  return ty;
}

// Converts the given value to the type represented by the constraint
// E.g. if val is of type llvmptr and constraint is 'r', then we convert
// val to i32 using ptrtoint(i32_ty, val)
Value convertToType(Value val, std::string constraint, Location loc,
                    PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto isConstraintNumber = isNumber(constraint);
  if (!isConstraintNumber) {
    auto ty = getTypeFromConstraint(constraint[0], rewriter);
    if (isa<LLVM::LLVMPointerType>(val.getType())) {
      return b.ptrtoint(ty, val);
    } else {
      assert(val.getType().getIntOrFloatBitWidth() <=
                 ty.getIntOrFloatBitWidth() &&
             "Cannot convert to a smaller type");
      if (val.getType().getIntOrFloatBitWidth() < ty.getIntOrFloatBitWidth())
        return b.zext(ty, val);
    }
  }
  return val;
}

SmallVector<GPNPUPTXBuilder::Operand *>
getPtxOutputs(const Constraints &outputConstraints,
              GPNPUPTXBuilder &ptxBuilder) {
  SmallVector<GPNPUPTXBuilder::Operand *> ptxOutputs;
  for (unsigned i = 0; i < outputConstraints.size(); i++) {
    auto *ptxOutput = ptxBuilder.newOperand(outputConstraints[i]);
    ptxOutputs.push_back(ptxOutput);
  }
  return ptxOutputs;
}

OperandsAndConstraints
unpackOperands(const OperandsAndConstraints &operandsAndConstraints,
               GPNPUPTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  OperandsAndConstraints unpackedOperands;
  for (const auto &[operand, constraint] : operandsAndConstraints) {
    auto llvmStruct = llvm::dyn_cast<LLVM::LLVMStructType>(operand.getType());
    // if a constraint is a number, then we are doing input/output tying
    // if the operand is a struct, then we need to unpack it, and
    // add the constraint to each of the unpacked operands uses the constraint
    // as an offset
    auto isConstraintNumber = isNumber(constraint);
    if (llvmStruct) {
      for (unsigned i = 0; i < llvmStruct.getBody().size(); i++) {
        if (isConstraintNumber) {
          auto constraintInt = std::stoi(constraint) + i;
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i),
               std::to_string(constraintInt)});
        } else {
          unpackedOperands.push_back(
              {b.extract_val(llvmStruct.getBody()[i], operand, i), constraint});
        }
      }
    } else {
      unpackedOperands.push_back({operand, constraint});
    }
  }
  return unpackedOperands;
}

SmallVector<GPNPUPTXBuilder::Operand *>
getPtxOperands(const OperandsAndConstraints &operandsAndConstraints,
               GPNPUPTXBuilder &ptxBuilder, Location loc,
               PatternRewriter &rewriter) {
  SmallVector<GPNPUPTXBuilder::Operand *> ptxOperands;
  auto unpackedOperandsAndConstraints =
      unpackOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  for (auto &[operand, constraint] : unpackedOperandsAndConstraints) {
    auto convertedOperand = convertToType(operand, constraint, loc, rewriter);
    auto *ptxOperand = ptxBuilder.newOperand(convertedOperand, constraint);
    ptxOperands.push_back(ptxOperand);
  }
  return ptxOperands;
}

std::string patchPtxAsm(Operation *op, std::string ptxAsm) {
  std::vector<std::pair<int, int>> patchLocations;
  std::vector<std::string> patchValues;
  auto start = ptxAsm.find("#", 0);
  while (start != std::string::npos) {
    auto endIterator =
        std::find_if(ptxAsm.begin() + start + 1, ptxAsm.end(),
                     [](unsigned char c) { return !std::isalnum(c); });

    assert(endIterator != ptxAsm.end() && "unexpected asm format");

    auto end = std::distance(ptxAsm.begin(), endIterator);
    auto patchLocation = std::make_pair(start, end);
    patchLocations.push_back(patchLocation);
    auto patchValue = ptxAsm.substr(start + 1, end - start - 1);
    patchValues.push_back(patchValue);
    start = ptxAsm.find("#", end);
  }
  assert(patchLocations.size() == patchValues.size() &&
         "patchLocations and patchValues should have the same size");
  if (patchLocations.size() == 0) {
    return ptxAsm;
  }
  std::string res = "";
  size_t prevStart = 0;
  unsigned i = 0;
  for (auto &[start, end] : patchLocations) {
    res += ptxAsm.substr(prevStart, start - prevStart);
    auto integerAttr = op->getAttrOfType<IntegerAttr>(patchValues[i]);
    auto attr = integerAttr.getInt();
    res += std::to_string(attr);
    prevStart = end;
    i++;
  }
  if (prevStart < ptxAsm.size())
    res += ptxAsm.substr(prevStart, ptxAsm.size() - prevStart);
  return res;
}

template <typename SourceOp>
class NVGPUOpGenericPattern : public OpRewritePattern<SourceOp> {
public:
  explicit NVGPUOpGenericPattern(MLIRContext *context, std::string ptxAsm,
                                 Constraints outputConstraints,
                                 Constraints inputConstraints)
      : OpRewritePattern<SourceOp>(context), ptxAsm(std::move(ptxAsm)),
        outputConstraints(outputConstraints),
        inputConstraints(inputConstraints) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    OperandsAndConstraints operandsAndConstraints;
    for (unsigned i = 0; i < inputConstraints.size(); i++) {
      operandsAndConstraints.push_back(
          {op->getOperand(i), inputConstraints[i]});
    }
    return rewriteAsPtxAsm(op, rewriter, ptxAsm, operandsAndConstraints,
                           outputConstraints);
  }

private:
  std::string ptxAsm;
  Constraints outputConstraints;
  Constraints inputConstraints;
};

class WarpIdOpPattern : public OpRewritePattern<mlir::triton::gpu::WarpIdOp> {
public:
  using OpRewritePattern<mlir::triton::gpu::WarpIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::gpu::WarpIdOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (triton::gpu::lookupNumWarps(op) == 1) {
      // If there is only one warp, the warp ID is always 0.
      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    Value tid = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
    Value warpId = b.udiv(tid, b.i32_val(32));
    if (!op.getOmitUniformHint()) {
      // This indicates to PTXAS that the result and its derived values are
      // uniform across the warp. For example, if a branch condition derives
      // from this value, it can be proven to be non-divergent.
      TargetInfo targetInfo(/*threadsPerWarp=*/32,
                            triton::gpu::lookupNumWarps(op));
      warpId = targetInfo.shuffleIdx(rewriter, loc, warpId, 0);
    }
    rewriter.replaceOp(op, warpId);
    return success();
  }
};

class ClusterCTAIdOpPattern : public OpRewritePattern<ttn::ClusterCTAIdOp> {
  using OpRewritePattern<ttn::ClusterCTAIdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::ClusterCTAIdOp op,
                                PatternRewriter &rewriter) const override {
    // We could use the value range from LLVM, but it seems to change the
    // codegen quite a bit. Adding an `and` with `nCTAs - 1` generates similar
    // code than not doing anything, so we don't do anything for now. At the end
    // of the day, we are setting reqnctapercluster so both LLVM and PTXAS
    // already know about the range of the cluster ID.
    Value res = NVVM::ClusterId::create(rewriter, op.getLoc(), i32_ty);
    rewriter.replaceOp(op, res);
    return success();
  }
};

class LoadAcquireOpPattern : public OpRewritePattern<ttn::LoadAcquireOp> {
public:
  using OpRewritePattern<ttn::LoadAcquireOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttn::LoadAcquireOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type valueTy = op.getType();
    const unsigned valueNBits =
        std::max(8u, (unsigned)getIntOrFloatOrPtrBitWidth(valueTy));
    const size_t maxWordWidth = std::max<size_t>(32, valueNBits);
    const size_t width = std::min((size_t)valueNBits, maxWordWidth);

    const std::string writeConstraint =
        (width == 64) ? "=l" : ((width == 32) ? "=r" : "=c");
    GPNPUPTXBuilder ptxBuilder;
    bool init = true;
    auto *dstOpr = ptxBuilder.newOperand(writeConstraint, init); // =r operation
    auto *addrOpr =
        ptxBuilder.newAddrOperand(op.getAddr(), "l", 0 /* in_off */);
    auto &ld =
        ptxBuilder.create("ld")
            ->global()
            .o("cta", op.getScope() == triton::nvgpu::MemSyncScope::CTA)
            .o("gpu", op.getScope() == triton::nvgpu::MemSyncScope::GPU)
            .o("sys", op.getScope() == triton::nvgpu::MemSyncScope::SYSTEM)
            .o("acquire", op.getSem() == triton::nvgpu::MemSemantic::ACQUIRE)
            .o("relaxed", op.getSem() == triton::nvgpu::MemSemantic::RELAXED)
            .b(width);
    ld(dstOpr, addrOpr).maybePredicate(op.getMask(), "b");

    // Create inline ASM signature
    Type retTy = IntegerType::get(getContext(), width);
    Value ret = ptxBuilder.launch(rewriter, loc, retTy);
    ret = b.bitcast(ret, op.getType());

    rewriter.replaceOp(op, {ret});
    return success();
  }
};

static Value createTMAlloc(IRRewriter &rewriter, LLVM::LLVMFuncOp func,
                           std::string exclusive, size_t size, Value pred,
                           bool twoCTAs) {
  GPNPUPTXBuilder ptxBuilder;
  Location loc = func.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value sharedMem = mlir::LLVM::getStackPointer(rewriter, func);
  std::string ptxString = "@$0 tcgen05.alloc" + exclusive +
                          ".cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
                          ".sync.aligned.shared::cta.b32 [$1], " +
                          std::to_string(size) + ";";

  auto &allocOp = *ptxBuilder.create(ptxString);
  allocOp(
      {ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(sharedMem, "r")},
      /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(func->getContext()));
  NVVM::BarrierOp::create(rewriter, loc);
  Value address = b.load(i32_ty, sharedMem);
  NVVM::BarrierOp::create(rewriter, loc);
  address = b.inttoptr(ptr_ty(func.getContext(), 6), address);
  return address;
}

static void createRelinquishAlloc(IRRewriter &rewriter, Location loc,
                                  Value pred, bool twoCTAs) {
  GPNPUPTXBuilder ptxBuilder;
  std::string ptxString = "@$0 tcgen05.relinquish_alloc_permit.cta_group::" +
                          std::to_string(twoCTAs ? 2 : 1) + ".sync.aligned;";
  auto &f = *ptxBuilder.create(ptxString);
  f({ptxBuilder.newOperand(pred, "b")}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void freeTMAlloc(LLVM::LLVMFuncOp func, Value alloc, std::string exclusive,
                 size_t size, Value pred, bool twoCTAs) {
  func.walk([&](LLVM::ReturnOp ret) {
    OpBuilder b(ret);
    auto ctx = ret->getContext();
    auto loc = ret.getLoc();
    if (twoCTAs) {
      NVVM::ClusterArriveOp::create(b, loc, UnitAttr::get(ctx));
      NVVM::ClusterWaitOp::create(b, loc, UnitAttr::get(ctx));
    } else {
      NVVM::BarrierOp::create(b, loc);
    }
    GPNPUPTXBuilder ptxBuilder;
    // Calculate the predicate in the inline asm to avoid creating long
    // liveranges.
    std::string ptxString = "@$0 tcgen05.dealloc" + exclusive +
                            ".cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
                            ".sync.aligned.b32 $1, " + std::to_string(size) +
                            ";";
    auto &dealloc = *ptxBuilder.create(ptxString);
    dealloc(
        {ptxBuilder.newOperand(pred, "b"), ptxBuilder.newOperand(alloc, "r")},
        /*onlyAttachMLIRArgs=*/true);
    ptxBuilder.launch(b, loc, void_ty(ctx));
  });
}

static Value initTensorMemory(LLVM::LLVMFuncOp func) {
  auto mod = func->getParentOfType<ModuleOp>();
  assert(mod->hasAttr("ttg.tensor_memory_size"));
  size_t size = cast<IntegerAttr>(mod->getAttr("ttg.tensor_memory_size"))
                    .getValue()
                    .getZExtValue();
  if (size == 0)
    return Value();
  IRRewriter rewriter(func.getContext());
  rewriter.setInsertionPointToStart(&func.front());
  auto ctx = mod.getContext();
  auto loc = func.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  constexpr size_t tmemMaxSize = target::kMaxTensorMemoryColumns;
  // A proper error is raised by GPNPU allocation, but keep a trap here for
  // externally-authored IR that bypasses that pass.
  // continue we emit a trap.
  if (size > tmemMaxSize) {
    LLVM::Trap::create(rewriter, loc);
    return LLVM::UndefOp::create(rewriter, loc, ptr_ty(ctx, 6));
  }

  std::string exclusive;
  bool useTwoCTAs = mlir::triton::nvidia_gpu::getModuleTwoCTAs(mod);
  // This code is only executed by the default warp group.
  Value threadId = NVVM::ThreadIdXOp::create(rewriter, loc, i32_ty);
  Value pred = b.icmp_ult(threadId, b.i32_val(32));
  Value alloc =
      createTMAlloc(rewriter, func, exclusive, size, pred, useTwoCTAs);
  createRelinquishAlloc(rewriter, loc, pred, useTwoCTAs);
  // TODO: pred will have a long liverange, we need to check if this is a
  // problem and how it can be fixed.
  freeTMAlloc(func, alloc, exclusive, size, pred, useTwoCTAs);
  return alloc;
}

static void lowerTensorMemoryAlloc(ModuleOp mod) {
  SmallVector<Operation *> baseOps;
  LLVM::LLVMFuncOp kernel = nullptr;
  mod.walk([&](ttn::TensorMemoryBaseAddress baseOp) {
    baseOps.push_back(baseOp);
    if (!kernel)
      kernel = baseOp->getParentOfType<LLVM::LLVMFuncOp>();
    assert(kernel == baseOp->getParentOfType<LLVM::LLVMFuncOp>() &&
           "TODO: add support for function calls using tmem.");
  });
  if (baseOps.empty())
    return;
  // TODO: Handle cases of matmul used in noinline functions.
  assert(triton::isKernel(kernel));
  Value newBase = initTensorMemory(kernel);
  if (!newBase)
    return;
  for (auto baseOp : baseOps) {
    baseOp->getResult(0).replaceAllUsesWith(newBase);
    baseOp->erase();
  }
}

} // anonymous namespace

class GPNPUCompatToLLVMPass
    : public PassWrapper<GPNPUCompatToLLVMPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPUCompatToLLVMPass)

  StringRef getArgument() const final { return "gpnpu-nvgpu-to-llvm"; }
  StringRef getDescription() const final {
    return "lower supported NVGPU operations to NVVM/LLVM for GPNPU";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect,
                    triton::gpu::TritonGPUDialect,
                    triton::nvidia_gpu::TritonNvidiaGPUDialect,
                    triton::nvgpu::NVGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    bool hasUnsupportedWGMMA = false;
    mod.walk([&](Operation *op) {
      if (isa<ttn::WGMMAOp, ttn::WGMMAWaitGroupOp>(op)) {
        op->emitError("GPNPU does not support MMA v3/WGMMA lowering");
        hasUnsupportedWGMMA = true;
      }
    });
    if (hasUnsupportedWGMMA)
      return signalPassFailure();

    RewritePatternSet patterns(context);

    patterns.add<ClusterCTAIdOpPattern, LoadAcquireOpPattern, WarpIdOpPattern>(
        context);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

    lowerTensorMemoryAlloc(mod);
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }
};

static LogicalResult
rewriteAsPtxAsm(Operation *op, PatternRewriter &rewriter, std::string ptxAsm,
                const OperandsAndConstraints &operandsAndConstraints,
                const Constraints &outputConstraints) {
  auto ctx = rewriter.getContext();
  auto loc = op->getLoc();
  ptxAsm = patchPtxAsm(op, std::move(ptxAsm));
  auto hasSideEffects = !isMemoryEffectFree(op);

  GPNPUPTXBuilder ptxBuilder;
  auto ptxOutputs = getPtxOutputs(outputConstraints, ptxBuilder);
  auto ptxOperands =
      getPtxOperands(operandsAndConstraints, ptxBuilder, loc, rewriter);
  SmallVector<GPNPUPTXBuilder::Operand *> outputsAndOperands = ptxOutputs;
  outputsAndOperands.append(ptxOperands.begin(), ptxOperands.end());
  auto &ptxInstr = *ptxBuilder.create(ptxAsm);
  ptxInstr(outputsAndOperands, /*onlyAttachMLIRArgs=*/true);
  auto retTy =
      op->getNumResults() == 0 ? void_ty(ctx) : op->getResult(0).getType();
  auto res = ptxBuilder.launch(rewriter, loc, retTy,
                               /*hasSideEffects*/ hasSideEffects);
  if (op->getNumResults() == 0) {
    rewriter.eraseOp(op);
  } else {
    rewriter.replaceOp(op, res);
  }

  return success();
}

std::unique_ptr<Pass> createGPNPUCompatToLLVMPass() {
  return std::make_unique<GPNPUCompatToLLVMPass>();
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
