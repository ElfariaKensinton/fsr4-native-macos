// SPDX-License-Identifier: MIT
// Shared implementation for the standalone runner and MetalFX live bridge.
// Compile Objective-C++ users with ARC. encode() never commits the caller's
// command buffer. One View is one ordered temporal stream / MetalFX scaler.
#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "fsr4_input.h"
#include "fsr4_log.h"
#include "fsr411_model.h"
#include "fsr411_scratch_layout.h"

namespace fsr411 {

// Simd: Apple7+ (M1–M4) simdgroup half matrices with exact float accumulation;
// byte-identical to Scalar. Auto: Native on Apple10, else Simd, else Scalar.
enum class Backend { Auto, Reference, Native, Scalar, Simd };
// Integer network implementation chosen for one session.
enum class Kernels { Reference, Scalar, Simd, Mpp };
inline const char *kernels_label(Kernels k) {
    return k==Kernels::Mpp?"int8-mpp":k==Kernels::Simd?"int8-simd":k==Kernels::Scalar?"int8-scalar":"int8-reference";
}
// Exact: original coefficients, IEEE PRE/POST math, inference every frame.
// ExactFast: Exact's network with reciprocal division and fast exp/log in
// PRE/POST. Lossy: folded coefficients, half-up rounding and inference reuse.
enum class Profile { Exact, Lossy, ExactFast };
inline const char *profile_name(Profile p) {
    return p==Profile::Lossy?"lossy":p==Profile::ExactFast?"exact-fast":"exact";
}
inline bool parse_profile(const std::string &name, Profile &p) {
    if(name=="exact") p=Profile::Exact;
    else if(name=="exact-fast") p=Profile::ExactFast;
    else if(name=="lossy") p=Profile::Lossy;
    else return false;
    return true;
}
// Live settings encoding (panel.conf, fsr4_live_get_profile): 0/1 keep their
// original meaning so older settings files still load.
inline Profile profile_from_int(int v) { return v==1?Profile::Lossy:v==2?Profile::ExactFast:Profile::Exact; }
inline int profile_to_int(Profile p) { return p==Profile::Lossy?1:p==Profile::ExactFast?2:0; }
// Synchronous, opt-in CLI instrumentation. A profiler may replace the classic
// compute encoder to attach stage-boundary counters. Ordinary/live encodes do
// not install one, retain their single encoder, and record no counter samples.
struct StageProfiler {
    virtual ~StageProfiler() = default;
    virtual void begin_stage(id __strong &encoder, const std::string &name) = 0;
    virtual void end_stage(id encoder) = 0;
};
struct FrameSnapshot {
    Fsr4InputDesc input = {};
    Fsr4InputPrepared prepared = {};
    // Logical input/output regions, with source format values preserved.
    // A fresh snapshot belongs to one encode; its completion token never resets.
    id<MTLTexture> color = nil, depth = nil, motion = nil, output = nil;
    id<MTLBuffer> completion = nil;
    bool ready() const {
        return completion && __atomic_load_n((uint32_t *)completion.contents, __ATOMIC_ACQUIRE) == 1u;
    }
};
struct Options {
    Backend backend = Backend::Auto;
    Profile profile = Profile::Lossy;
    int tier = 0, colorspace = 0, optimize = 1;
    bool rcas = false, allowCompile = true, ultraPerformance = false;
    // Tests only: Exact Fast/Lossy PRE/POST with ordered arithmetic, so their
    // optimized graph can be compared bit for bit with the imported HLSL.
    bool strictMath = false;
    float sharpness = 0;
    // Diagnostic snapshots are GPU copies at a precise graph boundary.
    // They are not enabled by the live bridge or ordinary benchmarking.
    std::vector<std::string> captureStages;
    // Optional immutable CPU-readable frame. No allocation or dispatch when null.
    std::shared_ptr<FrameSnapshot> snapshot;
    StageProfiler *stageProfiler = nullptr; // caller owns it until encode returns
};
struct Frame {
    id<MTLTexture> color = nil, depth = nil, motion = nil, output = nil, exposure = nil;
    Fsr4InputDesc input = {};
};

struct FrameConstants {
    float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2];
    float tex_size[2], max_renderSize[2], jitterCancellation[2];
    uint32_t width, height, reset, width_lr;
    uint32_t height_lr; float preExposure, previousPreExposure; uint32_t rcas;
    float sharpness, pad;
};
struct SpdConstants {
    uint32_t mips, groups, offset[2];
    float invInputSize[2], preExposure, padding;
    uint32_t validSize[2]; // 4.1.1 reimplementation's viewport extension
};
struct RcasConstants { uint32_t config[4]; float preExposure; uint32_t pad[3]; };
struct LossyConstants { uint32_t skip, blockWidth, blockHeight, reserved; };
struct RegionConstants { uint32_t source[4], destination[4]; };
struct TensorSizes { uint32_t v[17][4]; };
static_assert(sizeof(FrameConstants) == 104 && sizeof(SpdConstants) == 40,
              "FSR 4.1.1 frame constant ABI");
static_assert(sizeof(TensorSizes) == 272 && sizeof(RcasConstants) == 32,
              "FSR 4.1.1 tensor/RCAS ABI");

inline uint32_t ceil_div(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }
inline NSString *ns(const std::string &x) { return [NSString stringWithUTF8String:x.c_str()]; }
inline std::string utf8(NSString *x) { return x ? std::string(x.UTF8String) : std::string(); }
inline std::string metal_error(NSError *error) { return error ? utf8(error.localizedDescription) : "Metal allocation failed"; }
// The compiler publishes bindings.json after each stage. Its presence alone
// does not mean an interrupted or --only build is ready for --no-compile.
// This checks completeness only; layout/key/order validation stays in the
// loader so a complete but incompatible alias bundle is still rejected.
inline bool shader_bundle_files_ready(const std::string &directory) {
    NSString *dir=ns(directory);
    NSData *data=[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"bindings.json"]];
    id manifest=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:nil]:nil;
    if(![manifest isKindOfClass:[NSDictionary class]]) return false;
    NSDictionary *entries=manifest[@"kernels"];
    if(![entries isKindOfClass:[NSDictionary class]]) return false;
    for(unsigned index=0;index<27;++index) {
        const std::string name=index==0?"pre":index==26?"post":
            std::string(index&1?"pad":"pass")+std::to_string(index/2);
        if(![entries[ns(name)] isKindOfClass:[NSDictionary class]]) return false;
    }
    if(!entries[@"ae"] || !entries[@"rcas"]) return false;
    for(NSString *flag in @[@"auto_exposure_dispatches",@"separate_reuse_pre"])
        if(manifest[flag] && ![manifest[flag] isKindOfClass:[NSNumber class]]) return false;
    if([manifest[@"auto_exposure_dispatches"] unsignedIntValue]==2 && !entries[@"ae_finalize"]) return false;
    if([manifest[@"separate_reuse_pre"] boolValue] && !entries[@"pre_reuse"]) return false;
    NSFileManager *files=[NSFileManager defaultManager];
    auto readyFile=[&](id name) {
        if(![name isKindOfClass:[NSString class]] || ![name length]) return false;
        NSString *path=[[dir stringByAppendingPathComponent:name] stringByResolvingSymlinksInPath];
        NSDictionary *attributes=[files attributesOfItemAtPath:path error:nil];
        return [attributes[NSFileType] isEqual:NSFileTypeRegular] &&
            [attributes[NSFileSize] unsignedLongLongValue]>0 && [files isReadableFileAtPath:path];
    };
    // Inspect every entry: the loader also loads optional helpers. Requiring
    // generated pass files permits its fallback when a native pass is absent.
    for(NSString *name in entries) {
        NSDictionary *entry=entries[name];
        if(![entry isKindOfClass:[NSDictionary class]] || !readyFile(entry[@"library"]) ||
           ![entry[@"entry"] isKindOfClass:[NSString class]] || ![entry[@"entry"] length]) return false;
        NSArray *wg=entry[@"wgsize"];
        if(![wg isKindOfClass:[NSArray class]] || wg.count!=3) return false;
        for(id size in wg) if(![size isK