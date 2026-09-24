// Loaded by the copied D3DMetal (LC_LOAD_DYLIB). Swizzles only the temporal
// scaler D3DMetal creates for DLSS. Spatial, frame generation and denoise
// stay on MetalFX. FSR4_REPLACE=1 is the switch; until the encoder returns
// success the original encode still runs, so a bad build does not black the frame.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <stdio.h>
#import <stdarg.h>
#import <string.h>
#import <unistd.h>
#import "fsr4_live.h"

static IMP g_orig_new = NULL;
static IMP g_orig_new4 = NULL;
static const void *kFlagsAssoc = &kFlagsAssoc;

typedef struct { Class cls; IMP imp; } EncodeSlot;
static EncodeSlot g_encode[4];

static IMP orig_encode(Class cls) {
    for (int i = 0; i < 4; i++)
        if (g_encode[i].cls == cls) return g_encode[i].imp;
    return NULL;
}

static void hook_log(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[fsr4] %s\n", buf);
    const char *path = getenv("FSR4_HOOK_LOG");
    if (!path || !path[0]) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f, "[fsr4] %s\n", buf);
    fclose(f);
}

// Descriptor flags D3DMetal set before creating the scaler. Read on the
// descriptor itself (self of newTemporalScalerWithDevice:), then attached to
// the scaler object: jittered MVs, output-resolution MVs, auto exposure.
static int desc_flag(id desc, SEL sel) {
    if (![desc respondsToSelector:sel]) return -1;
    BOOL (*get)(id, SEL) = (BOOL (*)(id, SEL))objc_msgSend;
    return get(desc, sel) ? 1 : 0;
}

static void remember_flags(id scaler, id desc) {
    if (!scaler) return;
    int jit = desc_flag(desc, @selector(isJitteredMotionVectorsEnabled));
    int outres = desc_flag(desc, @selector(isOutputResolutionMotionVectorsEnabled));
    int ae = desc_flag(desc, @selector(isAutoExposureEnabled));
    int props = desc_flag(desc, @selector(isInputContentPropertiesEnabled));
    NSArray *flags = @[@(jit), @(outres), @(ae)];
    objc_setAssociatedObject(scaler, kFlagsAssoc, flags, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    MTLFXTemporalScalerDescriptor *d = desc;
    hook_log("scaler %p created: desc in %lux%lu out %lux%lu color %lu depth %lu motion %lu out %lu "
             "jitteredMV=%d outputResMV=%d autoExposure=%d inputContentProps=%d minScale %.3f maxScale %.3f",
             scaler, (unsigned long)d.inputWidth, (unsigned long)d.inputHeight,
             (unsigned long)d.outputWidth, (unsigned long)d.outputHeight,
             (unsigned long)d.colorTextureFormat, (unsigned long)d.depthTextureFormat,
             (unsigned long)d.motionTextureFormat, (unsigned long)d.outputTextureFormat,
             jit, outres, ae, props, d.inputContentMinScale, d.inputContentMaxScale);
}

// Returns 1 if this encode was consumed by FSR4.
static int fsr4_replace_encode(id scaler, id commandBuffer) {
    id<MTLFXTemporalScaler> s = scaler;
    id<MTLTexture> color = s.colorTexture;
    id<MTLTexture> depth = s.depthTexture;
    id<MTLTexture> motion = s.motionTexture;
    id<MTLTexture> output = s.outputTexture;
    if (!color || !depth || !motion || !output) { fsr4_live_set_inactive("скейлер без текстур"); return 0; }
    uint32_t rw = (uint32_t)s.inputContentWidth;
    uint32_t rh = (uint32_t)s.inputContentHeight;
    if (!rw) rw = (uint32_t)color.width;
    if (!rh) rh = (uint32_t)color.height;
    NSArray *flags = objc_getAssociatedObject(scaler, kFlagsAssoc);
    Fsr4LiveFrame f;
    memset(&f, 0, sizeof f);
    f.device = (__bridge void *)color.device;
    f.commandBuffer = (__bridge void *)commandBuffer;
    f.owner = (__bridge void *)scaler;
    f.color = (__bridge void *)color;
    f.depth = (__bridge void *)depth;
    f.motion = (__bridge void *)motion;
    f.output = (__bridge void *)output;
    f.exposure = (__bridge void *)s.exposureTexture;
    f.rw = rw; f.rh = rh;
    f.uw = (uint32_t)output.width; f.uh = (uint32_t)output.height;
    f.jitterX = s.jitterOffsetX; f.jitterY = s.jitterOffsetY;
    f.mvScaleX = s.motionVectorScaleX; f.mvScaleY = s.motionVectorScaleY;
    f.preExposure = s.preExposure;
    f.reset = s.reset ? 1 : 0;
    f.depthReversed = s.depthReversed ? 1 : 0;
    f.jitteredMV = flags ? [flags[0] intValue] : -1;
    f.outputResMV = flags ? [flags[1] intValue] : -1;
    f.autoExposure = flags ? [flags[2] intValue] : -1;
    if ([scaler respondsToSelector:@selector(motionContentOffsetX)]) {
        f.mvOffX = (uint32_t)[scaler motionContentOffsetX];
        f.mvOffY = (uint32_t)[scaler motionContentOffsetY];
    }
    return fsr4_encode_live_frame(&f);
}

static void hooked_encode(id self, SEL _cmd, id commandBuffer) {
    IMP orig = orig_encode(object_getClass(self));
    const char *rep = getenv("FSR4_REPLACE");
    if (rep && rep[0] == '0') fsr4_live_set_inactive("FSR4_REPLACE=0");
    else if (fsr4_replace_encode(self, commandBuffer))
        return;
    if (!orig) {
        hook_log("encode: no original IMP on %s", class_getName(object_getClass(self)));
        return;
    }
    ((void (*)(id, SEL, id))orig)(self, _cmd, commandBuffer);
}

static void install_encode(Class cls) {
    if (!cls || orig_encode(cls)) return;
    SEL sel = @selector(encodeToCommandBuffer:);
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) {
        hook_log("no encodeToCommandBuffer: on %s", class_getName(cls));
        return;
    }
    int slot = -1;
    for (int i = 0; i < 4; i++)
        if (!g_encode[i].cls) { slot = i; break; }
    if (slot < 0) {
        hook_log("encode swizzle table full");
        return;
    }
    g_encode[slot].cls = cls;
    g_encode[slot].imp = method_getImplementation(m);
    method_setImplementation(m, (IMP)hooked_encode);
    hook_log("swizzled %s encodeToCommandBuffer:", class_getName(cls));
}

static id hooked_new(id self, SEL _cmd, id device) {
    id scaler = ((id (*)(id, SEL, id))g_orig_new)(self, _cmd, device);
    if (scaler) {
        remember_flags(scaler, self);
        install_encode(object_getClass(scaler));
    }
    return scaler;
}

static id hooked_new4(id self, SEL _cmd, id device, id compiler) {
    id scaler = ((id (*)(id, SEL, id, id))g_orig_new4)(self, _cmd, device, compiler);
    if (scaler) {
        remember_flags(scaler, self);
        install_encode(object_getClass(scaler));
    }
    return scaler;
}

static void swizzle_new(Class cls, SEL sel, IMP repl, IMP *slot) {
    Method m = class_getInstanceMethod(cls, sel);
    if (!m) return;
    *slot = method_getImplementation(m);
    method_setImplementation(m, repl);
    hook_log("swizzled %s %s", class_getName(cls), sel_getName(sel));
}

__attribute__((constructor))
static void fsr4_hook_init(void) {
    @autoreleasepool {
    hook_log("loaded pid=%d", getpid());
    Class desc = objc_getClass("MTLFXTemporalScalerDescriptor");
    if (!desc) {
        hook_log("MTLFXTemporalScalerDescriptor missing");
        return;
    }
    swizzle_new(desc, @selector(newTemporalScalerWithDevice:), (IMP)hooked_new, &g_orig_new);
    swizzle_new(desc, @selector(newTemporalScalerWithDevice:compiler:), (IMP)hooked_new4, &g_orig_new4);
    }
}
