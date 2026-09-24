// Standalone harness for src/i8mpp.metal: run one model pass on a reference
// scratch dump and compare with the reference output dump; optional timing.
//   i8_test <gendir> <ref_dump_dir> <frame> <pass> [iters] [debugStage]
#import <Metal/Metal.h>
#include "../../src/i8mpp_host.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

static bool rd(const std::string &p, std::vector<uint8_t> &o) { return I8Mpp::read_all(p, o); }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: i8_test <gendir> <refdir> <frame> <pass> [iters] [debugStage]\n"); return 2; }
    @autoreleasepool {
        std::string gendir = argv[1], refdir = argv[2];
        int frame = atoi(argv[3]), pass = atoi(argv[4]);
        int iters = argc > 5 ? atoi(argv[5]) : 0;
        int dbg = argc > 6 ? atoi(argv[6]) : 0;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        I8Mpp mpp;
        if (dbg) setenv("FSR4_I8_DEBUG", std::to_string(dbg).c_str(), 1);
        std::string src = std::string(getenv("FSR4_ROOT") ? getenv("FSR4_ROOT") : ".") + "/src/i8mpp.metal";
        if (!mpp.load(dev, gendir, src)) { fprintf(stderr, "%s\n", mpp.error.c_str()); return 1; }
        char p[512];
        snprintf(p, sizeof p, "%s/f%02d_pass%02d_scratch.bin", refdir.c_str(), frame, pass - 1);
        std::vector<uint8_t> in, ref;
        if (!rd(p, in)) { fprintf(stderr, "no %s\n", p); return 1; }
        snprintf(p, sizeof p, "%s/f%02d_pass%02d_scratch.bin", refdir.c_str(), frame, pass);
        if (!rd(p, ref)) { fprintf(stderr, "no %s\n", p); return 1; }
        id<MTLBuffer> scratch = [dev newBufferWithLength:in.size() + i8mpp_params_bytes(2) options:MTLResourceStorageModeShared];
        memcpy(scratch.contents, in.data(), in.size());
        // pass 13 writes the recurrent texture (not part of the reference dumps)
        id<MTLTexture> tex = nil;
        {
            const I8P &q = mpp.passes.at(pass).p;
            if (q.prmOff) {
                MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                             width:q.prmW height:q.prmH mipmapped:NO];
                d.usage = MTLTextureUsageShaderWrite;
                tex = [dev newTextureWithDescriptor:d];
            }
        }
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        mpp.dispatch(pass, enc, scratch, MTLBarrierScopeBuffers, tex);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) { fprintf(stderr, "gpu error: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        const I8P &pp = mpp.passes.at(pass).p;
        const uint8_t *got = (const uint8_t *)scratch.contents;
        size_t bad = 0, tot = 0; int maxd = 0;
        for (uint32_t y = 0; y < pp.outH; y++)
            for (uint32_t x = 0; x < pp.outW * pp.outSX; x++) {
                size_t i = pp.outOff + (size_t)y * pp.outSY + x;
                int d = abs((int)(int8_t)got[i] - (int)(int8_t)ref[i]);
                tot++;
                if (d) bad++;
                if (d > maxd) maxd = d;
            }
        printf("pass %d frame %d: %zu/%zu mismatches, max %d\n", pass, frame, bad, tot, maxd);
        if (const char *dp = getenv("FSR4_I8_DUMP")) {
            // whole scratch after the pass, for external (per-column) analysis
            FILE *f = fopen(dp, "wb");
            if (f) { fwrite(got, 1, in.size(), f); fclose(f); printf("  scratch dump %s\n", dp); }
        }
        if (const char *tp = getenv("FSR4_I8_DUMP_TEX")) {
            if (tex) {
                std::vector<uint8_t> t((size_t)tex.width * tex.height * 4);
                [tex getBytes:t.data() bytesPerRow:tex.width * 4 fromRegion:MTLRegionMake2D(0, 0, tex.width, tex.height) mipmapLevel:0];
                FILE *f = fopen(tp, "wb");
                if (f) { fwrite(t.data(), 1, t.size(), f); fclose(f); printf("  texture dump %s (%lux%lu RGBA8)\n", tp, (unsigned long)tex.width, (unsigned long)tex.height); }
            }
        }
        if (bad) {
            size_t i = pp.outOff + (size_t)5 * pp.outSY + 5 * pp.outSX;
            printf("  (5,5) ref:"); for (int c = 0; c < 8; c++) printf(" %d", (int8_t)ref[i + c]);
            printf("   got:"); for (int c = 0; c < 8; c++) printf(" %d", (int8_t)got[i + c]);
            printf("\n");
        }
        if (dbg) {
            // dump the output tensor region for external comparison
            snprintf(p, sizeof p, "/tmp/i8_dbg_pass%02d_stage%d.bin", pass, dbg);
            FILE *f = fopen(p, "wb");
            fwrite(got, 1, in.size(), f);
            fclose(f);
            printf("  debug dump %s\n", p);
        }
        if (iters > 0) {
            double best = 1e9;
            for (int it = 0; it < iters; it++) {
                id<MTLCommandBuffer> tb = [q commandBuffer];
                id<MTLComputeCommandEncoder> te = [tb computeCommandEncoder];
                for (int r = 0; r < 10; r++) mpp.dispatch(pass, te, scratch, MTLBarrierScopeBuffers, tex);
                [te endEncoding];
                [tb commit];
                [tb waitUntilCompleted];
                double ms = (tb.GPUEndTime - tb.GPUStartTime) * 1000.0 / 10.0;
                if (ms < best) best = ms;
            }
            printf("pass %d: %.4f ms (best of %d x10)\n", pass, best, iters);
        }
    }
    return 0;
}
