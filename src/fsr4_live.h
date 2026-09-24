#pragma once
#include <stdint.h>

// One FSR4 fp16 frame encoded into the caller's command buffer. The encoder
// does not commit. Textures are retained by the caller for the life of the
// buffer. Returns 1 if FSR4 consumed the encode, 0 to keep the original scaler.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Fsr4LiveFrame {
    void *device;
    void *commandBuffer;
    // Identity of the MetalFX scaler object. History, recurrent state and
    // per-frame bookkeeping live per owner: D3DMetal encodes more than one
    // scaler per game frame and they must not share a history buffer.
    void *owner;
    void *color, *depth, *motion, *output, *exposure;
    uint32_t rw, rh, uw, uh;
    float jitterX, jitterY;
    float mvScaleX, mvScaleY;
    float preExposure;
    int reset;
    int depthReversed;
    // Descriptor flags as D3DMetal configured the scaler (-1 = unknown).
    int jitteredMV;      // motion vectors include the jitter pattern
    int outputResMV;     // motion texture is at output resolution
    int autoExposure;    // scaler computes exposure itself (no exposure texture)
    uint32_t mvOffX, mvOffY;   // motionContentOffset (subrect origin), 0 normally
} Fsr4LiveFrame;

int fsr4_encode_live_frame(const Fsr4LiveFrame *frame);

// Legacy entry point (owner = NULL, flags unknown).
int fsr4_encode_live(void *device, void *commandBuffer,
                     void *color, void *depth, void *motion, void *output,
                     void *exposure,
                     uint32_t rw, uint32_t rh, uint32_t uw, uint32_t uh,
                     float jitterX, float jitterY,
                     float mvScaleX, float mvScaleY,
                     float preExposure, int reset, int depthReversed);

// --- status / settings for the panel (src/fsr4_panel.mm) and the hook ---

typedef struct Fsr4LiveStatus {
    int active;              // the last real encode ran FSR4
    char reason[96];         // why not, when !active
    char preset[16];
    char model[8];           // i8mpp | f16
    char variant[96];        // shader variant key
    uint32_t rw, rh, uw, uh, uwA, uhA;
    float frameMs;           // EMA of the interval between real encodes (game frame time)
    uint64_t encodes, replaced, stubs;
    float sharpness;         // effective RCAS sharpness (0 = off)
} Fsr4LiveStatus;

void fsr4_live_get_status(Fsr4LiveStatus *out);
// Mark FSR4 inactive with a reason (hook: no textures, FSR4_REPLACE=0).
void fsr4_live_set_inactive(const char *reason);

// RCAS sharpness setting: FSR scale 0..1 plus an on/off flag. Defaults come
// from FSR4_SHARPNESS / FSR4_RCAS, then dist/crossover/panel.conf (written
// by the panel) overrides them; FSR4_SETTINGS=none ignores the file.
void fsr4_live_get_sharpness(float *sharpness, int *rcas);
void fsr4_live_set_sharpness(float sharpness, int rcas);

// FSR4_PANEL=1: launch dist/crossover/FSR4 Panel.app once (src/fsr4_panel.mm).
void fsr4_panel_request(void);
// Publish the status for the panel app to dist/crossover/status.conf (2 Hz).
void fsr4_panel_publish(void);

#ifdef __cplusplus
}
#endif
