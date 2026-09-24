// Same sequence D3DMetal uses: descriptor -> scaler -> textures -> encode.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <stdio.h>

static id<MTLTexture> tex(id<MTLDevice> dev, NSUInteger w, NSUInteger h, MTLPixelFormat fmt, BOOL priv) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
    d.storageMode = priv ? MTLStorageModePrivate : MTLStorageModeShared;
    return [dev newTextureWithDescriptor:d];
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no device\n"); return 1; }
        if (![MTLFXTemporalScalerDescriptor supportsDevice:dev]) {
            fprintf(stderr, "temporal scaler unsupported\n");
            return 1;
        }
        MTLFXTemporalScalerDescriptor *desc = [MTLFXTemporalScalerDescriptor new];
        desc.inputWidth = 320;
        desc.inputHeight = 180;
        desc.outputWidth = 640;
        desc.outputHeight = 360;
        desc.colorTextureFormat = MTLPixelFormatRGBA16Float;
        desc.depthTextureFormat = MTLPixelFormatR32Float;
        desc.motionTextureFormat = MTLPixelFormatRG16Float;
        desc.outputTextureFormat = MTLPixelFormatRGBA16Float;
        id<MTLFXTemporalScaler> scaler = [desc newTemporalScalerWithDevice:dev];
        if (!scaler) { fprintf(stderr, "scaler nil\n"); return 1; }
        scaler.colorTexture = tex(dev, 320, 180, MTLPixelFormatRGBA16Float, NO);
        scaler.depthTexture = tex(dev, 320, 180, MTLPixelFormatR32Float, NO);
        scaler.motionTexture = tex(dev, 320, 180, MTLPixelFormatRG16Float, NO);
        scaler.outputTexture = tex(dev, 640, 360, MTLPixelFormatRGBA16Float, YES);
        scaler.inputContentWidth = 320;
        scaler.inputContentHeight = 180;
        scaler.jitterOffsetX = 0.1f;
        scaler.jitterOffsetY = -0.2f;
        scaler.motionVectorScaleX = 320;
        scaler.motionVectorScaleY = 180;
        scaler.reset = YES;
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [scaler encodeToCommandBuffer:cb];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "encode failed: %s\n", cb.error.localizedDescription.UTF8String);
            return 1;
        }
        printf("smoke ok class=%s\n", object_getClassName(scaler));
    }
    return 0;
}
