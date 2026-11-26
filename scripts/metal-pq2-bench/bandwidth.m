#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
int main(){@autoreleasepool{
 id<MTLDevice> d=MTLCreateSystemDefaultDevice();
 NSString*src=@"#include <metal_stdlib>\nusing namespace metal;\n"
 "kernel void rd(device const uint4* a[[buffer(0)]], device uint* o[[buffer(1)]], constant uint& n[[buffer(2)]], uint gid[[thread_position_in_grid]], uint gs[[threads_per_grid]]){uint4 s=0; for(uint i=gid;i<n;i+=gs){s+=a[i];} if((s.x^s.y^s.z^s.w)==0x12345678u) o[0]=1;}\n"
 "kernel void rd4(device const uint* a[[buffer(0)]], device uint* o[[buffer(1)]], constant uint& n[[buffer(2)]], uint gid[[thread_position_in_grid]], uint gs[[threads_per_grid]]){uint s=0; for(uint i=gid;i<n*4;i+=gs){s+=a[i];} if(s==0x12345678u) o[0]=1;}\n";
 NSError*e=nil; id<MTLLibrary> lib=[d newLibraryWithSource:src options:nil error:&e]; if(!lib){NSLog(@"%@",e);return 1;}
 size_t bytes=(size_t)1<<31; uint n=(uint)(bytes/16);
 id<MTLBuffer> a=[d newBufferWithLength:bytes options:MTLResourceStorageModeShared]; memset(a.contents,1,bytes);
 id<MTLBuffer> o=[d newBufferWithLength:16 options:MTLResourceStorageModeShared];
 id<MTLCommandQueue> q=[d newCommandQueue];
 for (NSString* kn in @[@"rd",@"rd4"]) {
 id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:kn] error:&e];
 for(int tg=0;tg<3;tg++){ uint grid=(uint[]){16384,65536,262144}[tg];
  double best=1e9; for(int r=0;r<6;r++){
  id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
  [en setComputePipelineState:p]; [en setBuffer:a offset:0 atIndex:0]; [en setBuffer:o offset:0 atIndex:1]; [en setBytes:&n length:4 atIndex:2];
  [en dispatchThreads:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
  double t=cb.GPUEndTime-cb.GPUStartTime; if(r>0&&t<best)best=t;}
  printf("%s grid=%u  %.1f GB/s\n",kn.UTF8String,grid,bytes/best/1e9);}}
}}
