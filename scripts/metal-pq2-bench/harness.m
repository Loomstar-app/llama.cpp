#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
// args: rows k
int main(int argc,char**argv){@autoreleasepool{
 int M=atoi(argv[1]), K=atoi(argv[2]); int nb=K/128; size_t tb=(size_t)M*nb*34;
 int NC = (int)((size_t)1600000000/tb); if(NC<2)NC=2; // copies to defeat SLC
 id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); NSError*e=nil;
 NSString*src=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[3]] encoding:NSUTF8StringEncoding error:&e];
 id<MTLLibrary> lib=[dev newLibraryWithSource:src options:nil error:&e]; if(!lib){NSLog(@"%@",e);return 1;}
 // build stock data and repacked data
 uint8_t* st=malloc(tb); uint8_t* rp=malloc(tb); srand(1);
 for(int r=0;r<M;r++) for(int b=0;b<nb;b++){ uint8_t* p=st+((size_t)r*nb+b)*34; _Float16 d=(_Float16)((rand()%1000)/1000.0f*0.02f+0.001f); memcpy(p,&d,2);
   for(int i=0;i<32;i++){ uint8_t v=0; for(int f=0;f<4;f++) v|=(rand()%3)<<(2*f); p[2+i]=v; }
   memcpy(rp+(size_t)r*nb*32+b*32,p+2,32); memcpy(rp+(size_t)M*nb*32+((size_t)r*nb+b)*2,p,2); }
 id<MTLBuffer> bs[2][64]; int nc= NC>64?64:NC;
 for(int c=0;c<nc;c++){ bs[0][c]=[dev newBufferWithBytes:st length:tb options:MTLResourceStorageModeShared]; bs[1][c]=[dev newBufferWithBytes:rp length:tb options:MTLResourceStorageModeShared]; }
 float* yv=malloc(K*4); for(int i=0;i<K;i++) yv[i]=(rand()%2000-1000)/1000.0f;
 id<MTLBuffer> yb=[dev newBufferWithBytes:yv length:K*4 options:MTLResourceStorageModeShared];
 id<MTLBuffer> ob=[dev newBufferWithLength:M*4 options:MTLResourceStorageModeShared];
 // reference (double) for correctness
 double* ref=malloc(M*8); for(int r=0;r<M;r++){ double s=0; for(int b=0;b<nb;b++){ uint8_t*p=st+((size_t)r*nb+b)*34; _Float16 d; memcpy(&d,p,2); double a=0;
   for(int i=0;i<32;i++) for(int f=0;f<4;f++) a+=(double)(((p[2+i]>>(2*f))&3)-1)*yv[b*128+i*4+f]; s+=a*(double)d;} ref[r]=s; }
 id<MTLCommandQueue> q=[dev newCommandQueue];
 struct {int nb; int nrows;} A={nb,M};
 const char* names[]={"stock8","stock8_load","stock8_math","rp16_r2","rp16_r4","rp16_r8","rp8_r2","rp8_r4","rp8_r8","rp16_r4_load","alt_mask","alt_half","alt_half_exact"};
 int nr0s[]={8,8,8,2,4,8,2,4,8,4,8,8,8}; int lay[]={0,0,0,1,1,1,1,1,1,1,0,0,0};
 for(int ki=0;ki<13;ki++){ id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:names[ki]]] error:&e];
  for(int nsg=1;nsg<=4;nsg*=2){ if(M%(nr0s[ki]*nsg)) continue;
   double tot=0; int it=0; double err=0;
   for(int rep=0;rep<3;rep++){ id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    for(int c=0;c<nc;c++){ [en setComputePipelineState:p]; [en setBytes:&A length:8 atIndex:0]; [en setBuffer:bs[lay[ki]][c] offset:0 atIndex:1]; [en setBuffer:yb offset:0 atIndex:2]; [en setBuffer:ob offset:0 atIndex:3];
     [en dispatchThreadgroups:MTLSizeMake(M/(nr0s[ki]*nsg),1,1) threadsPerThreadgroup:MTLSizeMake(32,nsg,1)]; [en memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
    [en endEncoding]; [cb commit]; [cb waitUntilCompleted]; if(rep>0){tot+=cb.GPUEndTime-cb.GPUStartTime; it+=nc;} }
   float*o=ob.contents; for(int r=0;r<M;r++){ double dlt=fabs(o[r]-ref[r]); if(dlt>err)err=dlt; }
   double us=tot/it*1e6; printf("%-14s nsg=%d  %8.1f us  %6.1f GB/s  maxerr=%.2e\n",names[ki],nsg,us,tb/(us*1e-6)/1e9,err); }}
}}
