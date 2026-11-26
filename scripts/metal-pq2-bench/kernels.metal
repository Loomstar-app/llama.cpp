#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
struct blk { half d; uchar qs[32]; };
struct A { int nb; int nrows; };

// MODE 0 full math (stock), 1 loads only, 2 math only (no weight loads)
template<int NR0, int MODE>
kernel void stock(constant A& a[[buffer(0)]], device const char* w[[buffer(1)]], device const float* y[[buffer(2)]], device float* dst[[buffer(3)]],
  uint3 tg[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sg[[simdgroup_index_in_threadgroup]], ushort nsg[[simdgroups_per_threadgroup]]) {
  const int first = (tg.x*nsg + sg)*NR0;
  device const blk* ax[NR0];
  for (int r=0;r<NR0;r++) ax[r] = (device const blk*)(w + (uint64_t)(first+r)*a.nb*34);
  float yl[16]; float sumf[NR0]={0.f};
  const short ix = tiisg/8; const short il=(tiisg%8)*16;
  device const float* yb = y + ix*128 + il;
  for (int ib=ix; ib<a.nb; ib+=4) {
    float sumy=0.f;
    FOR_UNROLL(short j=0;j<4;j++){ float y0=yb[4*j],y1=yb[4*j+1],y2=yb[4*j+2],y3=yb[4*j+3]; sumy+=(y0+y1)+(y2+y3);
      yl[4*j]=y3-4.f*y2; yl[4*j+1]=y2-4.f*y1; yl[4*j+2]=y1-4.f*y0; yl[4*j+3]=y0; }
    FOR_UNROLL(short r=0;r<NR0;r++){
      device const blk* qb = ax[r]+ib;
      if (MODE==1) { device const uchar* qs=qb->qs+il/4; sumf[r] += (float)qs[0]+(float)qs[1]+(float)qs[2]+(float)qs[3] + (float)qb->d; continue; }
      float acc=0.f; float d;
      uchar bs[4];
      if (MODE==2) { FOR_UNROLL(short j=0;j<4;j++) bs[j]=(uchar)(ib*7+r*3+j+il); d=1.0f; }
      else { device const uchar* qs=qb->qs+il/4; FOR_UNROLL(short j=0;j<4;j++) bs[j]=qs[j]; d=qb->d; }
      FOR_UNROLL(short j=0;j<4;j++){ const float b=(float)bs[j]; const float u=b*(1.0f/256.0f);
        acc+=floor(4.f*u)*yl[4*j]; acc+=floor(16.f*u)*yl[4*j+1]; acc+=floor(64.f*u)*yl[4*j+2]; acc+=b*yl[4*j+3]; }
      sumf[r]+= d*(acc-sumy);
    }
    yb += 128*4;
  }
  for (int r=0;r<NR0;r++){ float t=simd_sum(sumf[r]); if(tiisg==0) dst[first+r]=t; }
}
template [[host_name("stock8")]] kernel void stock<8,0>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
template [[host_name("stock8_load")]] kernel void stock<8,1>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
template [[host_name("stock8_math")]] kernel void stock<8,2>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);

// repacked: [qs of all rows, row stride nb*32][d of all rows, row stride nb]. Lane handles CH bytes (CH=8 uint2 or 16 uint4) per step.
template<int NR0, int CH, int MODE>
kernel void rp(constant A& a[[buffer(0)]], device const char* w[[buffer(1)]], device const float* y[[buffer(2)]], device float* dst[[buffer(3)]],
  uint3 tg[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sg[[simdgroup_index_in_threadgroup]], ushort nsg[[simdgroups_per_threadgroup]]) {
  const int first = (tg.x*nsg + sg)*NR0;
  const int rowq = a.nb*32;
  device const half* dd = (device const half*)(w + (uint64_t)a.nrows*rowq);
  float sumf[NR0]={0.f};
  const int nch = rowq/CH;           // chunks per row
  const int cpb = 32/CH;             // chunks per block
  for (int c = tiisg; c < nch; c += 32) {
    const int ib = c / cpb;
    device const float* yb = y + c*CH*4;
    float yl[CH*4]; float sumy=0.f;
    FOR_UNROLL(short j=0;j<CH;j++){ float y0=yb[4*j],y1=yb[4*j+1],y2=yb[4*j+2],y3=yb[4*j+3]; sumy+=(y0+y1)+(y2+y3);
      yl[4*j]=y3-4.f*y2; yl[4*j+1]=y2-4.f*y1; yl[4*j+2]=y1-4.f*y0; yl[4*j+3]=y0; }
    FOR_UNROLL(short r=0;r<NR0;r++){
      device const char* q = w + (uint64_t)(first+r)*rowq + c*CH;
      uchar bs[CH];
      if (CH==16) { uint4 v = *(device const uint4*)q; FOR_UNROLL(short j=0;j<16;j++) bs[j]=(uchar)(v[j/4]>>(8*(j%4))); }
      else { uint2 v = *(device const uint2*)q; FOR_UNROLL(short j=0;j<8;j++) bs[j]=(uchar)(v[j/4]>>(8*(j%4))); }
      const float d = dd[(uint64_t)(first+r)*a.nb + ib];
      if (MODE==1) { float s=0; FOR_UNROLL(short j=0;j<CH;j++) s+=(float)bs[j]; sumf[r]+=s+d; continue; }
      float acc=0.f;
      FOR_UNROLL(short j=0;j<CH;j++){ const float b=(float)bs[j]; const float u=b*(1.0f/256.0f);
        acc+=floor(4.f*u)*yl[4*j]; acc+=floor(16.f*u)*yl[4*j+1]; acc+=floor(64.f*u)*yl[4*j+2]; acc+=b*yl[4*j+3]; }
      sumf[r]+= d*(acc-sumy);
    }
  }
  for (int r=0;r<NR0;r++){ float t=simd_sum(sumf[r]); if(tiisg==0) dst[first+r]=t; }
}
#define RP(N,C,M,NAME) template [[host_name(NAME)]] kernel void rp<N,C,M>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
RP(2,16,0,"rp16_r2") RP(4,16,0,"rp16_r4") RP(8,16,0,"rp16_r8") RP(4,8,0,"rp8_r4") RP(8,8,0,"rp8_r8") RP(2,8,0,"rp8_r2")
RP(4,16,1,"rp16_r4_load")

// cheaper-ALU decodes on the stock layout. ALT 1: int mask per field + byte cvt (y staged raw, not as coefficients).
// ALT 2: half magic 0x6400|q -> 1024+q in half, offset folded into sumy (fp32): not bit-identical.
template<int NR0, int ALT>
kernel void alt(constant A& a[[buffer(0)]], device const char* w[[buffer(1)]], device const float* y[[buffer(2)]], device float* dst[[buffer(3)]],
  uint3 tg[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sg[[simdgroup_index_in_threadgroup]], ushort nsg[[simdgroups_per_threadgroup]]) {
  const int first = (tg.x*nsg + sg)*NR0;
  device const blk* ax[NR0];
  for (int r=0;r<NR0;r++) ax[r] = (device const blk*)(w + (uint64_t)(first+r)*a.nb*34);
  float yl[16]; float sumf[NR0]={0.f};
  const short ix = tiisg/8; const short il=(tiisg%8)*16;
  device const float* yb = y + ix*128 + il;
  for (int ib=ix; ib<a.nb; ib+=4) {
    float sumy=0.f;
    FOR_UNROLL(short j=0;j<16;j++){ yl[j]=yb[j]; sumy+=yb[j]; }
    if (ALT==2) sumy *= 1025.0f; if (ALT==3) sumy = 0.f; // codes come as 1024+q; value q-1
    FOR_UNROLL(short r=0;r<NR0;r++){
      device const blk* qb = ax[r]+ib;
      device const ushort* qs=(device const ushort*)(qb->qs+il/4);
      const uint wv = (uint)qs[0] | ((uint)qs[1] << 16);
      float acc=0.f;
      if (ALT==1) {
        FOR_UNROLL(short f=0;f<4;f++){ const uchar4 m = as_type<uchar4>((wv >> (2*f)) & 0x03030303u);
          FOR_UNROLL(short j=0;j<4;j++) acc = fma((float)m[j], yl[4*j+f], acc); }
      } else {
        FOR_UNROLL(short f=0;f<4;f++){
          half2 lo = as_type<half2>(((wv >> (2*f)) & 0x00030003u) | 0x64006400u);        // bytes 0,2
          half2 hi = as_type<half2>(((wv >> (2*f+8)) & 0x00030003u) | 0x64006400u);      // bytes 1,3
          if (ALT==3) { lo -= half2(1025.0h); hi -= half2(1025.0h); }
          acc = fma((float)lo[0], yl[0*4+f], acc); acc = fma((float)hi[0], yl[1*4+f], acc);
          acc = fma((float)lo[1], yl[2*4+f], acc); acc = fma((float)hi[1], yl[3*4+f], acc); }
      }
      sumf[r]+= (float)qb->d*(acc-sumy);
    }
    yb += 128*4;
  }
  for (int r=0;r<NR0;r++){ float t=simd_sum(sumf[r]); if(tiisg==0) dst[first+r]=t; }
}
template [[host_name("alt_mask")]] kernel void alt<8,1>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
template [[host_name("alt_half")]] kernel void alt<8,2>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
template [[host_name("alt_half_exact")]] kernel void alt<8,3>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
