#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
struct blk { half d; uchar qs[32]; };
struct A { int nb; int nrows; };

// DEC 0: stock floor chain (coef staging); 1: half magic + exact half sub; 2: half magic, offset in fp32
//     3: g-chain via int shifts/masks + cvt (coef staging, bit-different but exact ints)
//     4: half magic exact, but subtract done on y side: sum (1024+q)*y computed as fma on (h-1024) in half via fma(h, y) - 1024*y? (no) -> 4 = per-field int mask + cvt raw y
template<int DEC>
inline float dotw(uint wv, thread const float* yl, float sumy) {
  float acc = 0.f;
  if (DEC==0) {
    FOR_UNROLL(short j=0;j<4;j++){ const float b=(float)((wv>>(8*j))&0xFF); const float u=b*(1.0f/256.0f);
      acc+=floor(4.f*u)*yl[4*j]; acc+=floor(16.f*u)*yl[4*j+1]; acc+=floor(64.f*u)*yl[4*j+2]; acc+=b*yl[4*j+3]; }
    return acc - sumy;
  } else if (DEC==1 || DEC==2) {
    FOR_UNROLL(short f=0;f<4;f++){
      half2 lo = as_type<half2>(((wv >> (2*f)) & 0x00030003u) | 0x64006400u);
      half2 hi = as_type<half2>(((wv >> (2*f+8)) & 0x00030003u) | 0x64006400u);
      if (DEC==1) { lo -= half2(1025.0h); hi -= half2(1025.0h); }
      acc = fma((float)lo[0], yl[0*4+f], acc); acc = fma((float)hi[0], yl[1*4+f], acc);
      acc = fma((float)lo[1], yl[2*4+f], acc); acc = fma((float)hi[1], yl[3*4+f], acc); }
    return DEC==1 ? acc : acc - sumy;
  } else if (DEC==3) {
    const uchar4 g1 = as_type<uchar4>((wv >> 6) & 0x03030303u);
    const uchar4 g2 = as_type<uchar4>((wv >> 4) & 0x0F0F0F0Fu);
    const uchar4 g3 = as_type<uchar4>((wv >> 2) & 0x3F3F3F3Fu);
    const uchar4 b  = as_type<uchar4>(wv);
    FOR_UNROLL(short j=0;j<4;j++){ acc=fma((float)g1[j],yl[4*j],acc); acc=fma((float)g2[j],yl[4*j+1],acc); acc=fma((float)g3[j],yl[4*j+2],acc); acc=fma((float)b[j],yl[4*j+3],acc); }
    return acc - sumy;
  } else if (DEC==6) {
    // fp16 subnormals: a 2-bit field masked into mantissa bits 0-1 of a half lane is exactly q*2^-24, at bits 8-9
    // exactly q*2^-16, with no offset to cancel. One shift serves both byte pairs: (wv>>2f) holds field f of bytes
    // 0,2 at bits 0-1 / 16-17 and of bytes 1,3 at bits 8-9 / 24-25.
    float al = 0.f, ah = 0.f;
    FOR_UNROLL(short f=0;f<4;f++){
      const uint x = wv >> (2*f);
      const half2 lo = as_type<half2>(x & 0x00030003u);   // bytes 0,2
      const half2 hi = as_type<half2>(x & 0x03000300u);   // bytes 1,3
      al = fma((float)lo[0], yl[0*4+f], al); ah = fma((float)hi[0], yl[1*4+f], ah);
      al = fma((float)lo[1], yl[2*4+f], al); ah = fma((float)hi[1], yl[3*4+f], ah); }
    return fma(al, 16777216.0f, fma(ah, 65536.0f, -sumy));
  } else if (DEC==5) {
    // half 1 + q/4: field moved to mantissa bits 8-9 of each 16-bit lane; sum (q-1)*y = 4*acc - 5*sumy
    FOR_UNROLL(short f=0;f<4;f++){
      half2 lo = as_type<half2>(((wv << (8-2*f)) & 0x03000300u) | 0x3C003C00u);   // bytes 0,2
      half2 hi = as_type<half2>(((wv >> (2*f)) & 0x03000300u) | 0x3C003C00u);     // bytes 1,3
      acc = fma((float)lo[0], yl[0*4+f], acc); acc = fma((float)hi[0], yl[1*4+f], acc);
      acc = fma((float)lo[1], yl[2*4+f], acc); acc = fma((float)hi[1], yl[3*4+f], acc); }
    return 4.0f*acc - 5.0f*sumy;
  } else { // 4: half magic on g-chain values (<=255 exact in half 1024+x), exact sub in half
    const uint g1 = (wv >> 6) & 0x03030303u, g2 = (wv >> 4) & 0x0F0F0F0Fu, g3 = (wv >> 2) & 0x3F3F3F3Fu;
    const uint gs[4] = {g1,g2,g3,wv};
    FOR_UNROLL(short k=0;k<4;k++){
      half2 lo = as_type<half2>((gs[k] & 0x00FF00FFu) | 0x64006400u) - half2(1024.0h);
      half2 hi = as_type<half2>(((gs[k] >> 8) & 0x00FF00FFu) | 0x64006400u) - half2(1024.0h);
      acc=fma((float)lo[0],yl[0*4+k],acc); acc=fma((float)hi[0],yl[1*4+k],acc); acc=fma((float)lo[1],yl[2*4+k],acc); acc=fma((float)hi[1],yl[3*4+k],acc); }
    return acc - sumy;
  }
}

template<int NR0, int DEC, int PF>
kernel void mv(constant A& a[[buffer(0)]], device const char* w[[buffer(1)]], device const float* y[[buffer(2)]], device float* dst[[buffer(3)]],
  uint3 tg[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sg[[simdgroup_index_in_threadgroup]], ushort nsg[[simdgroups_per_threadgroup]]) {
  const int first = (tg.x*nsg + sg)*NR0;
  device const blk* ax[NR0];
  for (int r=0;r<NR0;r++) ax[r] = (device const blk*)(w + (uint64_t)(first+r)*a.nb*34);
  float yl[16]; float sumf[NR0]={0.f};
  const short ix = tiisg/8; const short il=(tiisg%8)*16;
  device const float* yb = y + ix*128 + il;
  uint wn[NR0]; half dn[NR0];
  if (PF) { FOR_UNROLL(short r=0;r<NR0;r++){ device const ushort* qs=(device const ushort*)(ax[r][ix].qs+il/4); wn[r]=(uint)qs[0]|((uint)qs[1]<<16); dn[r]=ax[r][ix].d; } }
  for (int ib=ix; ib<a.nb; ib+=4) {
    float sumy=0.f;
    if (DEC==1 || DEC==2 || DEC==5 || DEC==6) { FOR_UNROLL(short j=0;j<16;j++){ yl[j]=yb[j]; sumy+=yb[j]; } if (DEC==2) sumy*=1025.0f; }
    else { FOR_UNROLL(short j=0;j<4;j++){ float y0=yb[4*j],y1=yb[4*j+1],y2=yb[4*j+2],y3=yb[4*j+3]; sumy+=(y0+y1)+(y2+y3);
      yl[4*j]=y3-4.f*y2; yl[4*j+1]=y2-4.f*y1; yl[4*j+2]=y1-4.f*y0; yl[4*j+3]=y0; } }
    uint wc[NR0]; half dc[NR0];
    if (PF) {
      FOR_UNROLL(short r=0;r<NR0;r++){ wc[r]=wn[r]; dc[r]=dn[r]; }
      const int nx = ib+4 < a.nb ? ib+4 : ib;
      FOR_UNROLL(short r=0;r<NR0;r++){ device const ushort* qs=(device const ushort*)(ax[r][nx].qs+il/4); wn[r]=(uint)qs[0]|((uint)qs[1]<<16); dn[r]=ax[r][nx].d; }
    } else {
      FOR_UNROLL(short r=0;r<NR0;r++){ device const ushort* qs=(device const ushort*)(ax[r][ib].qs+il/4); wc[r]=(uint)qs[0]|((uint)qs[1]<<16); dc[r]=ax[r][ib].d; }
    }
    FOR_UNROLL(short r=0;r<NR0;r++) sumf[r] += (float)dc[r]*dotw<DEC>(wc[r], yl, sumy);
    yb += 128*4;
  }
  for (int r=0;r<NR0;r++){ float t=simd_sum(sumf[r]); if(tiisg==0) dst[first+r]=t; }
}
#define K(N,D,P) template [[host_name("mv_r" #N "_d" #D "_p" #P)]] kernel void mv<N,D,P>(constant A&, device const char*, device const float*, device float*, uint3, ushort, ushort, ushort);
K(8,0,0) K(8,0,1) K(8,1,0) K(8,1,1) K(8,2,0) K(8,2,1) K(8,3,0) K(8,3,1) K(8,4,0) K(8,4,1)
K(4,1,0) K(4,1,1) K(16,1,0) K(4,3,0) K(16,3,0) K(4,4,1) K(16,4,0)
K(8,5,0) K(4,5,0) K(16,5,0) K(2,5,0)
K(8,6,0) K(4,6,0) K(16,6,0)

// split-K: the threadgroup's simdgroups share NR0 rows, each takes every nsg-th group of 4 blocks; threadgroup reduce.
template<int NR0>
kernel void mvks(constant A& a[[buffer(0)]], device const char* w[[buffer(1)]], device const float* y[[buffer(2)]], device float* dst[[buffer(3)]],
  threadgroup float* sh[[threadgroup(0)]],
  uint3 tg[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sg[[simdgroup_index_in_threadgroup]], ushort nsg[[simdgroups_per_threadgroup]]) {
  const int first = tg.x*NR0;
  device const blk* ax[NR0];
  for (int r=0;r<NR0;r++) ax[r] = (device const blk*)(w + (uint64_t)(first+r)*a.nb*34);
  float yl[16]; float sumf[NR0]={0.f};
  const short ix = tiisg/8; const short il=(tiisg%8)*16;
  for (int ib=ix+4*sg; ib<a.nb; ib+=4*nsg) {
    device const float* yb = y + ib*128 + il;
    float sumy=0.f;
    FOR_UNROLL(short j=0;j<16;j++){ yl[j]=yb[j]; sumy+=yb[j]; }
    FOR_UNROLL(short r=0;r<NR0;r++){ device const ushort* qs=(device const ushort*)(ax[r][ib].qs+il/4); const uint wv=(uint)qs[0]|((uint)qs[1]<<16);
      sumf[r] += (float)ax[r][ib].d*dotw<6>(wv, yl, sumy); }
  }
  FOR_UNROLL(short r=0;r<NR0;r++){ float t=simd_sum(sumf[r]); if(tiisg==0) sh[sg*NR0+r]=t; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg==0 && tiisg<NR0) { float t=0; for(int s=0;s<nsg;s++) t+=sh[s*NR0+tiisg]; dst[first+tiisg]=t; }
}
template [[host_name("ks_r8")]] kernel void mvks<8>(constant A&, device const char*, device const float*, device float*, threadgroup float*, uint3, ushort, ushort, ushort);
template [[host_name("ks_r4")]] kernel void mvks<4>(constant A&, device const char*, device const float*, device float*, threadgroup float*, uint3, ushort, ushort, ushort);
template [[host_name("ks_r16")]] kernel void mvks<16>(constant A&, device const char*, device const float*, device float*, threadgroup float*, uint3, ushort, ushort, ushort);
