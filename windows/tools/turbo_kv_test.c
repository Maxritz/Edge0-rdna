/* turbo_kv.c - verify the TurboQuant KV quant/dequant round-trip (FWHT + codebook).
 * Mirror of ggml-quants.c quantize_row_turbo3/4_ref / dequantize_row_turbo3/4.
 * Build: gcc -O2 -o turbo_kv turbo_kv.c -lm
 */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

typedef uint16_t ggml_half;
static float h2f(ggml_half h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, b;
    if (e == 0) { if (m == 0) b = s; else { int ex = 127-15+1; uint32_t mm=m; while(!(mm&0x400)){mm<<=1;ex--;} b=s|((uint32_t)ex<<23)|((mm&0x3ff)<<13);} }
    else if (e == 0x1f) b = s | 0x7f800000u | (m<<13);
    else b = s | ((e+127-15)<<23) | (m<<13);
    float f; memcpy(&f,&b,4); return f;
}
static ggml_half f2h(float x) {
    uint32_t b; memcpy(&b,&x,4);
    uint32_t s=(b>>16)&0x8000, e=((b>>23)&0xff)-127+15, m=b&0x7fffff;
    if (e <= 0) { if (e < -10) return (ggml_half)s; uint32_t mm=m|0x800000; uint32_t sh=14-e; uint32_t h=mm>>sh; uint32_t rem=mm&((1u<<sh)-1), mid=1u<<(sh-1); if(rem>mid||(rem==mid&&(h&1)))h++; return (ggml_half)(s|h);} 
    if (e >= 0x1f) return (ggml_half)(s|0x7c00);
    uint32_t mm=m>>13; uint32_t rem=m&0x1fff; if(rem>0x1000||(rem==0x1000&&(mm&1))){mm++; if(mm==0x400){mm=0;e++; if(e>=0x1f) return (ggml_half)(s|0x7c00);}}
    return (ggml_half)(s|(e<<10)|mm);
}

static void fwht(float *x, int n){ for(int h=1;h<n;h<<=1) for(int i=0;i<n;i+=2*h) for(int j=i;j<i+h;j++){float a=x[j],b=x[j+h];x[j]=a+b;x[j+h]=a-b;} float s=1.0f/sqrtf((float)n); for(int i=0;i<n;i++)x[i]*=s; }

static const float CB3[8]={-0.1883972972f,-0.1181399059f,-0.0665857641f,-0.0216044751f,0.0216041461f,0.0665854520f,0.1181396281f,0.1883970748f};
static const float CB4[16]={-0.2376389871f,-0.1808080141f,-0.1417777640f,-0.1102646123f,-0.0828112376f,-0.0577640422f,-0.0341540905f,-0.0113168380f,0.0112761586f,0.0341139667f,0.0577250301f,0.0827738972f,0.1102295202f,0.1417455465f,0.1807794468f,0.2376153882f};
static uint8_t near3(float v){uint8_t b=0;float bd=fabsf(CB3[0]-v);for(int i=1;i<8;i++){float d=fabsf(CB3[i]-v);if(d<bd){bd=d;b=i;}}return b;}
static uint8_t near4(float v){uint8_t b=0;float bd=fabsf(CB4[0]-v);for(int i=1;i<16;i++){float d=fabsf(CB4[i]-v);if(d<bd){bd=d;b=i;}}return b;}

#define QK 128
typedef struct { ggml_half norm; uint8_t qs[48]; } b3;
typedef struct { ggml_half norm; uint8_t qs[64]; } b4;

static void q3(const float*x, b3*y){ float nrm=0; for(int c=0;c<QK;c++)nrm+=x[c]*x[c]; nrm=sqrtf(nrm); y->norm=f2h(nrm); float inv=nrm>1e-10f?1.0f/nrm:0; float t[QK]; for(int c=0;c<QK;c++)t[c]=x[c]*inv; fwht(t,QK); memset(y->qs,0,48); for(int c=0;c<QK;c++){int idx=near3(t[c]);int bit=c*3,by=bit>>3,sh=bit&7;y->qs[by]|=(uint8_t)(idx<<sh); if(sh>5&&by+1<48)y->qs[by+1]|=(uint8_t)(idx>>(8-sh));}}
static void d3(const b3*x,float*y){ float t[QK]; for(int c=0;c<QK;c++){int bit=c*3,by=bit>>3,sh=bit&7;int idx=x->qs[by]>>sh; if(sh>5&&by+1<48)idx|=(x->qs[by+1]<<(8-sh)); t[c]=CB3[idx&7];} fwht(t,QK); float n=h2f(x->norm); for(int c=0;c<QK;c++)y[c]=t[c]*n; }
static void q4(const float*x, b4*y){ float nrm=0; for(int c=0;c<QK;c++)nrm+=x[c]*x[c]; nrm=sqrtf(nrm); y->norm=f2h(nrm); float inv=nrm>1e-10f?1.0f/nrm:0; float t[QK]; for(int c=0;c<QK;c++)t[c]=x[c]*inv; fwht(t,QK); memset(y->qs,0,64); for(int c=0;c<QK;c++){int idx=near4(t[c]); if(c&1)y->qs[c>>1]|=(uint8_t)(idx<<4); else y->qs[c>>1]|=(uint8_t)idx;}}
static void d4(const b4*x,float*y){ float t[QK]; for(int c=0;c<QK;c++){uint8_t by=x->qs[c>>1];int idx=(c&1)?(by>>4):(by&0x0f); t[c]=CB4[idx];} fwht(t,QK); float n=h2f(x->norm); for(int c=0;c<QK;c++)y[c]=t[c]*n; }

static uint32_t st=0x12345678;
static float rnd(){ st^=st<<13; st^=st>>17; st^=st<<5; return ((float)st/(float)UINT32_MAX)*2.0f-1.0f; }

int main(void){
    printf("TurboQuant KV round-trip (gcc, CPU)\n");
    float x[256], y[256];
    for(int i=0;i<256;i++)x[i]=rnd();
    float e3=0,e4=0;
    for(int b=0;b<2;b++){ b3 z; q3(&x[b*QK],&z); d3(&z,&y[b*QK]); }
    for(int i=0;i<256;i++){float a=fabsf(x[i]-y[i]); if(a>e3)e3=a;}
    for(int b=0;b<2;b++){ b4 z; q4(&x[b*QK],&z); d4(&z,&y[b*QK]); }
    for(int i=0;i<256;i++){float a=fabsf(x[i]-y[i]); if(a>e4)e4=a;}
    printf("  turbo3 max_abs_err=%.5f  %s\n", e3, e3<0.45f?"PASS":"FAIL");
    printf("  turbo4 max_abs_err=%.5f  %s\n", e4, e4<0.25f?"PASS":"FAIL");
    return (e3<0.45f && e4<0.25f)?0:1;
}
