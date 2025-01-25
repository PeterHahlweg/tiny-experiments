# Kernel Optimisation Report - edge.py

Program: `python examples/edge_detection/edge.py --use-test-image`


## 1. Non-Optimized Compute Kernels

| Kernel | Shape | Memory (GB) | Time (μs) | GFLOPS | Bandwidth (GB/s) | Backend |
|---------|-------|-------------|------------|---------|-----------------|----------|
| Metal kernel | r_135_40_8_16_3_7 | 0.017 | 2396.46 | 11.25 | 6.92 | METAL |
| Metal kernel | r_135_40_8_16_3_7n1 | 0.025 | 3797.45 | 7.10 | 4.37 | METAL |
| Metal kernel | r_2_135_40_8_16_3_3_3 | 0.033 | 6619.79 | 10.65 | 3.76 | METAL |
| Metal kernel | E_135_40_8_16_3 | 0.033 | 9136.30 | 21.56 | 2.72 | METAL |
| Metal kernel | r_135_40_8_16_3_3_3 | 0.041 | 1528.23 | 19.00 | 27.14 | METAL |
| **Total Time** | | | 23.48 ms | | | |

## 2. Optimized Compute Kernels - BEAM 100

| Kernel | Shape | Memory (GB) | Time (μs) | GFLOPS | Bandwidth (GB/s) | Backend |
|---------|-------|-------------|------------|---------|-----------------|----------|
| Metal kernel | r_540_30_8_2_4_7_2 | 0.017 | 796.20 | 33.86 | 20.84 | METAL |
| Metal kernel | r_360_64_16_2_7_3 | 0.025 | 615.42 | 46.72 | 26.96 | METAL |
| Metal kernel | r_1080_24_16_3_3_2_5 | 0.033 | 1872.60 | 39.86 | 13.29 | METAL |
| Metal kernel | E_360_30_4_4_4_3 | 0.033 | 1662.45 | 107.27 | 14.97 | METAL |
| Metal kernel | r_270_15_16_2_8_2_3_3 | 0.041 | 496.25 | 58.50 | 83.57 | METAL |
| **Total Time** | | | 5.44 ms | | | |

## 3. Memory Transfer Operations

| Transfer Direction | Memory (GB) | Duration (μs) |
|-------------------|-------------|---------------|
| NPY <- METAL | 0.017 | 6332.44 |
| NPY <- METAL | 0.008 | 109.07 |
| NPY <- METAL | 0.008 | 38.21 |
| NPY <- METAL | 0.008 | 80.14 |
| NPY <- METAL | 0.008 | 72.65 |
| METAL <- NPY | 0.025 | 4324.97 |
| **Total** | 0.075 | 10957.48 |

## 4. Performance Analysis

| Metric | Value | Notes |
|--------|-------|-------|
| Complete Runtime | 16.40ms | Includes compute (5.44ms) and transfers (10.96ms) |
| Speed-up Factor | 2.08x | Total execution time improvement |
| Time Reduction | 17.77ms | Absolute time saved |
| Improvement | 52.0% | Reduction in execution time |
| Memory Impact | 0.224GB vs 0.224GB | Memory footprint comparison |
| GFLOPS | 286.21 vs 69.56 | Computational throughput |
| GFLOPS Improvement | 311.5% | Compute efficiency gain |

## Execution Statistics

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Best Runtime (μs) | 34166.21 | 16400.40 |
| Mean Runtime (μs) | 37208.10 | 18610.08 |
| Std Dev (μs) | 2029.80 | 1692.97 |
| Worst Runtime (μs) | 39827.20 | 20208.02 |

## 5. Kernel Source Code


### Kernel: r_135_40_8_16_3_7

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_135_40_8_16_3_7(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 40 */
  int gidx1 = gid.y; /* 135 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 16 */
  float4 val0 = *((device float4*)((data2+0)));
  float2 val1 = *((device float2*)((data2+4)));
  float val2 = *(data2+6);
  int alu0 = (gidx0*48);
  int alu1 = (gidx1*15360);
  int alu2 = (lidx0*1920);
  int alu3 = (lidx1*3);
  int alu4 = (alu0+alu3);
  int alu5 = (alu4+alu1+alu2);
  float val3 = *(data1+alu5);
  float val4 = *(data1+(alu5+1));
  float val5 = *(data1+(alu5+2));
  float val6 = ((alu4<1915)?*(data1+(alu5+5)):0.0f);
  float val7 = ((alu4<1916)?*(data1+(alu5+4)):0.0f);
  bool alu6 = (((gidx0+lidx1)<1)!=1);
  float val8 = (alu6?*(data1+(alu5+-3)):0.0f);
  float val9 = (alu6?*(data1+(alu5+-1)):0.0f);
  float val10 = (((alu4<2)!=1)?*(data1+(alu5+-2)):0.0f);
  float val11 = (((lidx1+(gidx0<<4))<639)?*(data1+(alu5+3)):0.0f);
  int alu7 = (alu0+alu1+alu2+alu3);
  *(data0+alu7) = ((val0.x*val8)+(val0.y*val10)+(val0.z*val9)+(val0.w*val3)+(val1.x*val4)+(val1.y*val5)+(val2*val11));
  *(data0+(alu7+1)) = ((val0.x*val10)+(val0.y*val9)+(val0.z*val3)+(val0.w*val4)+(val1.x*val5)+(val1.y*val11)+(val2*val7));
  *(data0+(alu7+2)) = ((val0.x*val9)+(val0.y*val3)+(val0.z*val4)+(val0.w*val5)+(val1.x*val11)+(val1.y*val7)+(val2*val6));
}
```


### Kernel: r_135_40_8_16_3_7n1

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_135_40_8_16_3_7n1(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 40 */
  int gidx1 = gid.y; /* 135 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 16 */
  float4 val0 = *((device float4*)((data2+0)));
  float2 val1 = *((device float2*)((data2+4)));
  float val2 = *(data2+6);
  int alu0 = (gidx0*48);
  int alu1 = (gidx1*15360);
  int alu2 = (lidx0*1920);
  int alu3 = (lidx1*3);
  int alu4 = (alu0+alu3+alu1+alu2);
  float val3 = *(data1+alu4);
  float val4 = *(data1+(alu4+1));
  float val5 = *(data1+(alu4+2));
  bool alu5 = (((gidx1+lidx0)<1)!=1);
  float val6 = (alu5?*(data1+(alu4+-1920)):0.0f);
  float val7 = (alu5?*(data1+(alu4+-1919)):0.0f);
  float val8 = (alu5?*(data1+(alu4+-1918)):0.0f);
  int alu6 = (lidx0+(gidx1<<3));
  bool alu7 = (alu6<1077);
  float val9 = (alu7?*(data1+(alu4+5760)):0.0f);
  float val10 = (alu7?*(data1+(alu4+5761)):0.0f);
  float val11 = (alu7?*(data1+(alu4+5762)):0.0f);
  bool alu8 = (alu6<1078);
  float val12 = (alu8?*(data1+(alu4+3840)):0.0f);
  float val13 = (alu8?*(data1+(alu4+3841)):0.0f);
  float val14 = (alu8?*(data1+(alu4+3842)):0.0f);
  bool alu9 = (alu6<1079);
  float val15 = (alu9?*(data1+(alu4+1920)):0.0f);
  float val16 = (alu9?*(data1+(alu4+1921)):0.0f);
  float val17 = (alu9?*(data1+(alu4+1922)):0.0f);
  bool alu10 = ((alu6<2)!=1);
  float val18 = (alu10?*(data1+(alu4+-3840)):0.0f);
  float val19 = (alu10?*(data1+(alu4+-3839)):0.0f);
  float val20 = (alu10?*(data1+(alu4+-3838)):0.0f);
  bool alu11 = ((alu6<3)!=1);
  float val21 = (alu11?*(data1+(alu4+-5760)):0.0f);
  float val22 = (alu11?*(data1+(alu4+-5759)):0.0f);
  float val23 = (alu11?*(data1+(alu4+-5758)):0.0f);
  int alu12 = (alu0+alu1+alu2+alu3);
  *(data0+alu12) = ((val0.x*val21)+(val0.y*val18)+(val0.z*val6)+(val0.w*val3)+(val1.x*val15)+(val1.y*val12)+(val2*val9));
  *(data0+(alu12+1)) = ((val0.x*val22)+(val0.y*val19)+(val0.z*val7)+(val0.w*val4)+(val1.x*val16)+(val1.y*val13)+(val2*val10));
  *(data0+(alu12+2)) = ((val0.x*val23)+(val0.y*val20)+(val0.z*val8)+(val0.w*val5)+(val1.x*val17)+(val1.y*val14)+(val2*val11));
}
```


### Kernel: r_2_135_40_8_16_3_3_3

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_2_135_40_8_16_3_3_3(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 40 */
  int gidx1 = gid.y; /* 135 */
  int gidx2 = gid.z; /* 2 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 16 */
  int alu0 = (gidx0*48);
  int alu1 = (gidx1*15360);
  int alu2 = (gidx2*9);
  float val0 = *(data2+alu2);
  float val1 = *(data2+(alu2+1));
  float val2 = *(data2+(alu2+2));
  float val3 = *(data2+(alu2+3));
  float val4 = *(data2+(alu2+4));
  float val5 = *(data2+(alu2+5));
  float val6 = *(data2+(alu2+6));
  float val7 = *(data2+(alu2+7));
  float val8 = *(data2+(alu2+8));
  int alu3 = (lidx0*1920);
  int alu4 = (lidx1*3);
  int alu5 = (alu0+alu4+alu1+alu3);
  float val9 = *(data1+alu5);
  float val10 = *(data1+(alu5+1));
  float val11 = *(data1+(alu5+2));
  bool alu6 = (((gidx0+lidx1)<1)!=1);
  float val12 = (alu6?*(data1+(alu5+-1)):0.0f);
  bool alu7 = (((gidx1+lidx0)<1)!=1);
  float val13 = (alu7?*(data1+(alu5+-1920)):0.0f);
  float val14 = (alu7?*(data1+(alu5+-1919)):0.0f);
  float val15 = (alu7?*(data1+(alu5+-1918)):0.0f);
  bool alu8 = ((lidx1+(gidx0<<4))<639);
  float val16 = (alu8?*(data1+(alu5+3)):0.0f);
  bool alu9 = ((lidx0+(gidx1<<3))<1079);
  float val17 = (alu9?*(data1+(alu5+1920)):0.0f);
  float val18 = (alu9?*(data1+(alu5+1921)):0.0f);
  float val19 = (alu9?*(data1+(alu5+1922)):0.0f);
  float val20 = ((alu9&alu8)?*(data1+(alu5+1923)):0.0f);
  float val21 = ((alu9&alu6)?*(data1+(alu5+1919)):0.0f);
  float val22 = ((alu8&alu7)?*(data1+(alu5+-1917)):0.0f);
  float val23 = ((alu6&alu7)?*(data1+(alu5+-1921)):0.0f);
  int alu10 = (alu1+(gidx2*2073600)+alu0+alu3+alu4);
  *(data0+alu10) = ((val3*val12)+(val0*val23)+(val6*val21)+(val1*val13)+(val4*val9)+(val7*val17)+(val2*val14)+(val5*val10)+(val8*val18));
  *(data0+(alu10+1)) = ((val3*val9)+(val0*val13)+(val6*val17)+(val1*val14)+(val4*val10)+(val7*val18)+(val2*val15)+(val5*val11)+(val8*val19));
  *(data0+(alu10+2)) = ((val3*val10)+(val0*val14)+(val6*val18)+(val1*val15)+(val4*val11)+(val7*val19)+(val2*val22)+(val5*val16)+(val8*val20));
}
```


### Kernel: E_135_40_8_16_3

```c
#include <metal_stdlib>
using namespace metal;
kernel void E_135_40_8_16_3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 40 */
  int gidx1 = gid.y; /* 135 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 16 */
  int alu0 = (gidx0*48);
  int alu1 = (gidx1*15360);
  int alu2 = (lidx0*1920);
  int alu3 = (lidx1*3);
  int alu4 = (alu0+alu1+alu2+alu3);
  float val0 = *(data1+alu4);
  int alu5 = (alu4+1);
  float val1 = *(data1+alu5);
  int alu6 = (alu4+2);
  float val2 = *(data1+alu6);
  float val3 = *(data1+(alu4+2073600));
  float val4 = *(data1+(alu4+2073601));
  float val5 = *(data1+(alu4+2073602));
  int alu7 = (alu0+alu3+alu1+alu2);
  float val6 = *(data1+alu7);
  float val7 = *(data1+(alu7+1));
  float val8 = *(data1+(alu7+2));
  float val9 = *(data1+(alu7+2073600));
  float val10 = *(data1+(alu7+2073601));
  float val11 = *(data1+(alu7+2073602));
  bool alu8 = (((gidx0+lidx1)<1)!=1);
  float val12 = (alu8?*(data1+(alu7+-1)):0.0f);
  float val13 = (alu8?*(data1+(alu7+2073599)):0.0f);
  bool alu9 = (((gidx1+lidx0)<1)!=1);
  float val14 = (alu9?*(data1+(alu4+-1920)):0.0f);
  float val15 = (alu9?*(data1+(alu4+-1919)):0.0f);
  float val16 = (alu9?*(data1+(alu4+-1918)):0.0f);
  float val17 = (alu9?*(data1+(alu4+2071680)):0.0f);
  float val18 = (alu9?*(data1+(alu4+2071681)):0.0f);
  float val19 = (alu9?*(data1+(alu4+2071682)):0.0f);
  float val20 = (alu9?*(data1+(alu7+-1920)):0.0f);
  float val21 = (alu9?*(data1+(alu7+-1919)):0.0f);
  float val22 = (alu9?*(data1+(alu7+-1918)):0.0f);
  float val23 = (alu9?*(data1+(alu7+2071680)):0.0f);
  float val24 = (alu9?*(data1+(alu7+2071681)):0.0f);
  float val25 = (alu9?*(data1+(alu7+2071682)):0.0f);
  bool alu10 = ((lidx1+(gidx0<<4))<639);
  float val26 = (alu10?*(data1+(alu7+3)):0.0f);
  float val27 = (alu10?*(data1+(alu7+2073603)):0.0f);
  bool alu11 = ((lidx0+(gidx1<<3))<1079);
  float val28 = (alu11?*(data1+(alu4+1920)):0.0f);
  float val29 = (alu11?*(data1+(alu4+1921)):0.0f);
  float val30 = (alu11?*(data1+(alu4+1922)):0.0f);
  float val31 = (alu11?*(data1+(alu4+2075520)):0.0f);
  float val32 = (alu11?*(data1+(alu4+2075521)):0.0f);
  float val33 = (alu11?*(data1+(alu4+2075522)):0.0f);
  float val34 = (alu11?*(data1+(alu7+1920)):0.0f);
  float val35 = (alu11?*(data1+(alu7+1921)):0.0f);
  float val36 = (alu11?*(data1+(alu7+1922)):0.0f);
  float val37 = (alu11?*(data1+(alu7+2075520)):0.0f);
  float val38 = (alu11?*(data1+(alu7+2075521)):0.0f);
  float val39 = (alu11?*(data1+(alu7+2075522)):0.0f);
  bool alu12 = (alu11&alu10);
  float val40 = (alu12?*(data1+(alu7+1923)):0.0f);
  float val41 = (alu12?*(data1+(alu7+2075523)):0.0f);
  bool alu13 = (alu11&alu8);
  float val42 = (alu13?*(data1+(alu7+1919)):0.0f);
  float val43 = (alu13?*(data1+(alu7+2075519)):0.0f);
  bool alu14 = (alu10&alu9);
  float val44 = (alu14?*(data1+(alu7+-1917)):0.0f);
  float val45 = (alu14?*(data1+(alu7+2071683)):0.0f);
  bool alu15 = (alu8&alu9);
  float val46 = (alu15?*(data1+(alu7+-1921)):0.0f);
  float val47 = (alu15?*(data1+(alu7+2071679)):0.0f);
  float alu16 = ((val1*val1)+(val4*val4));
  float alu17 = ((val2*val2)+(val5*val5));
  float alu18 = ((val3*val3)+(val0*val0));
  float alu19 = ((val21*val21)+(val24*val24));
  float alu20 = ((val7*val7)+(val10*val10));
  float alu21 = ((val35*val35)+(val38*val38));
  bool alu22 = (val1<0.0f);
  bool alu23 = (val2<0.0f);
  bool alu24 = (val3<0.0f);
  bool alu25 = (val4<0.0f);
  bool alu26 = (val5<0.0f);
  bool alu27 = (val0<0.0f);
  float cast0 = ((float)(((((float)((alu22!=1)))!=((float)((alu25!=1))))!=1)));
  float cast1 = ((float)(((((float)((alu23!=1)))!=((float)((alu26!=1))))!=1)));
  float cast2 = ((float)(((((float)((alu24!=1)))!=((float)((alu27!=1))))!=1)));
  float cast3 = ((float)((((val4*(((bool)(val4))?(alu25?-1.0f:1.0f):0.0f))<(val1*(((bool)(val1))?(alu22?-1.0f:1.0f):0.0f)))!=1)));
  float alu28 = (cast3+(cast3*cast0)+((1.0f-cast3)*(1.0f-cast0)));
  float alu29 = (((((float)((((bool)(alu28))!=1)))*((float)((((alu16<((val8*val8)+(val11*val11)))!=1)&((alu16<((val9*val9)+(val6*val6)))!=1)))))+(((float)(((alu28!=1.0f)!=1)))*((float)((((alu16<((val22*val22)+(val25*val25)))!=1)&((alu16<((val34*val34)+(val37*val37)))!=1)))))+(((float)(((alu28!=2.0f)!=1)))*((float)((((alu16<((val15*val15)+(val18*val18)))!=1)&((alu16<((val29*val29)+(val32*val32)))!=1)))))+(((float)(((alu28!=3.0f)!=1)))*((float)((((alu16<((val20*val20)+(val23*val23)))!=1)&((alu16<((val36*val36)+(val39*val39)))!=1))))))*alu16);
  float cast4 = ((float)((((val5*(((bool)(val5))?(alu26?-1.0f:1.0f):0.0f))<(val2*(((bool)(val2))?(alu23?-1.0f:1.0f):0.0f)))!=1)));
  float alu30 = (cast4+(cast4*cast1)+((1.0f-cast4)*(1.0f-cast1)));
  float alu31 = (((((float)((((bool)(alu30))!=1)))*((float)((((alu17<alu20)!=1)&((alu17<((val26*val26)+(val27*val27)))!=1)))))+(((float)(((alu30!=1.0f)!=1)))*((float)((((alu17<((val44*val44)+(val45*val45)))!=1)&((alu17<alu21)!=1)))))+(((float)(((alu30!=2.0f)!=1)))*((float)((((alu17<((val16*val16)+(val19*val19)))!=1)&((alu17<((val30*val30)+(val33*val33)))!=1)))))+(((float)(((alu30!=3.0f)!=1)))*((float)((((alu17<alu19)!=1)&((alu17<((val40*val40)+(val41*val41)))!=1))))))*alu17);
  float cast5 = ((float)((((val3*(((bool)(val3))?(alu24?-1.0f:1.0f):0.0f))<(val0*(((bool)(val0))?(alu27?-1.0f:1.0f):0.0f)))!=1)));
  float alu32 = (cast5+(cast5*cast2)+((1.0f-cast5)*(1.0f-cast2)));
  float alu33 = (((((float)((((bool)(alu32))!=1)))*((float)((((alu18<((val12*val12)+(val13*val13)))!=1)&((alu18<alu20)!=1)))))+(((float)(((alu32!=1.0f)!=1)))*((float)((((alu18<alu19)!=1)&((alu18<((val42*val42)+(val43*val43)))!=1)))))+(((float)(((alu32!=2.0f)!=1)))*((float)((((alu18<((val14*val14)+(val17*val17)))!=1)&((alu18<((val28*val28)+(val31*val31)))!=1)))))+(((float)(((alu32!=3.0f)!=1)))*((float)((((alu18<((val46*val46)+(val47*val47)))!=1)&((alu18<alu21)!=1))))))*alu18);
  float alu34 = ((alu33<0.0f)?0.0f:((((bool)(alu33))!=1)?(alu33*0.5f):alu33));
  float alu35 = -alu34;
  float alu36 = ((alu29<0.0f)?0.0f:((((bool)(alu29))!=1)?(alu29*0.5f):alu29));
  float alu37 = -alu36;
  float alu38 = ((alu31<0.0f)?0.0f:((((bool)(alu31))!=1)?(alu31*0.5f):alu31));
  float alu39 = -alu38;
  *(data0+alu4) = ((float)(((-((alu35<-1.0f)?-1.0f:(((alu35!=-1.0f)!=1)?((alu34*-0.5f)+-0.5f):alu35))<0.09f)!=1)));
  *(data0+alu5) = ((float)(((-((alu37<-1.0f)?-1.0f:(((alu37!=-1.0f)!=1)?((alu36*-0.5f)+-0.5f):alu37))<0.09f)!=1)));
  *(data0+alu6) = ((float)(((-((alu39<-1.0f)?-1.0f:(((alu39!=-1.0f)!=1)?((alu38*-0.5f)+-0.5f):alu39))<0.09f)!=1)));
}
```


### Kernel: r_135_40_8_16_3_3_3

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_135_40_8_16_3_3_3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 40 */
  int gidx1 = gid.y; /* 135 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 16 */
  int alu0 = ((gidx0*48)+(gidx1*15360)+(lidx0*1920)+(lidx1*3));
  float val0 = *(data1+alu0);
  int alu1 = (alu0+1);
  float val1 = *(data1+alu1);
  int alu2 = (alu0+2);
  float val2 = *(data1+alu2);
  float alu3 = ((val1<0.0f)?0.0f:((((bool)(val1))!=1)?(val1*0.5f):val1));
  float alu4 = -alu3;
  float alu5 = ((val2<0.0f)?0.0f:((((bool)(val2))!=1)?(val2*0.5f):val2));
  float alu6 = -alu5;
  float alu7 = ((val0<0.0f)?0.0f:((((bool)(val0))!=1)?(val0*0.5f):val0));
  float alu8 = -alu7;
  *(data0+alu1) = -((alu4<-1.0f)?-1.0f:(((alu4!=-1.0f)!=1)?((alu3*-0.5f)+-0.5f):alu4));
  *(data0+alu2) = -((alu6<-1.0f)?-1.0f:(((alu6!=-1.0f)!=1)?((alu5*-0.5f)+-0.5f):alu6));
  *(data0+alu0) = -((alu8<-1.0f)?-1.0f:(((alu8!=-1.0f)!=1)?((alu7*-0.5f)+-0.5f):alu8));
}
```


### Optimized Kernels

### Kernel: r_540_30_8_2_4_7_2

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_540_30_8_2_4_7_2(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 30 */
  int gidx1 = gid.y; /* 540 */
  int lidx0 = lid.x; /* 8 */
  int lidx1 = lid.y; /* 2 */
  float4 val0 = *((device float4*)((data2+0)));
  float2 val1 = *((device float2*)((data2+4)));
  float val2 = *(data2+6);
  int alu0 = (gidx1*3840);
  int alu1 = (lidx1*1920);
  int alu2 = (lidx0+(gidx0<<5));
  int alu3 = (gidx0<<6);
  int alu4 = (lidx0<<1);
  int alu5 = (alu3+alu4);
  int alu6 = (alu5+alu0+alu1);
  float2 val3 = *((device float2*)((data1+alu6)));
  float2 val4 = ((((gidx0+lidx0)<1)!=1)?*((device float2*)((data1+(alu6+-2)))):float2(0.0f,0.0f));
  float2 val5 = *((device float2*)((data1+(alu6+2))));
  float val6 = *(data1+(alu6+4));
  float val7 = *(data1+(alu6+13));
  float2 val8 = *((device float2*)((data1+(alu6+14))));
  float2 val9 = *((device float2*)((data1+(alu6+16))));
  float2 val10 = *((device float2*)((data1+(alu6+18))));
  float val11 = *(data1+(alu6+20));
  float val12 = *(data1+(alu6+29));
  float2 val13 = *((device float2*)((data1+(alu6+30))));
  float2 val14 = *((device float2*)((data1+(alu6+32))));
  float2 val15 = *((device float2*)((data1+(alu6+34))));
  float val16 = *(data1+(alu6+36));
  float val17 = *(data1+(alu6+45));
  float2 val18 = *((device float2*)((data1+(alu6+46))));
  float2 val19 = *((device float2*)((data1+(alu6+48))));
  float val20 = ((alu2<935)?*(data1+(alu6+50)):0.0f);
  float val21 = ((alu2<934)?*(data1+(alu6+52)):0.0f);
  float val22 = ((alu5<1869)?*(data1+(alu6+51)):0.0f);
  float val23 = (((alu5<3)!=1)?*(data1+(alu6+-3)):0.0f);
  int alu7 = (alu3+alu0+alu4+alu1);
  *((device float2*)((data0+alu7))) = float2(((val4.x*val0.y)+(val0.x*val23)+(val4.y*val0.z)+(val3.x*val0.w)+(val1.x*val3.y)+(val5.x*val1.y)+(val5.y*val2)),((val4.x*val0.x)+(val4.y*val0.y)+(val3.x*val0.z)+(val3.y*val0.w)+(val1.x*val5.x)+(val1.y*val5.y)+(val6*val2)));
  *((device float2*)((data0+(alu7+16)))) = float2(((val8.x*val0.y)+(val0.x*val7)+(val8.y*val0.z)+(val9.x*val0.w)+(val1.x*val9.y)+(val10.x*val1.y)+(val10.y*val2)),((val8.x*val0.x)+(val8.y*val0.y)+(val9.x*val0.z)+(val9.y*val0.w)+(val1.x*val10.x)+(val1.y*val10.y)+(val11*val2)));
  *((device float2*)((data0+(alu7+32)))) = float2(((val13.x*val0.y)+(val0.x*val12)+(val13.y*val0.z)+(val14.x*val0.w)+(val1.x*val14.y)+(val15.x*val1.y)+(val15.y*val2)),((val13.x*val0.x)+(val13.y*val0.y)+(val14.x*val0.z)+(val14.y*val0.w)+(val1.x*val15.x)+(val1.y*val15.y)+(val16*val2)));
  *((device float2*)((data0+(alu7+48)))) = float2(((val18.x*val0.y)+(val0.x*val17)+(val18.y*val0.z)+(val19.x*val0.w)+(val1.x*val19.y)+(val1.y*val20)+(val22*val2)),((val18.x*val0.x)+(val18.y*val0.y)+(val19.x*val0.z)+(val19.y*val0.w)+(val1.x*val20)+(val1.y*val22)+(val21*val2)));
}
```


### Kernel: r_360_64_16_2_7_3

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_360_64_16_2_7_3(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 64 */
  int gidx1 = gid.y; /* 360 */
  int lidx0 = lid.x; /* 16 */
  int lidx1 = lid.y; /* 2 */
  bool alu0 = (gidx0<60);
  float4 val0 = (alu0?*((device float4*)((data2+0))):float4(0.0f,0.0f,0.0f,0.0f));
  float2 val1 = (alu0?*((device float2*)((data2+4))):float2(0.0f,0.0f));
  float val2 = (alu0?*(data2+6):0.0f);
  int alu1 = (gidx1*5760);
  int alu2 = (gidx0<<5);
  int alu3 = (lidx1<<4);
  int alu4 = (lidx0+alu2+alu3+alu1);
  float val3 = (alu0?*(data1+alu4):0.0f);
  float val4 = (alu0?*(data1+(alu4+1920)):0.0f);
  float val5 = (alu0?*(data1+(alu4+3840)):0.0f);
  bool alu5 = (alu0&(gidx1<359));
  float val6 = (alu5?*(data1+(alu4+5760)):0.0f);
  float val7 = (alu5?*(data1+(alu4+7680)):0.0f);
  float val8 = (alu5?*(data1+(alu4+9600)):0.0f);
  bool alu6 = (alu0&((gidx1<1)!=1));
  float val9 = (alu6?*(data1+(alu4+-5760)):0.0f);
  float val10 = (alu6?*(data1+(alu4+-3840)):0.0f);
  float val11 = (alu6?*(data1+(alu4+-1920)):0.0f);
  int alu7 = (lidx0+alu2+alu1+alu3);
  if (alu0) {
    *(data0+alu7) = ((val0.x*val9)+(val0.y*val10)+(val0.z*val11)+(val0.w*val3)+(val1.x*val4)+(val1.y*val5)+(val2*val6));
    *(data0+(alu7+1920)) = ((val0.x*val10)+(val0.y*val11)+(val0.z*val3)+(val0.w*val4)+(val1.x*val5)+(val1.y*val6)+(val2*val7));
    *(data0+(alu7+3840)) = ((val0.x*val11)+(val0.y*val3)+(val0.z*val4)+(val0.w*val5)+(val1.x*val6)+(val1.y*val7)+(val2*val8));
  }
}
```


### Kernel: r_1080_24_16_3_3_2_5

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_1080_24_16_3_3_2_5(device float* data0, device float* data1, device float* data2, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 24 */
  int gidx1 = gid.y; /* 1080 */
  int lidx0 = lid.x; /* 16 */
  int alu0 = (gidx0*80);
  int alu1 = (gidx1*1920);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  float acc4 = 0.0f;
  float acc5 = 0.0f;
  float acc6 = 0.0f;
  float acc7 = 0.0f;
  float acc8 = 0.0f;
  float acc9 = 0.0f;
  for (int ridx0 = 0; ridx0 < 3; ridx0++) {
    int alu2 = (gidx1+ridx0);
    bool alu3 = ((alu2<1081)&((alu2<1)!=1));
    for (int ridx1 = 0; ridx1 < 3; ridx1++) {
      int alu4 = (lidx0+alu0+ridx1);
      int alu5 = (alu4+alu1+(ridx0*1920));
      float val0 = (alu3?*(data1+(alu5+-1905)):0.0f);
      float val1 = (alu3?*(data1+(alu5+-1889)):0.0f);
      float val2 = (alu3?*(data1+(alu5+-1873)):0.0f);
      int alu6 = ((ridx0*3)+ridx1);
      float val3 = *(data2+alu6);
      float val4 = *(data2+(alu6+9));
      float val5 = (((alu4<1857)&alu3)?*(data1+(alu5+-1857)):0.0f);
      float val6 = (((((gidx0+lidx0+ridx1)<1)!=1)&alu3)?*(data1+(alu5+-1921)):0.0f);
      acc0 = (acc0+(val3*val6));
      acc1 = (acc1+(val3*val0));
      acc2 = (acc2+(val3*val1));
      acc3 = (acc3+(val3*val2));
      acc4 = (acc4+(val3*val5));
      acc5 = (acc5+(val4*val6));
      acc6 = (acc6+(val4*val0));
      acc7 = (acc7+(val4*val1));
      acc8 = (acc8+(val4*val2));
      acc9 = (acc9+(val4*val5));
    }
  }
  int alu19 = (lidx0+alu0+alu1);
  *(data0+alu19) = acc0;
  *(data0+(alu19+16)) = acc1;
  *(data0+(alu19+32)) = acc2;
  *(data0+(alu19+48)) = acc3;
  *(data0+(alu19+64)) = acc4;
  *(data0+(alu19+2073600)) = acc5;
  *(data0+(alu19+2073616)) = acc6;
  *(data0+(alu19+2073632)) = acc7;
  *(data0+(alu19+2073648)) = acc8;
  *(data0+(alu19+2073664)) = acc9;
}
```


### Kernel: E_360_30_4_4_4_3

```c
#include <metal_stdlib>
using namespace metal;
kernel void E_360_30_4_4_4_3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 30 */
  int gidx1 = gid.y; /* 360 */
  int lidx0 = lid.x; /* 4 */
  int lidx1 = lid.y; /* 4 */
  float2 cast0 = float2(0.0f,0.0f);
  float4 cast1 = float4(0.0f,0.0f,0.0f,0.0f);
  int alu0 = (gidx1*5760);
  bool alu1 = (gidx1<359);
  bool alu2 = ((gidx1<1)!=1);
  bool alu3 = (((lidx1+gidx0+lidx0)<1)!=1);
  int alu4 = (gidx0<<6);
  int alu5 = (lidx0<<2);
  bool alu6 = ((lidx0+(gidx0<<4)+(lidx1<<2))<479);
  int alu7 = (lidx1<<4);
  int alu8 = (alu4+alu0+alu5+alu7);
  float4 val0 = *((device float4*)((data1+alu8)));
  float4 val1 = (alu2?*((device float4*)((data1+(alu8+-1920)))):cast1);
  int alu9 = (alu8+1920);
  float4 val2 = *((device float4*)((data1+alu9)));
  int alu10 = (alu8+3840);
  float4 val3 = *((device float4*)((data1+alu10)));
  float4 val4 = (alu1?*((device float4*)((data1+(alu8+5760)))):cast1);
  float4 val5 = (alu2?*((device float4*)((data1+(alu8+2071680)))):cast1);
  float4 val6 = *((device float4*)((data1+(alu8+2073600))));
  float4 val7 = *((device float4*)((data1+(alu8+2075520))));
  float4 val8 = *((device float4*)((data1+(alu8+2077440))));
  float4 val9 = (alu1?*((device float4*)((data1+(alu8+2079360)))):cast1);
  int alu11 = (alu4+alu5+alu7+alu0);
  float2 val10 = *((device float2*)((data1+alu11)));
  float2 val11 = (alu2?*((device float2*)((data1+(alu11+-1920)))):cast0);
  float val12 = (alu2?*(data1+(alu11+-1919)):0.0f);
  float val13 = (alu2?*(data1+(alu11+-1918)):0.0f);
  float2 val14 = (alu2?*((device float2*)((data1+(alu11+-1918)))):cast0);
  float val15 = (alu3?*(data1+(alu11+-1)):0.0f);
  float val16 = *(data1+(alu11+1));
  float val17 = *(data1+(alu11+2));
  float2 val18 = *((device float2*)((data1+(alu11+2))));
  float val19 = (alu6?*(data1+(alu11+4)):0.0f);
  float val20 = (alu3?*(data1+(alu11+1919)):0.0f);
  float2 val21 = *((device float2*)((data1+(alu11+1920))));
  float val22 = *(data1+(alu11+1921));
  float val23 = *(data1+(alu11+1922));
  float2 val24 = *((device float2*)((data1+(alu11+1922))));
  float val25 = (alu6?*(data1+(alu11+1924)):0.0f);
  float val26 = (alu3?*(data1+(alu11+3839)):0.0f);
  float2 val27 = *((device float2*)((data1+(alu11+3840))));
  float val28 = *(data1+(alu11+3841));
  float val29 = *(data1+(alu11+3842));
  float2 val30 = *((device float2*)((data1+(alu11+3842))));
  float val31 = (alu6?*(data1+(alu11+3844)):0.0f);
  float2 val32 = (alu1?*((device float2*)((data1+(alu11+5760)))):cast0);
  float val33 = (alu1?*(data1+(alu11+5761)):0.0f);
  float val34 = (alu1?*(data1+(alu11+5762)):0.0f);
  float2 val35 = (alu1?*((device float2*)((data1+(alu11+5762)))):cast0);
  float2 val36 = (alu2?*((device float2*)((data1+(alu11+2071680)))):cast0);
  float val37 = (alu2?*(data1+(alu11+2071681)):0.0f);
  float val38 = (alu2?*(data1+(alu11+2071682)):0.0f);
  float2 val39 = (alu2?*((device float2*)((data1+(alu11+2071682)))):cast0);
  float val40 = (alu3?*(data1+(alu11+2073599)):0.0f);
  float2 val41 = *((device float2*)((data1+(alu11+2073600))));
  float val42 = *(data1+(alu11+2073601));
  float val43 = *(data1+(alu11+2073602));
  float2 val44 = *((device float2*)((data1+(alu11+2073602))));
  float val45 = (alu6?*(data1+(alu11+2073604)):0.0f);
  float val46 = (alu3?*(data1+(alu11+2075519)):0.0f);
  float2 val47 = *((device float2*)((data1+(alu11+2075520))));
  float val48 = *(data1+(alu11+2075521));
  float val49 = *(data1+(alu11+2075522));
  float2 val50 = *((device float2*)((data1+(alu11+2075522))));
  float val51 = (alu6?*(data1+(alu11+2075524)):0.0f);
  float val52 = (alu3?*(data1+(alu11+2077439)):0.0f);
  float2 val53 = *((device float2*)((data1+(alu11+2077440))));
  float val54 = *(data1+(alu11+2077441));
  float val55 = *(data1+(alu11+2077442));
  float2 val56 = *((device float2*)((data1+(alu11+2077442))));
  float val57 = (alu6?*(data1+(alu11+2077444)):0.0f);
  float2 val58 = (alu1?*((device float2*)((data1+(alu11+2079360)))):cast0);
  float val59 = (alu1?*(data1+(alu11+2079361)):0.0f);
  float val60 = (alu1?*(data1+(alu11+2079362)):0.0f);
  float2 val61 = (alu1?*((device float2*)((data1+(alu11+2079362)))):cast0);
  bool alu12 = (alu1&alu6);
  float val62 = (alu12?*(data1+(alu11+5764)):0.0f);
  float val63 = (alu12?*(data1+(alu11+2079364)):0.0f);
  bool alu13 = (alu1&alu3);
  float val64 = (alu13?*(data1+(alu11+5759)):0.0f);
  float val65 = (alu13?*(data1+(alu11+2079359)):0.0f);
  bool alu14 = (alu6&alu2);
  float val66 = (alu14?*(data1+(alu11+-1916)):0.0f);
  float val67 = (alu14?*(data1+(alu11+2071684)):0.0f);
  bool alu15 = (alu2&alu3);
  float val68 = (alu15?*(data1+(alu11+-1921)):0.0f);
  float val69 = (alu15?*(data1+(alu11+2071679)):0.0f);
  float alu16 = ((val18.x*val18.x)+(val44.x*val44.x));
  float alu17 = ((val21.x*val21.x)+(val47.x*val47.x));
  float alu18 = ((val24.x*val24.x)+(val50.x*val50.x));
  float alu19 = ((val27.x*val27.x)+(val53.x*val53.x));
  float alu20 = ((val30.x*val30.x)+(val56.x*val56.x));
  float alu21 = ((val41.x*val41.x)+(val10.x*val10.x));
  float alu22 = ((val2.x*val2.x)+(val7.x*val7.x));
  float alu23 = ((val3.x*val3.x)+(val8.x*val8.x));
  float alu24 = ((val6.x*val6.x)+(val0.x*val0.x));
  float alu25 = ((val18.y*val18.y)+(val44.y*val44.y));
  float alu26 = ((val21.y*val21.y)+(val47.y*val47.y));
  float alu27 = ((val24.y*val24.y)+(val50.y*val50.y));
  float alu28 = ((val27.y*val27.y)+(val53.y*val53.y));
  float alu29 = ((val30.y*val30.y)+(val56.y*val56.y));
  float alu30 = ((val41.y*val41.y)+(val10.y*val10.y));
  float alu31 = ((val2.y*val2.y)+(val7.y*val7.y));
  float alu32 = ((val3.y*val3.y)+(val8.y*val8.y));
  float alu33 = ((val6.y*val6.y)+(val0.y*val0.y));
  float alu34 = ((val2.z*val2.z)+(val7.z*val7.z));
  float alu35 = ((val3.z*val3.z)+(val8.z*val8.z));
  float alu36 = ((val6.z*val6.z)+(val0.z*val0.z));
  float alu37 = ((val2.w*val2.w)+(val7.w*val7.w));
  float alu38 = ((val3.w*val3.w)+(val8.w*val8.w));
  float alu39 = ((val6.w*val6.w)+(val0.w*val0.w));
  float alu40 = ((val15*val15)+(val40*val40));
  float alu41 = ((val16*val16)+(val42*val42));
  float alu42 = ((val17*val17)+(val43*val43));
  float alu43 = ((val19*val19)+(val45*val45));
  float alu44 = ((val20*val20)+(val46*val46));
  float alu45 = ((val22*val22)+(val48*val48));
  float alu46 = ((val23*val23)+(val49*val49));
  float alu47 = ((val25*val25)+(val51*val51));
  float alu48 = ((val26*val26)+(val52*val52));
  float alu49 = ((val28*val28)+(val54*val54));
  float alu50 = ((val29*val29)+(val55*val55));
  float alu51 = ((val31*val31)+(val57*val57));
  bool alu52 = (val2.x<0.0f);
  bool alu53 = (val3.x<0.0f);
  bool alu54 = (val6.x<0.0f);
  bool alu55 = (val7.x<0.0f);
  bool alu56 = (val8.x<0.0f);
  bool alu57 = (val0.x<0.0f);
  bool alu58 = (val2.y<0.0f);
  bool alu59 = (val3.y<0.0f);
  bool alu60 = (val6.y<0.0f);
  bool alu61 = (val7.y<0.0f);
  bool alu62 = (val8.y<0.0f);
  bool alu63 = (val0.y<0.0f);
  bool alu64 = (val2.z<0.0f);
  bool alu65 = (val3.z<0.0f);
  bool alu66 = (val6.z<0.0f);
  bool alu67 = (val7.z<0.0f);
  bool alu68 = (val8.z<0.0f);
  bool alu69 = (val0.z<0.0f);
  bool alu70 = (val2.w<0.0f);
  bool alu71 = (val3.w<0.0f);
  bool alu72 = (val6.w<0.0f);
  bool alu73 = (val7.w<0.0f);
  bool alu74 = (val8.w<0.0f);
  bool alu75 = (val0.w<0.0f);
  float cast2 = ((float)(((((float)((alu52!=1)))!=((float)((alu55!=1))))!=1)));
  float cast3 = ((float)(((((float)((alu53!=1)))!=((float)((alu56!=1))))!=1)));
  float cast4 = ((float)(((((float)((alu54!=1)))!=((float)((alu57!=1))))!=1)));
  float cast5 = ((float)(((((float)((alu58!=1)))!=((float)((alu61!=1))))!=1)));
  float cast6 = ((float)(((((float)((alu59!=1)))!=((float)((alu62!=1))))!=1)));
  float cast7 = ((float)(((((float)((alu60!=1)))!=((float)((alu63!=1))))!=1)));
  float cast8 = ((float)(((((float)((alu64!=1)))!=((float)((alu67!=1))))!=1)));
  float cast9 = ((float)(((((float)((alu65!=1)))!=((float)((alu68!=1))))!=1)));
  float cast10 = ((float)(((((float)((alu66!=1)))!=((float)((alu69!=1))))!=1)));
  float cast11 = ((float)(((((float)((alu70!=1)))!=((float)((alu73!=1))))!=1)));
  float cast12 = ((float)(((((float)((alu71!=1)))!=((float)((alu74!=1))))!=1)));
  float cast13 = ((float)(((((float)((alu72!=1)))!=((float)((alu75!=1))))!=1)));
  float cast14 = ((float)((((val7.x*(((bool)(val7.x))?(alu55?-1.0f:1.0f):0.0f))<(val2.x*(((bool)(val2.x))?(alu52?-1.0f:1.0f):0.0f)))!=1)));
  float alu76 = (cast14+(cast14*cast2)+((1.0f-cast14)*(1.0f-cast2)));
  float alu77 = (((((float)((((bool)(alu76))!=1)))*((float)((((alu22<alu44)!=1)&((alu22<alu45)!=1)))))+(((float)(((alu76!=1.0f)!=1)))*((float)((((alu22<alu41)!=1)&((alu22<alu48)!=1)))))+(((float)(((alu76!=2.0f)!=1)))*((float)((((alu22<alu23)!=1)&((alu22<alu24)!=1)))))+(((float)(((alu76!=3.0f)!=1)))*((float)((((alu22<alu40)!=1)&((alu22<alu49)!=1))))))*alu22);
  float cast15 = ((float)((((val8.x*(((bool)(val8.x))?(alu56?-1.0f:1.0f):0.0f))<(val3.x*(((bool)(val3.x))?(alu53?-1.0f:1.0f):0.0f)))!=1)));
  float alu78 = (cast15+(cast15*cast3)+((1.0f-cast15)*(1.0f-cast3)));
  float alu79 = (((((float)((((bool)(alu78))!=1)))*((float)((((alu23<alu48)!=1)&((alu23<alu49)!=1)))))+(((float)(((alu78!=1.0f)!=1)))*((float)((((alu23<alu45)!=1)&((alu23<((val64*val64)+(val65*val65)))!=1)))))+(((float)(((alu78!=2.0f)!=1)))*((float)((((alu23<alu22)!=1)&((alu23<((val4.x*val4.x)+(val9.x*val9.x)))!=1)))))+(((float)(((alu78!=3.0f)!=1)))*((float)((((alu23<alu44)!=1)&((alu23<((val33*val33)+(val59*val59)))!=1))))))*alu23);
  float cast16 = ((float)((((val6.x*(((bool)(val6.x))?(alu54?-1.0f:1.0f):0.0f))<(val0.x*(((bool)(val0.x))?(alu57?-1.0f:1.0f):0.0f)))!=1)));
  float alu80 = (cast16+(cast16*cast4)+((1.0f-cast16)*(1.0f-cast4)));
  float alu81 = (((((float)((((bool)(alu80))!=1)))*((float)((((alu24<alu40)!=1)&((alu24<alu41)!=1)))))+(((float)(((alu80!=1.0f)!=1)))*((float)((((alu24<((val12*val12)+(val37*val37)))!=1)&((alu24<alu44)!=1)))))+(((float)(((alu80!=2.0f)!=1)))*((float)((((alu24<((val1.x*val1.x)+(val5.x*val5.x)))!=1)&((alu24<alu22)!=1)))))+(((float)(((alu80!=3.0f)!=1)))*((float)((((alu24<((val68*val68)+(val69*val69)))!=1)&((alu24<alu45)!=1))))))*alu24);
  float cast17 = ((float)((((val7.y*(((bool)(val7.y))?(alu61?-1.0f:1.0f):0.0f))<(val2.y*(((bool)(val2.y))?(alu58?-1.0f:1.0f):0.0f)))!=1)));
  float alu82 = (cast17+(cast17*cast5)+((1.0f-cast17)*(1.0f-cast5)));
  float alu83 = (((((float)((((bool)(alu82))!=1)))*((float)((((alu31<alu17)!=1)&((alu31<alu18)!=1)))))+(((float)(((alu82!=1.0f)!=1)))*((float)((((alu31<alu16)!=1)&((alu31<alu19)!=1)))))+(((float)(((alu82!=2.0f)!=1)))*((float)((((alu31<alu32)!=1)&((alu31<alu33)!=1)))))+(((float)(((alu82!=3.0f)!=1)))*((float)((((alu31<alu20)!=1)&((alu31<alu21)!=1))))))*alu31);
  float cast18 = ((float)((((val8.y*(((bool)(val8.y))?(alu62?-1.0f:1.0f):0.0f))<(val3.y*(((bool)(val3.y))?(alu59?-1.0f:1.0f):0.0f)))!=1)));
  float alu84 = (cast18+(cast18*cast6)+((1.0f-cast18)*(1.0f-cast6)));
  float alu85 = (((((float)((((bool)(alu84))!=1)))*((float)((((alu32<alu19)!=1)&((alu32<alu20)!=1)))))+(((float)(((alu84!=1.0f)!=1)))*((float)((((alu32<alu18)!=1)&((alu32<((val32.x*val32.x)+(val58.x*val58.x)))!=1)))))+(((float)(((alu84!=2.0f)!=1)))*((float)((((alu32<alu31)!=1)&((alu32<((val4.y*val4.y)+(val9.y*val9.y)))!=1)))))+(((float)(((alu84!=3.0f)!=1)))*((float)((((alu32<alu17)!=1)&((alu32<((val35.x*val35.x)+(val61.x*val61.x)))!=1))))))*alu32);
  float cast19 = ((float)((((val6.y*(((bool)(val6.y))?(alu60?-1.0f:1.0f):0.0f))<(val0.y*(((bool)(val0.y))?(alu63?-1.0f:1.0f):0.0f)))!=1)));
  float alu86 = (cast19+(cast19*cast7)+((1.0f-cast19)*(1.0f-cast7)));
  float alu87 = (((((float)((((bool)(alu86))!=1)))*((float)((((alu33<alu16)!=1)&((alu33<alu21)!=1)))))+(((float)(((alu86!=1.0f)!=1)))*((float)((((alu33<((val14.x*val14.x)+(val39.x*val39.x)))!=1)&((alu33<alu17)!=1)))))+(((float)(((alu86!=2.0f)!=1)))*((float)((((alu33<((val1.y*val1.y)+(val5.y*val5.y)))!=1)&((alu33<alu31)!=1)))))+(((float)(((alu86!=3.0f)!=1)))*((float)((((alu33<((val11.x*val11.x)+(val36.x*val36.x)))!=1)&((alu33<alu18)!=1))))))*alu33);
  float cast20 = ((float)((((val7.z*(((bool)(val7.z))?(alu67?-1.0f:1.0f):0.0f))<(val2.z*(((bool)(val2.z))?(alu64?-1.0f:1.0f):0.0f)))!=1)));
  float alu88 = (cast20+(cast20*cast8)+((1.0f-cast20)*(1.0f-cast8)));
  float alu89 = (((((float)((((bool)(alu88))!=1)))*((float)((((alu34<alu26)!=1)&((alu34<alu27)!=1)))))+(((float)(((alu88!=1.0f)!=1)))*((float)((((alu34<alu25)!=1)&((alu34<alu28)!=1)))))+(((float)(((alu88!=2.0f)!=1)))*((float)((((alu34<alu35)!=1)&((alu34<alu36)!=1)))))+(((float)(((alu88!=3.0f)!=1)))*((float)((((alu34<alu29)!=1)&((alu34<alu30)!=1))))))*alu34);
  float cast21 = ((float)((((val8.z*(((bool)(val8.z))?(alu68?-1.0f:1.0f):0.0f))<(val3.z*(((bool)(val3.z))?(alu65?-1.0f:1.0f):0.0f)))!=1)));
  float alu90 = (cast21+(cast21*cast9)+((1.0f-cast21)*(1.0f-cast9)));
  float alu91 = (((((float)((((bool)(alu90))!=1)))*((float)((((alu35<alu28)!=1)&((alu35<alu29)!=1)))))+(((float)(((alu90!=1.0f)!=1)))*((float)((((alu35<alu27)!=1)&((alu35<((val32.y*val32.y)+(val58.y*val58.y)))!=1)))))+(((float)(((alu90!=2.0f)!=1)))*((float)((((alu35<alu34)!=1)&((alu35<((val4.z*val4.z)+(val9.z*val9.z)))!=1)))))+(((float)(((alu90!=3.0f)!=1)))*((float)((((alu35<alu26)!=1)&((alu35<((val35.y*val35.y)+(val61.y*val61.y)))!=1))))))*alu35);
  float cast22 = ((float)((((val6.z*(((bool)(val6.z))?(alu66?-1.0f:1.0f):0.0f))<(val0.z*(((bool)(val0.z))?(alu69?-1.0f:1.0f):0.0f)))!=1)));
  float alu92 = (cast22+(cast22*cast10)+((1.0f-cast22)*(1.0f-cast10)));
  float alu93 = (((((float)((((bool)(alu92))!=1)))*((float)((((alu36<alu25)!=1)&((alu36<alu30)!=1)))))+(((float)(((alu92!=1.0f)!=1)))*((float)((((alu36<((val14.y*val14.y)+(val39.y*val39.y)))!=1)&((alu36<alu26)!=1)))))+(((float)(((alu92!=2.0f)!=1)))*((float)((((alu36<((val1.z*val1.z)+(val5.z*val5.z)))!=1)&((alu36<alu34)!=1)))))+(((float)(((alu92!=3.0f)!=1)))*((float)((((alu36<((val11.y*val11.y)+(val36.y*val36.y)))!=1)&((alu36<alu27)!=1))))))*alu36);
  float cast23 = ((float)((((val7.w*(((bool)(val7.w))?(alu73?-1.0f:1.0f):0.0f))<(val2.w*(((bool)(val2.w))?(alu70?-1.0f:1.0f):0.0f)))!=1)));
  float alu94 = (cast23+(cast23*cast11)+((1.0f-cast23)*(1.0f-cast11)));
  float alu95 = (((((float)((((bool)(alu94))!=1)))*((float)((((alu37<alu46)!=1)&((alu37<alu47)!=1)))))+(((float)(((alu94!=1.0f)!=1)))*((float)((((alu37<alu43)!=1)&((alu37<alu50)!=1)))))+(((float)(((alu94!=2.0f)!=1)))*((float)((((alu37<alu38)!=1)&((alu37<alu39)!=1)))))+(((float)(((alu94!=3.0f)!=1)))*((float)((((alu37<alu42)!=1)&((alu37<alu51)!=1))))))*alu37);
  float cast24 = ((float)((((val8.w*(((bool)(val8.w))?(alu74?-1.0f:1.0f):0.0f))<(val3.w*(((bool)(val3.w))?(alu71?-1.0f:1.0f):0.0f)))!=1)));
  float alu96 = (cast24+(cast24*cast12)+((1.0f-cast24)*(1.0f-cast12)));
  float alu97 = (((((float)((((bool)(alu96))!=1)))*((float)((((alu38<alu50)!=1)&((alu38<alu51)!=1)))))+(((float)(((alu96!=1.0f)!=1)))*((float)((((alu38<alu47)!=1)&((alu38<((val34*val34)+(val60*val60)))!=1)))))+(((float)(((alu96!=2.0f)!=1)))*((float)((((alu38<alu37)!=1)&((alu38<((val4.w*val4.w)+(val9.w*val9.w)))!=1)))))+(((float)(((alu96!=3.0f)!=1)))*((float)((((alu38<alu46)!=1)&((alu38<((val62*val62)+(val63*val63)))!=1))))))*alu38);
  float cast25 = ((float)((((val6.w*(((bool)(val6.w))?(alu72?-1.0f:1.0f):0.0f))<(val0.w*(((bool)(val0.w))?(alu75?-1.0f:1.0f):0.0f)))!=1)));
  float alu98 = (cast25+(cast25*cast13)+((1.0f-cast25)*(1.0f-cast13)));
  float alu99 = (((((float)((((bool)(alu98))!=1)))*((float)((((alu39<alu42)!=1)&((alu39<alu43)!=1)))))+(((float)(((alu98!=1.0f)!=1)))*((float)((((alu39<((val66*val66)+(val67*val67)))!=1)&((alu39<alu46)!=1)))))+(((float)(((alu98!=2.0f)!=1)))*((float)((((alu39<((val1.w*val1.w)+(val5.w*val5.w)))!=1)&((alu39<alu37)!=1)))))+(((float)(((alu98!=3.0f)!=1)))*((float)((((alu39<((val13*val13)+(val38*val38)))!=1)&((alu39<alu47)!=1))))))*alu39);
  float alu100 = ((alu81<0.0f)?0.0f:((((bool)(alu81))!=1)?(alu81*0.5f):alu81));
  float alu101 = -alu100;
  float alu102 = ((alu77<0.0f)?0.0f:((((bool)(alu77))!=1)?(alu77*0.5f):alu77));
  float alu103 = -alu102;
  float alu104 = ((alu79<0.0f)?0.0f:((((bool)(alu79))!=1)?(alu79*0.5f):alu79));
  float alu105 = -alu104;
  float alu106 = ((alu87<0.0f)?0.0f:((((bool)(alu87))!=1)?(alu87*0.5f):alu87));
  float alu107 = -alu106;
  float alu108 = ((alu83<0.0f)?0.0f:((((bool)(alu83))!=1)?(alu83*0.5f):alu83));
  float alu109 = -alu108;
  float alu110 = ((alu85<0.0f)?0.0f:((((bool)(alu85))!=1)?(alu85*0.5f):alu85));
  float alu111 = -alu110;
  float alu112 = ((alu93<0.0f)?0.0f:((((bool)(alu93))!=1)?(alu93*0.5f):alu93));
  float alu113 = -alu112;
  float alu114 = ((alu89<0.0f)?0.0f:((((bool)(alu89))!=1)?(alu89*0.5f):alu89));
  float alu115 = -alu114;
  float alu116 = ((alu91<0.0f)?0.0f:((((bool)(alu91))!=1)?(alu91*0.5f):alu91));
  float alu117 = -alu116;
  float alu118 = ((alu99<0.0f)?0.0f:((((bool)(alu99))!=1)?(alu99*0.5f):alu99));
  float alu119 = -alu118;
  float alu120 = ((alu95<0.0f)?0.0f:((((bool)(alu95))!=1)?(alu95*0.5f):alu95));
  float alu121 = -alu120;
  float alu122 = ((alu97<0.0f)?0.0f:((((bool)(alu97))!=1)?(alu97*0.5f):alu97));
  float alu123 = -alu122;
  *((device float4*)((data0+alu8))) = float4(((float)(((-((alu101<-1.0f)?-1.0f:(((alu101!=-1.0f)!=1)?((alu100*-0.5f)+-0.5f):alu101))<0.09f)!=1))),((float)(((-((alu107<-1.0f)?-1.0f:(((alu107!=-1.0f)!=1)?((alu106*-0.5f)+-0.5f):alu107))<0.09f)!=1))),((float)(((-((alu113<-1.0f)?-1.0f:(((alu113!=-1.0f)!=1)?((alu112*-0.5f)+-0.5f):alu113))<0.09f)!=1))),((float)(((-((alu119<-1.0f)?-1.0f:(((alu119!=-1.0f)!=1)?((alu118*-0.5f)+-0.5f):alu119))<0.09f)!=1))));
  *((device float4*)((data0+alu9))) = float4(((float)(((-((alu103<-1.0f)?-1.0f:(((alu103!=-1.0f)!=1)?((alu102*-0.5f)+-0.5f):alu103))<0.09f)!=1))),((float)(((-((alu109<-1.0f)?-1.0f:(((alu109!=-1.0f)!=1)?((alu108*-0.5f)+-0.5f):alu109))<0.09f)!=1))),((float)(((-((alu115<-1.0f)?-1.0f:(((alu115!=-1.0f)!=1)?((alu114*-0.5f)+-0.5f):alu115))<0.09f)!=1))),((float)(((-((alu121<-1.0f)?-1.0f:(((alu121!=-1.0f)!=1)?((alu120*-0.5f)+-0.5f):alu121))<0.09f)!=1))));
  *((device float4*)((data0+alu10))) = float4(((float)(((-((alu105<-1.0f)?-1.0f:(((alu105!=-1.0f)!=1)?((alu104*-0.5f)+-0.5f):alu105))<0.09f)!=1))),((float)(((-((alu111<-1.0f)?-1.0f:(((alu111!=-1.0f)!=1)?((alu110*-0.5f)+-0.5f):alu111))<0.09f)!=1))),((float)(((-((alu117<-1.0f)?-1.0f:(((alu117!=-1.0f)!=1)?((alu116*-0.5f)+-0.5f):alu117))<0.09f)!=1))),((float)(((-((alu123<-1.0f)?-1.0f:(((alu123!=-1.0f)!=1)?((alu122*-0.5f)+-0.5f):alu123))<0.09f)!=1))));
}
```


### Kernel: r_270_15_16_2_8_2_3_3

```c
#include <metal_stdlib>
using namespace metal;
kernel void r_270_15_16_2_8_2_3_3(device float* data0, device float* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
  int gidx0 = gid.x; /* 15 */
  int gidx1 = gid.y; /* 270 */
  int lidx0 = lid.x; /* 32 */
  int lidx1 = lid.y; /* 8 */
  int lidx2 = lid.z; /* 2 */
  int alu0 = ((gidx0<<7)+(gidx1*7680)+(lidx0&15)+((lidx0>>4)*1920)+(lidx1<<4)+(lidx2*3840));
  float val0 = *(data1+alu0);
  float alu1 = ((val0<0.0f)?0.0f:((((bool)(val0))!=1)?(val0*0.5f):val0));
  float alu2 = -alu1;
  *(data0+alu0) = -((alu2<-1.0f)?-1.0f:(((alu2!=-1.0f)!=1)?((alu1*-0.5f)+-0.5f):alu2));
}
```
