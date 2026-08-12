#define NF_CLOUD_SPATIAL_RESPONSE_F32_BUILD
#define NF_DENSITY_CONDITIONED_POISSON_U16_BUILD
#include "nf_cloud_spatial_response_f32_v2.h"
#include "nf_density_conditioned_poisson_u16_v2.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const size_t core=31u,width=47u,halo=9u,extended=core+2u*halo;
    const size_t values=extended*width*3u, out_values=core*width*3u;
    nf_density_conditioned_poisson_u16_profile_v2 cp={sizeof(cp),2u,{12.,18.,15.},2.,{2.,1.,1.5},76801u,1009u};
    nf_cloud_spatial_response_f32_profile_v2 sp={sizeof(sp),2u,{1.3,1.7,2.1},{.02,.018,.022},4.};
    float* scale=malloc(values*sizeof(float)); uint16_t* counts=malloc(values*sizeof(uint16_t));
    double* workspace=malloc(extended*width*sizeof(double)); float* density=malloc(out_values*sizeof(float)); float* trans=malloc(out_values*sizeof(float));
    int cs,ss,bad; size_t i;
    if(argc!=4||!scale||!counts||!workspace||!density||!trans)return 2;
    for(i=0;i<values;i++)scale[i]=(float)((i*37u+11u)%1001u)/1000.0f;
    cs=nf_density_conditioned_poisson_u16_sample_region_v2(&cp,extended,width,0,0,extended,width,scale,values,counts,values);
    ss=nf_cloud_spatial_response_f32_apply_v2(&sp,counts,core,width,halo,workspace,extended*width,density,trans,out_values);
    scale[values-1]=2.0f; bad=nf_density_conditioned_poisson_u16_sample_region_v2(&cp,extended,width,0,0,extended,width,scale,values,counts,values);
    FILE* a=fopen(argv[1],"wb"); FILE* b=fopen(argv[2],"wb"); FILE* c=fopen(argv[3],"wb");
    if(!a||!b||!c||fwrite(counts,sizeof(uint16_t),values,a)!=values||fwrite(density,sizeof(float),out_values,b)!=out_values||fwrite(trans,sizeof(float),out_values,c)!=out_values)return 4;
    fclose(a);fclose(b);fclose(c); printf("count=%d spatial=%d invalid=%d\n",cs,ss,bad);
    free(scale);free(counts);free(workspace);free(density);free(trans);
    return cs==0&&ss==0&&bad!=0?0:3;
}
