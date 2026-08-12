#include "nf_conditioned_cloud_row_chain_f32_v1.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const size_t full=73u,core=31u,width=47u,origin=61u,halo=9u;
    const size_t scale_n=full*width*3u, output_n=core*width*3u;
    size_t count_n=0u,conv_n=0u,scratch_n=0u,i;
    nf_density_conditioned_poisson_u16_profile_v3 cp={sizeof(cp),3u,{192.,288.,240.},32.,{32.,16.,24.},78277u,1009u};
    nf_cloud_spatial_response_f32_profile_v2 sp={sizeof(sp),2u,{1.3,1.7,2.1},{.00125,.001125,.001375},4.};
    const float gain[3]={.3f,.35f,.25f};
    double* scale; float* expected; uint16_t* counts; double* conv;
    float *sd,*st,*density,*trans; int status,bad; FILE *a,*b;
    if(argc!=3||nf_conditioned_cloud_row_chain_f32_workspace_v1(core,width,halo,&count_n,&conv_n,&scratch_n)!=0)return 2;
    scale=malloc(scale_n*sizeof(double)); expected=malloc(output_n*sizeof(float)); counts=malloc(count_n*sizeof(uint16_t)); conv=malloc(conv_n*sizeof(double)); sd=malloc(scratch_n*sizeof(float)); st=malloc(scratch_n*sizeof(float)); density=malloc(output_n*sizeof(float)); trans=malloc(output_n*sizeof(float));
    if(!scale||!expected||!counts||!conv||!sd||!st||!density||!trans)return 3;
    for(i=0u;i<scale_n;i++)scale[i]=(double)((i*37u+11u)%1001u)/1000.;
    for(i=0u;i<output_n;i++)expected[i]=.2f+(float)((i*19u+7u)%601u)/1000.f;
    status=nf_conditioned_cloud_row_chain_f32_apply_v1(&cp,&sp,full,width,origin,core,halo,scale,scale_n,expected,output_n,gain,counts,count_n,conv,conv_n,sd,st,scratch_n,density,trans,output_n);
    for(i=0u;i<output_n;i++){density[i]=-77.f;trans[i]=-77.f;}
    scale[origin*width*3u]=0./0.;
    bad=nf_conditioned_cloud_row_chain_f32_apply_v1(&cp,&sp,full,width,origin,core,halo,scale,scale_n,expected,output_n,gain,counts,count_n,conv,conv_n,sd,st,scratch_n,density,trans,output_n);
    for(i=0u;i<output_n;i++)if(density[i]!=-77.f||trans[i]!=-77.f)return 4;
    scale[origin*width*3u]=0.;
    status=nf_conditioned_cloud_row_chain_f32_apply_v1(&cp,&sp,full,width,origin,core,halo,scale,scale_n,expected,output_n,gain,counts,count_n,conv,conv_n,sd,st,scratch_n,density,trans,output_n);
    a=fopen(argv[1],"wb");b=fopen(argv[2],"wb");if(!a||!b||fwrite(density,sizeof(float),output_n,a)!=output_n||fwrite(trans,sizeof(float),output_n,b)!=output_n)return 5;fclose(a);fclose(b);
    printf("status=%d invalid=%d counts=%zu convolution=%zu core=%zu\n",status,bad,count_n,conv_n,scratch_n);
    free(scale);free(expected);free(counts);free(conv);free(sd);free(st);free(density);free(trans);
    return status==0&&bad!=0?0:6;
}
