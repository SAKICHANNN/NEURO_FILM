#include "nf_cloud_interpretation_chain_f32_v1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void profile_init(nf_physical_domains_f32_profile_v1* p) {
    const double x[3]={-4.0717306,-1.,.7446556};
    const double y[3]={.05,.65,1.25};
    const double d[3]={.1953296,.25,.3439074};
    size_t c,i; memset(p,0,sizeof(*p)); p->struct_size=sizeof(*p);p->abi_version=1u;
    memcpy(p->source_component_sha256,"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",65u);
    p->reference_linear=.18;p->black_offset=1./65536.;p->exposure_floor=1./65536.;p->matrix_minimum_determinant=.01;
    for(c=0;c<3u;++c){p->knot_count[c]=3u;for(i=0;i<3u;++i){p->x_knots[c][i]=x[i];p->y_knots[c][i]=y[i];p->derivatives[c][i]=d[i];}p->dye_absorption_matrix[c][c]=1.;p->print_matrix[c][c]=1.;p->paper_midpoints[c]=-1.;p->paper_slopes[c]=1.;p->paper_maximum_densities[c]=2.;p->black_reference_density[c]=.05;p->white_reference_density[c]=1.25;}
}

int main(int argc,char**argv){
    const size_t full=73u,core=31u,width=47u,halo=9u,first=(61u+73u-9u)%73u;
    size_t counts_n=0u,conv_n=0u,core_n=0u,i,y,x,c;int status,bad;FILE*f;
    nf_physical_domains_f32_profile_v1 dp;nf_density_conditioned_poisson_u16_profile_v3 cp={sizeof(cp),3u,{192.,288.,240.},32.,{32.,16.,24.},78277u,1009u};nf_cloud_spatial_response_f32_profile_v2 sp={sizeof(sp),2u,{1.3,1.7,2.1},{.00125,.001125,.001375},4.};
    const double capacity[3]={1.5,1.5,1.5};const float gain[3]={.3f,.35f,.25f};
    float *scene,*expected,*sd,*st,*cloud_d,*cloud_t,*scan,*manual_d,*manual_t,*manual_scan;double*scale;uint16_t*counts;double*conv;
    if(argc!=2||nf_conditioned_cloud_row_chain_f32_workspace_v1(core,width,halo,&counts_n,&conv_n,&core_n)!=0)return 2;
    scene=malloc(counts_n*sizeof(float));expected=malloc(core_n*sizeof(float));sd=malloc(core_n*sizeof(float));st=malloc(core_n*sizeof(float));cloud_d=malloc(core_n*sizeof(float));cloud_t=malloc(core_n*sizeof(float));scan=malloc(core_n*sizeof(float));manual_d=malloc(core_n*sizeof(float));manual_t=malloc(core_n*sizeof(float));manual_scan=malloc(core_n*sizeof(float));scale=malloc(counts_n*sizeof(double));counts=malloc(counts_n*sizeof(uint16_t));conv=malloc(conv_n*sizeof(double));
    if(!scene||!expected||!sd||!st||!cloud_d||!cloud_t||!scan||!manual_d||!manual_t||!manual_scan||!scale||!counts||!conv)return 3;profile_init(&dp);
    for(y=0;y<core+2u*halo;++y){size_t gy=(first+y)%full;for(x=0;x<width;++x)for(c=0;c<3u;++c)scene[(y*width+x)*3u+c]=(float)((gy*101u+x*37u+c*211u+11u)%1001u)/1000.f;}
    for(i=0;i<core_n;++i)expected[i]=.6f;
    status=nf_cloud_interpretation_chain_f32_apply_window_v1(&dp,&cp,&sp,full,width,first,core,halo,scene,counts_n,capacity,scale,counts_n,expected,core_n,gain,counts,counts_n,conv,conv_n,sd,st,core_n,cloud_d,cloud_t,core_n,scan,core_n);
    if(status!=0)return 4;
    status=nf_sensitometry_cloud_bridge_f32_apply_window_v1(&dp,&cp,&sp,full,width,first,core,halo,scene,counts_n,capacity,scale,counts_n,expected,core_n,gain,counts,counts_n,conv,conv_n,sd,st,core_n,manual_d,manual_t,core_n);
    if(status!=0||nf_physical_interpretation_f32_apply_v1(&dp,manual_d,core_n/3u,manual_scan)!=0||memcmp(scan,manual_scan,core_n*sizeof(float))!=0)return 5;
    for(i=0;i<core_n;++i)scan[i]=-77.f;scene[counts_n-1u]=-1.f;
    bad=nf_cloud_interpretation_chain_f32_apply_window_v1(&dp,&cp,&sp,full,width,first,core,halo,scene,counts_n,capacity,scale,counts_n,expected,core_n,gain,counts,counts_n,conv,conv_n,sd,st,core_n,cloud_d,cloud_t,core_n,scan,core_n);
    for(i=0;i<core_n;++i)if(scan[i]!=-77.f)return 6;
#if defined(_WIN32)
    if(fopen_s(&f,argv[1],"wb")!=0)f=NULL;
#else
    f=fopen(argv[1],"wb");
#endif
    if(!f||fwrite(manual_scan,sizeof(float),core_n,f)!=core_n)return 7;fclose(f);
    printf("status=0 invalid=%d manual_exact=1 core=%zu\n",bad,core_n);
    free(scene);free(expected);free(sd);free(st);free(cloud_d);free(cloud_t);free(scan);free(manual_d);free(manual_t);free(manual_scan);free(scale);free(counts);free(conv);return bad!=0?0:8;
}
