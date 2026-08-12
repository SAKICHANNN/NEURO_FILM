#include "nf_cloud_spatial_interpretation_chain_f32_v1.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char component_sha[65] =
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

static void domains_init(nf_physical_domains_f32_profile_v1* p) {
    const double x[3] = {-4.0717306, -1.0, .7446556};
    const double y[3] = {.05, .65, 1.25};
    const double d[3] = {.1953296, .25, .3439074};
    size_t c, i;
    memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p); p->abi_version = 1u;
    memcpy(p->source_component_sha256, component_sha, 65u);
    p->reference_linear = .18; p->black_offset = 1.0 / 65536.0;
    p->exposure_floor = 1.0 / 65536.0; p->matrix_minimum_determinant = .01;
    for (c = 0u; c < 3u; ++c) {
        p->knot_count[c] = 3u;
        for (i = 0u; i < 3u; ++i) {
            p->x_knots[c][i] = x[i]; p->y_knots[c][i] = y[i];
            p->derivatives[c][i] = d[i];
        }
        p->dye_absorption_matrix[c][c] = 1.0;
        p->print_matrix[c][c] = 1.0;
        p->paper_midpoints[c] = -1.0; p->paper_slopes[c] = 1.0;
        p->paper_maximum_densities[c] = 2.0;
        p->black_reference_density[c] = .05;
        p->white_reference_density[c] = 1.25;
    }
}

static void gaussian_init(
    nf_gaussian_f32_profile_v1* p, double r, double g, double b) {
    memset(p, 0, sizeof(*p)); p->struct_size = sizeof(*p); p->abi_version = 1u;
    memcpy(p->source_component_sha256, component_sha, 65u);
    p->sigma_pixels_rgb[0] = r; p->sigma_pixels_rgb[1] = g;
    p->sigma_pixels_rgb[2] = b; p->truncate = 3.0;
}

int main(int argc, char** argv) {
    const size_t full=73u, core=31u, width=47u, halo=9u;
    const size_t first=(61u+full-halo)%full;
    const double capacity[3]={1.5,1.5,1.5};
    const float gains[3]={.3f,.35f,.25f};
    size_t counts_n=0u, conv_n=0u, core_n=0u, i, yy, xx, cc;
    int status, bad, stage_nonzero=0; FILE* file;
    nf_physical_domains_f32_profile_v1 domains;
    nf_density_conditioned_poisson_u16_profile_v3 count={sizeof(count),3u,{192.,288.,240.},32.,{32.,16.,24.},78277u,1009u};
    nf_cloud_spatial_response_f32_profile_v2 cloud={sizeof(cloud),2u,{1.3,1.7,2.1},{.00125,.001125,.001375},4.};
    nf_gaussian_f32_profile_v1 adjacency_blur, diffusion, scanner;
    nf_bounded_adjacency_f32_profile_v1 adjacency;
    float *scene,*expected,*cloud_d,*cloud_t,*spatial_d,*spatial_t,*gw,*blurred,*adjacent,*diffused,*interpreted,*scan,*manual;
    double *scale,*conv; uint16_t* counts;
    if (argc != 2 || nf_conditioned_cloud_row_chain_f32_workspace_v1(
            core,width,halo,&counts_n,&conv_n,&core_n) != 0) return 2;
#define ALLOC_F(name,n) name=(float*)malloc((n)*sizeof(float)); if(name==NULL)return 3
    ALLOC_F(scene,counts_n); ALLOC_F(expected,core_n); ALLOC_F(cloud_d,core_n);
    ALLOC_F(cloud_t,core_n); ALLOC_F(spatial_d,core_n); ALLOC_F(spatial_t,core_n);
    ALLOC_F(gw,core_n); ALLOC_F(blurred,core_n); ALLOC_F(adjacent,core_n);
    ALLOC_F(diffused,core_n); ALLOC_F(interpreted,core_n); ALLOC_F(scan,core_n);
    ALLOC_F(manual,core_n);
    scale=(double*)malloc(counts_n*sizeof(double)); conv=(double*)malloc(conv_n*sizeof(double));
    counts=(uint16_t*)malloc(counts_n*sizeof(uint16_t));
    if(scale==NULL||conv==NULL||counts==NULL)return 3;
    domains_init(&domains); gaussian_init(&adjacency_blur,.8,1.,1.2);
    gaussian_init(&diffusion,.6,.8,1.); gaussian_init(&scanner,.7,.7,.7);
    memset(&adjacency,0,sizeof(adjacency)); adjacency.struct_size=sizeof(adjacency);adjacency.abi_version=1u;
    memcpy(adjacency.source_component_sha256,component_sha,65u);
    adjacency.maximum_absolute_transmittance_delta=.02;adjacency.maximum_absolute_density_delta=.04;
    for(cc=0u;cc<3u;++cc){adjacency.gain_rgb[cc]=.08+.02*(double)cc;adjacency.black_reference_density[cc]=.05;adjacency.white_reference_density[cc]=1.25;}
    for(yy=0u;yy<core+2u*halo;++yy){size_t gy=(first+yy)%full;for(xx=0u;xx<width;++xx)for(cc=0u;cc<3u;++cc)scene[(yy*width+xx)*3u+cc]=(float)((gy*101u+xx*37u+cc*211u+11u)%1001u)/1000.f;}
    for(i=0u;i<core_n;++i)expected[i]=.6f;
#define ARGS &domains,&count,&cloud,&adjacency_blur,&adjacency,&diffusion,&scanner,full,width,first,core,halo,scene,counts_n,capacity,scale,counts_n,expected,core_n,gains,counts,counts_n,conv,conv_n,cloud_d,cloud_t,spatial_d,spatial_t,core_n,gw,blurred,adjacent,diffused,interpreted,core_n
    status=nf_cloud_spatial_interpretation_chain_f32_apply_window_v1(ARGS,scan,core_n);
    if(status!=0)return 4;
    if(nf_sensitometry_cloud_bridge_f32_apply_window_v1(&domains,&count,&cloud,full,width,first,core,halo,scene,counts_n,capacity,scale,counts_n,expected,core_n,gains,counts,counts_n,conv,conv_n,spatial_d,spatial_t,core_n,cloud_d,cloud_t,core_n)!=0 ||
       nf_gaussian_f32_apply_v1(&adjacency_blur,cloud_d,core,width,gw,core_n,blurred)!=0 ||
       nf_bounded_adjacency_f32_apply_v1(&adjacency,cloud_d,blurred,core_n/3u,adjacent)!=0 ||
       nf_gaussian_f32_apply_v1(&diffusion,adjacent,core,width,gw,core_n,diffused)!=0 ||
       nf_physical_interpretation_f32_apply_v1(&domains,diffused,core_n/3u,interpreted)!=0 ||
       nf_gaussian_f32_apply_v1(&scanner,interpreted,core,width,gw,core_n,manual)!=0 ||
       memcmp(scan,manual,core_n*sizeof(float))!=0)return 5;
    for(i=0u;i<core_n;++i){
        if(cloud_d[i]!=adjacent[i] && adjacent[i]!=diffused[i] && interpreted[i]!=manual[i]){
            stage_nonzero=1;break;
        }
    }
    if(!stage_nonzero)return 9;
    for(i=0u;i<core_n;++i)scan[i]=-77.f; scene[counts_n-1u]=-1.f;
    bad=nf_cloud_spatial_interpretation_chain_f32_apply_window_v1(ARGS,scan,core_n);
    for(i=0u;i<core_n;++i)if(scan[i]!=-77.f)return 6;
#if defined(_WIN32)
    if(fopen_s(&file,argv[1],"wb")!=0)file=NULL;
#else
    file=fopen(argv[1],"wb");
#endif
    if(file==NULL||fwrite(manual,sizeof(float),core_n,file)!=core_n)return 7;
    fclose(file);
    printf("status=0 invalid=%d manual_exact=1 stages_nonzero=1 core=%zu\n",bad,core_n);
    return bad!=0?0:8;
}
