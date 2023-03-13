
/*

    + + + + + +
|   + + + + + +   |
|   + + + + + +   |
|   + + + + + +   |
|   + + + + + +   |  /
\                /   |
 \              /    |
  \            /     |
   \  r_small /      |
    \ <----> /       | First row height
    /        \       |
   /          \      |
  /            \     |
 /              \    |
/                \   |
 <--------------->   /
      r_big
*/



#include <iostream>
#include <cuda_runtime.h>
#include "cutil_math.h"
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <fstream>
#include <vector>
#include "device_launch_parameters.h"
#include "grains.cuh"
#include "bonds.cuh"
#include "ConeWall.cuh"
#include "CylinderWall.cuh"
#include "Cone_Dimensions.cuh"
#include "DEM_Parameters.cuh"
#define creatNewSample
//#define debugSample
#define importOldSample
#define M_PI 3.14159265359

//particles informations:
const float rmin = 0.00185;
const float rmax = 0.00215;
const float rhos = 2600.;
float first_row_height = 0.45;
const int row_number = 40;
const int ngw = 4; // number of grains in each row
const int nbg_row = 30; //

#ifdef creatNewSample
const int nbgrains = row_number * nbg_row;
#endif
#ifdef debugSample
const int nbgrains = 3;
#endif



//global variables:
std::vector <grains> g_vec; //particle vector:
cone cowall[2]; // cones array
cylinder cylwall; // cylinder array
float dt, dt_crt;
float x_min, x_max, y_min, y_max, z_min,z_max;



inline float  norm (std::vector<float>& a){
    float v = 0;
    for (int i = 0; i<3;i++){
        v += a[i] * a[i];
    }
    v = sqrtf(v);
    return v;
}
__constant__ DemParams params;

__device__ inline float  norm (float *a){
    float v = 0;
    for (int i = 0; i<3;i++){
        v += a[i] * a[i];
    }
    v = sqrtf(v);
    return v;
}


__device__ inline void  normalize (float *a){
    float len = norm(a);
    for (int i = 0; i<3;i++){
        if(len != 0){
            a[i]=a[i]/len;
        }
    }


}

__device__ inline void addArray(float *a, float *b, float *out){


    out[0]=a[0] + b[0];
    out[1]=a[1] + b[1];
    out[2]=a[2] + b[2];

}
__device__ inline void subArray(float *a, float *b, float *out){


    out[0]=a[0] - b[0];
    out[1]=a[1] - b[1];
    out[2]=a[2] - b[2];

}

__device__ inline void scalarMulti(float a, float *b, float *out){
    out[0]= a * b[0];
    out[1]= a * b[1];
    out[2]= a * b[2];
}

__device__ inline float dot(float *a, float *b){
    float out = 0.;

    out = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

    return out;
}

__device__ inline void cross(float *a, float *b, float *out){


    out[0] =   ( (a[1] * b[2]) - (a[2] * b[1]) );
    out[1] = -1 * ( (a[0] * b[2]) - (a[2] * b[0]) );
    out[2] =   ( (a[0] * b[1]) - (a[1] * b[0]) );

}

__global__ void preForce_cuda(grains *g, float dt, int size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    g[tid].vx = 0.5 * g[tid].ax * dt + g[tid].vx;
    g[tid].vy = 0.5 * g[tid].ay * dt + g[tid].vy;
    g[tid].vz = 0.5 * g[tid].az * dt + g[tid].vz;



    g[tid].vthx = 0.5 * g[tid].athx * dt + g[tid].vthx;
    g[tid].vthy = 0.5 * g[tid].athy * dt + g[tid].vthy;
    g[tid].vthz = 0.5 * g[tid].athz * dt + g[tid].vthz;



    //position(t + dt)
    g[tid].x = g[tid].x + g[tid].vx * dt;
    g[tid].y = g[tid].y + g[tid].vy * dt;
    g[tid].z = g[tid].z + g[tid].vz * dt;
    //printf("particle %d position %f\n",tid, g[tid].x);

    g[tid].thx = g[tid].thx + g[tid].vthx * dt;
    g[tid].thy = g[tid].thy + g[tid].vthy * dt;
    g[tid].thz = g[tid].thz + g[tid].vthz * dt;

    //reseting acceleration to zero
    g[tid].ax = 0.;
    g[tid].ay = 0.;
    g[tid].az = 0.;

    g[tid].athx = 0.;
    g[tid].athy = 0.;
    g[tid].athz = 0.;

}

__global__ void postForce_cuda(grains *g, float dt, int size){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    // applying gravity
    g[tid].ax = g[tid].ax + params.gx * g[tid].mass;
    g[tid].ay = g[tid].ay + params.gy * g[tid].mass;
    g[tid].az = g[tid].az + params.gz * g[tid].mass;


    // compute acceleration
    g[tid].ax = g[tid].ax/g[tid].mass;
    g[tid].ay = g[tid].ay/g[tid].mass;
    g[tid].az = g[tid].az/g[tid].mass;



    g[tid].athx = g[tid].athx/g[tid].inertia;
    g[tid].athy = g[tid].athy/g[tid].inertia;
    g[tid].athz = g[tid].athz/g[tid].inertia;

    //velocity (t + dt)
    g[tid].vx = 0.5 * g[tid].ax * dt + g[tid].vx;
    g[tid].vy = 0.5 * g[tid].ay * dt + g[tid].vy;
    g[tid].vz = 0.5 * g[tid].az * dt + g[tid].vz;

    g[tid].vthx = 0.5 * g[tid].athx * dt + g[tid].vthx;
    g[tid].vthy = 0.5 * g[tid].athy * dt + g[tid].vthy;
    g[tid].vthz = 0.5 * g[tid].athz * dt + g[tid].vthz;


}


__global__ void sortingParticles_cuda (grains *g,int size, int *head, int *list,
                                       float mesh_resol,
                                       int mesh_x, int mesh_y, int mesh_z, float x_min,float y_min,float z_min, float x_max,float y_max,float z_max){


    for(int i=0; i<size;i++){
        int ix = floor((g[i].x - x_min)/mesh_resol);
        int iy = floor((g[i].y - y_min)/mesh_resol);
        int iz = floor((g[i].z - z_min)/mesh_resol);
        if(g[i].x > x_max || g[i].y > y_max || g[i].z > z_max || g[i].x < x_min || g[i].y < y_min || g[i].z < z_min )
        {
            printf("Particles went out of bounds!");
            continue;
        }

        int ncell = ix + (iy*mesh_x) + (mesh_x*mesh_y*iz);

        list[i] = head[ncell];
        head[ncell] = i;
    }


}



__device__ void dryParticles_force(grains *g, int i, int j, float dt)
{
    float xij, yij, zij, cij, dn;
    xij = g[i].x - g[j].x;
    yij = g[i].y - g[j].y;
    zij = g[i].z - g[j].z;
    cij = sqrtf(xij*xij + yij*yij + zij*zij);
    dn = cij - (g[i].radius + g[j].radius);


    if (dn < 0){
        //relative velocity


        float kn = 2.f * params.E * (g[i].radius * g[j].radius)/(g[i].radius + g[j].radius);
        float meff = (g[i].mass * g[j].mass)/(g[i].mass + g[j].mass);
        float damp = -1.f *log(params.en)/sqrt(log(params.en)*log(params.en) + M_PI*M_PI);
        float nun = 2.f * sqrt(kn * meff) * damp  ;
        float ks = params.ks_to_kn * kn;
        float kr = params.kr_to_kn * kn;
        float ko = params.ko_to_kn * kn;
        float nus = params.nus_to_nun  * nun;
        float nur = params.nur_to_nun * nun;
        float nuo = params.nuo_to_nun * nun;

        //normal vector
        float nij[3]= {0};
        nij[0] = xij;
        nij[1] = yij;
        nij[2] = zij;
        if (norm(nij) != 0){
            normalize(nij);
        }

        float relVelij[3]= {0};
        relVelij[0] = g[i].vx - g[j].vx;
        relVelij[1] = g[i].vy - g[j].vy;
        relVelij[2] = g[i].vz - g[j].vz;

        float relVelN[3]= {0};
//        float relVelT[3]= {0};
        float dotN = dot(relVelij,nij);
        relVelN[0] = nij[0] * dotN; relVelN[1] = nij[1] * dotN; relVelN[2] = nij[2] * dotN;
//        relVelT[0] = relVelij[0] - relVelN[0];relVelT[1] = relVelij[1] - relVelN[1];relVelT[2] = relVelij[2] - relVelN[2];

        float fn[3]= {0};
        fn[0] = nij[0] * kn * abs(dn) - relVelN[0]*nun;
        fn[1] = nij[1] * kn * abs(dn) - relVelN[1]*nun;
        fn[2] = nij[2] * kn * abs(dn) - relVelN[2]*nun;



        float vthi[3] = {0};
        float vthj[3] = {0};
        float crossi[3] = {0};
        float crossj[3] = {0};
        float vij[3] = {0};
        float vijt[3] = {0};
        float vijr[3] = {0};
        float vijo[3] = {0};
        float ut[3] = {0};
        float ur[3] = {0};
        float uo[3] = {0};
        float doti, dotj;

        vthi[0] = g[i].vthx;
        vthi[1] = g[i].vthy;
        vthi[2] = g[i].vthz;

        vthj[0] = g[j].vthx;
        vthj[1] = g[j].vthy;
        vthj[2] = g[j].vthz;

        cross(nij, vthi,crossi);
        cross(nij, vthj,crossj);

        doti = dot(nij, vthi);
        dotj = dot(nij, vthj);

//        vij[0] = relVelij[0] + (g[i].radius - 0.5 * abs(dn)) * crossi[0] + (g[j].radius - 0.5 * abs(dn)) * crossj[0] ;
//        vij[1] = relVelij[1] + (g[i].radius - 0.5 * abs(dn)) * crossi[1] + (g[j].radius - 0.5 * abs(dn)) * crossj[1] ;
//        vij[2] = relVelij[2] + (g[i].radius - 0.5 * abs(dn)) * crossi[2] + (g[j].radius - 0.5 * abs(dn)) * crossj[2] ;

        vij[0] = relVelij[0]  ;
        vij[1] = relVelij[1] ;
        vij[2] = relVelij[2]  ;

        vijt[0] = vij[0] - nij[0] * dot(nij, vij);
        vijt[1] = vij[1] - nij[1] * dot(nij, vij);
        vijt[2] = vij[2] - nij[2] * dot(nij, vij);

//        float a_ij = (g[i].radius * g[j].radius)/ (g[i].radius + g[j].radius);
//        float a_prime_ij = ((g[i].radius - 0.5 * abs(dn)) * (g[j].radius - 0.5 * abs(dn))) /((g[i].radius - 0.5 * abs(dn))  + (g[j].radius - 0.5 * abs(dn)) );
//
//        vijr[0] = -1 * a_prime_ij * (crossi[0] - crossj[0]);
//        vijr[1] = -1 * a_prime_ij * (crossi[1] - crossj[1]);
//        vijr[2] = -1 * a_prime_ij * (crossi[2] - crossj[2]);
//
//        vijo[0] = a_ij * (doti - dotj) * nij[0];
//        vijo[1] = a_ij * (doti - dotj) * nij[1];
//        vijo[2] = a_ij * (doti - dotj) * nij[2];

        //ut ur uo pre must be added here

        ut[0] =  dt * vijt[0];
        ut[1] =  dt * vijt[1];
        ut[2] =  dt * vijt[2];
        float fs[3] = {0.f};
        float tan[3] = {0.f};
        fs[0] = -1 * ks * ut[0] - nus * vijt[0];
        fs[1] = -1 * ks * ut[1] - nus * vijt[1];
        fs[2] = -1 * ks * ut[2] - nus * vijt[2];

//        tan[0]=fs[0];
//        tan[1]=fs[1];
//        tan[2]=fs[2];
//        memcpy (tan, fs, sizeof(fs));

//        if (norm(tan)>0.f) {
//            tan[0] = tan[0]/norm(tan);
//            tan[1] = tan[1]/norm(tan);
//            tan[2] = tan[2]/norm(tan);
//        }
        float maxFs = params.mu * norm(fn);
        if( norm(fs) > maxFs){
            fs[0] = maxFs/norm(fs)*fs[0];
            fs[1] = maxFs/norm(fs)*fs[1];
            fs[2] = maxFs/norm(fs)*fs[2];

//            ut[0] = -1 / ks * (maxFs*tan[0]);
//            ut[1] = -1 / ks * (maxFs*tan[1]);
//            ut[2] = -1 / ks * (maxFs*tan[2]);
        }

//        ur[0] = dt * vijr[0];
//        ur[1] = dt * vijr[1];
//        ur[2] = dt * vijr[2];
//
//        float fr[3] = {0};
//        float torque[3] = {0};
//        float roll_[3] = {0};
//
//        fr[0] = -1 * kr * ur[0] - nur * vijr[0];
//        fr[1] = -1 * kr * ur[1] - nur * vijr[1];
//        fr[2] = -1 * kr * ur[2] - nur * vijr[2];
//
//        roll_[0]=fr[0];
//        roll_[1]=fr[1];
//        roll_[2]=fr[2];
//
//        if (norm(roll_)>0.0) {
//            roll_[0]/=norm(roll_);
//            roll_[1]/=norm(roll_);
//            roll_[2]/=norm(roll_);
//        }
//
//        float maxFr = params.mur * abs(fnabs);
//        if( norm(fr) > maxFr){
//            fr[0] = maxFr * roll_[0];
//            fr[1] = maxFr * roll_[1];
//            fr[2] = maxFr * roll_[2];
//
//            ur[0] = -1 / kr * (maxFr * roll_[0]);
//            ur[1] = -1 / kr * (maxFr * roll_[1]);
//            ur[2] = -1 / kr * (maxFr * roll_[2]);
//        }
//        float crossT[3] = {0};
//        cross(nij, fr, crossT);
//        scalarMulti(a_ij, crossT, torque);
//
//        uo[0] = dt * vijo[0];
//        uo[1] = dt * vijo[1];
//        uo[2] = dt * vijo[2];
//
//        float fo[3] = {0};
//        float torsion[3] = {0};
//        float tor_[3] = {0};
//
//        fo[0] = -1 * ko * uo[0] - nuo * vijo[0];
//        fo[1] = -1 * ko * uo[1] - nuo * vijo[1];
//        fo[2] = -1 * ko * uo[2] - nuo * vijo[2];
//
//        tor_[0]=fr[0];
//        tor_[1]=fr[1];
//        tor_[2]=fr[2];
//
//        if (norm(tor_)>0.0) {
//            tor_[0]/=norm(tor_);
//            tor_[1]/=norm(tor_);
//            tor_[2]/=norm(tor_);
//        }
//
//        float maxFo = params.muo * abs(fnabs);
//        if( norm(fo) > maxFo){
//            fo[0] = maxFo * fo[0] / norm(fo);
//            fo[1] = maxFo * fo[1] / norm(fo);
//            fo[2] = maxFo * fo[2] / norm(fo);
//
//            uo[0] = -1 / ko * (maxFo * tor_[0]);
//            uo[1] = -1 / ko * (maxFo * tor_[1]);
//            uo[2] = -1 / ko * (maxFo * tor_[2]);
//        }
//
//        scalarMulti(a_ij, fo, torsion);

        float f_[3] = {0};
        addArray(fn, fs, f_);
        atomicAdd(&g[i].ax, f_ [0]);
        atomicAdd(&g[i].ay, f_ [1]);
        atomicAdd(&g[i].az, f_ [2]);

        atomicAdd(&g[j].ax, -1 * f_[0]);
        atomicAdd(&g[j].ay, -1 * f_[1]);
        atomicAdd(&g[j].az, -1 * f_[2]);

        float crossthi[3] = {0};
        float fff[3] = {0};
        cross(nij, fs,crossthi);
        fff[0] = -1 * (g[i].radius - 0.5 * abs(dn)) * crossthi[0];
        fff[1] = -1 * (g[i].radius - 0.5 * abs(dn)) * crossthi[1];
        fff[2] = -1 * (g[i].radius - 0.5 * abs(dn)) * crossthi[2];
        atomicAdd(&g[i].athx, fff[0]);
        atomicAdd(&g[i].athy, fff[1]);
        atomicAdd(&g[i].athz, fff[2]);
        float temp = (g[j].radius - 0.5 * abs(dn)) / (g[i].radius - 0.5 * abs(dn));
        atomicAdd(&g[j].athx, temp*fff[0]);
        atomicAdd(&g[j].athy, temp*fff[1]);
        atomicAdd(&g[j].athz, temp*fff[2]);

    }



}


__global__ void verletList (grains *g,int size, float dt, int *head, int *list,
                            float mesh_resol,
                            int mesh_x, int mesh_y, int mesh_z, float x_min, float y_min, float z_min,float x_max, float y_max, float z_max ){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size) return;

    int ix = floor((g[tid].x - x_min)/mesh_resol);
    int iy = floor((g[tid].y - y_min)/mesh_resol);
    int iz = floor((g[tid].z - z_min)/mesh_resol);


    for (int z=-1; z<2; z++){
        for (int y=-1; y<2;y++){
            for (int x=-1; x<2;x++){

                int ix_neighbour = ix + x;
                int iy_neighbour = iy + y;
                int iz_neighbour = iz + z;

                if(ix_neighbour<0 || iy_neighbour<0 || iz_neighbour<0) continue;
                int ncell_neighbour = ix_neighbour + (iy_neighbour*mesh_x) + (mesh_x*mesh_y*iz_neighbour);
                if(ncell_neighbour>= mesh_x*mesh_y*mesh_z) continue;

                int j = head[ncell_neighbour];

//
                while(j>-1 && j>tid)
                {

                    dryParticles_force( g, tid, j,  dt);
                    j = list[j];

                }
            }
        }

    }

}


__device__ void wallKernel( int i,grains *g,float gap, float *nij, float dt)
{
//    float m_eff = g[i].mass;
//    float kn = 2.f * params.E * g[i].radius/2.f;
//    float damp = -1.f * log(params.en)/sqrt(log(params.en) * log(params.en) + M_PI* M_PI);
//    float dampN = 2.f * sqrt(kn * m_eff) * damp;
//    float ks = params.ks_to_kn * kn;
//    float dampS = params.nus_to_nun * dampN;

    float kn = 2 * params.E * g[i].radius/2.;
    float ks = kn  * params.ks_to_kn ;
    float meff = g[i].mass;
    float damp = -1.f * log(params.en)/sqrt(log(params.en) * log(params.en) + M_PI * M_PI);
    float nun = 2.f * sqrt(kn * meff) * damp  ;
    float nus = params.nus_to_nun * nun;

    float relVelij[3]= {0};
    relVelij[0] = g[i].vx ;
    relVelij[1] = g[i].vy ;
    relVelij[2] = g[i].vz ;


    float fnabs = kn * abs(gap) - nun * dot(relVelij, nij) ;
//    if (fnabs < 0.){
//        fnabs = 0.;
//    }

    float fn[3] = {0};
    fn[0] = fnabs * nij[0];
    fn[1] = fnabs * nij[1];
    fn[2] = fnabs * nij[2];

    float vthi[3] = {0};
    float crossi[3] = {0};
    float vij[3] = {0};
    float vijt[3] = {0};
    float ut[3] = {0};

    vthi[0] = g[i].vthx;
    vthi[1] = g[i].vthy;
    vthi[2] = g[i].vthz;

    cross(nij, vthi,crossi);


    vij[0] = relVelij[0]  ;
    vij[1] = relVelij[1]  ;
    vij[2] = relVelij[2]  ;

    vijt[0] = vij[0] - nij[0] * dot(nij, vij);
    vijt[1] = vij[1] - nij[1] * dot(nij, vij);
    vijt[2] = vij[2] - nij[2] * dot(nij, vij);


    ut[0] =  dt * vijt[0];
    ut[1] =  dt * vijt[1];
    ut[2] =  dt * vijt[2];
    float fs[3] = {0};
    float tan[3] = {0};
    fs[0] = -1 * ks * ut[0] - nus * vijt[0];
    fs[1] = -1 * ks * ut[1] - nus * vijt[1];
    fs[2] = -1 * ks * ut[2] - nus * vijt[2];

//    tan[0]=fs[0];
//    tan[1]=fs[1];
//    tan[2]=fs[2];

//    memcpy (tan, fs, sizeof(fs));
//
//
//    if (norm(tan)>0.0) {
//        tan[0]/=norm(tan);
//        tan[1]/=norm(tan);
//        tan[2]/=norm(tan);
//    }
    float maxFs = params.mu * abs(fnabs);
    if( norm(fs) > maxFs){
        fs[0] = maxFs/norm(fs)*fs[0];
        fs[1] = maxFs/norm(fs)*fs[1];
        fs[2] = maxFs/norm(fs)*fs[2];

//        ut[0] = -1 / ks * (maxFs*tan[0]);
//        ut[1] = -1 / ks * (maxFs*tan[1]);
//        ut[2] = -1 / ks * (maxFs*tan[2]);
    }

    float f_[3] = {0};
    addArray(fn, fs, f_);
    atomicAdd(&g[i].ax, f_ [0]);
    atomicAdd(&g[i].ay, f_ [1]);
    atomicAdd(&g[i].az, f_ [2]);

    float crossthi[3] = {0};
    float fff[3] = {0};
    cross(nij, fs,crossthi);
    fff[0] = -1 * (g[i].radius - 0.5 * abs(gap)) * crossthi[0];
    fff[1] = -1 * (g[i].radius - 0.5 * abs(gap)) * crossthi[1];
    fff[2] = -1 * (g[i].radius - 0.5 * abs(gap)) * crossthi[2];
    atomicAdd(&g[i].athx, fff[0]);
    atomicAdd(&g[i].athy, fff[1]);
    atomicAdd(&g[i].athz, fff[2]);



}


__device__ void sphereCylinder(int i, grains *g, cylinder *cyl, float dt){


    float id = -1;
    float gap = 0;
    float nij[3] = {0};
    float g_position[3] = {g[i].x, g[i].y, g[i].z};
    float e[3] = {0};
    float m[3] = {0};
    float d[3] = {0};
    float rq[3] = {0};
    float d_ = 0;

    e[0] = cyl->top[0]- cyl->bot[0];
    e[1] = cyl->top[1]- cyl->bot[1];
    e[2] = cyl->top[2]- cyl->bot[2];
    cross(cyl->bot, cyl->top, m);
    float temp[3]={0};
    cross(e, g_position, temp);
    d[0] = m[0] + temp[0];
    d[1] = m[1] + temp[1];
    d[2] = m[2] + temp[2];
    d_ = norm(d)/norm(e);
    float cross1[3]={0};
    float cross2[3]={0};
    float add1[3] = {0};

    cross(e, g_position, cross2);
    add1[0]=m[0]+cross2[0];
    add1[1]=m[0]+cross2[1];
    add1[2]=m[0]+cross2[2];
    cross(e, add1, cross1);
    cross1[0] = cross1[0] * 1./dot(e,e);
    cross1[1] = cross1[1] * 1./dot(e,e);
    cross1[2] = cross1[2] * 1./dot(e,e);
    rq[0] = g_position[0] + cross1[0];
    rq[1] = g_position[1] + cross1[1];
    rq[2] = g_position[2] + cross1[2];




    if((d_ + g[i].radius - cyl->r )>0){
        id = 1;
        gap = -1 * (d_ + g[i].radius - cyl->r);
        nij[0] = rq[0] - g_position[0];
        nij[1] = rq[1] - g_position[1];
        nij[2] = rq[2] - g_position[2];
        if(norm(nij) != 0)
        {
            nij[0] = nij[0]/ norm(nij);
            nij[1] = nij[1]/ norm(nij);
            nij[2] = nij[2]/ norm(nij);
        }
    }

    if(id != -1){
        wallKernel( i,g, gap, nij,  dt);
    }
}


__global__ void wallForce(grains *g,  cone *walls, cylinder *cyl, float dt, int size ){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) return;

    float gap = 0.f; float gap_e1 =0.f; float gap_e2 =0.f;
    int id = -1;

    float nij[3]= {0};float nij_e1[3]= {0}; float nij_e2[3]= {0};
    float AP[3]= {0};float coneVr[3]= {0};float VP[3]= {0};float VrP[3]= {0};

    float gamma =0.f;float gammar = 0.f;float AB_len = 0.f;float VE_len =0.f;

    AP[0] = g[i].x - walls[1].top[0];
    AP[1] = g[i].y - walls[1].top[1];
    AP[2] = g[i].z - walls[1].top[2];

    float SF = -1 * dot(AP, walls[1].axis) / (walls[0].height + walls[1].height);

    if(SF < (-1 * g[i].radius * cosf(walls[1].alpha)/(walls[0].height + walls[1].height)))
    {
        sphereCylinder( i,  g, cyl,  dt);
    }

    else if( (-1 * g[i].radius * cosf(walls[1].alpha)/(walls[0].height + walls[1].height))<=SF && SF<=0.5)
    {
        // upper wall
        float coeff = g[i].radius/sinf(walls[1].theta/2.f);
        coneVr[0] = walls[1].coneV[0] + coeff*walls[1].axis[0];
        coneVr[1] = walls[1].coneV[1] + coeff*walls[1].axis[1];
        coneVr[2] = walls[1].coneV[2] + coeff*walls[1].axis[2];

        VP[0] = g[i].x - walls[1].coneV[0];
        VP[1] = g[i].y - walls[1].coneV[1];
        VP[2] = g[i].z - walls[1].coneV[2];

        VrP[0] = g[i].x - coneVr[0];
        VrP[1] = g[i].y - coneVr[1];
        VrP[2] = g[i].z - coneVr[2];

        gamma = acosf(dot(VP, walls[1].axis)/norm(VP));
        gammar = acosf(dot(VrP, walls[1].axis)/norm(VrP));

        if((gammar > walls[1].theta/2.f) && (gamma < walls[1].theta / 2.f) )
        {
            VE_len = norm(VP) * sinf(M_PI/2 - (walls[1].theta/2. - gamma)) / sinf(M_PI/2. - walls[1].theta/2.);
            AB_len = norm(VP) * sinf(walls[1].theta/2 - gamma);
            gap = AB_len - g[i].radius;
            if(gap < 0.)
            {
                id = 1;
                float coneE[3] = {0};
                coneE[0] = walls[1].coneV[0] + VE_len*walls[1].axis[0];
                coneE[1] = walls[1].coneV[1] + VE_len*walls[1].axis[1];
                coneE[2] = walls[1].coneV[2] + VE_len*walls[1].axis[2];

                nij[0] = coneE[0] - g[i].x;
                nij[1] = coneE[1] - g[i].y;
                nij[2] = coneE[2] - g[i].z;

                if(norm(nij) != 0.)
                {
                    normalize(nij);
                }
            }


        }
    }
    else if( 0.5 <SF && SF <= (1 - g[i].radius /(walls[0].height + walls[1].height)))
    {
        // lower wall
        float coeff = g[i].radius/sinf(walls[0].theta/2);
        coneVr[0] = walls[0].coneV[0] + coeff*walls[0].axis[0];
        coneVr[1] = walls[0].coneV[1] + coeff*walls[0].axis[1];
        coneVr[2] = walls[0].coneV[2] + coeff*walls[0].axis[2];

        VP[0] = g[i].x - walls[0].coneV[0];
        VP[1] = g[i].y - walls[0].coneV[1];
        VP[2] = g[i].z - walls[0].coneV[2];

        VrP[0] = g[i].x - coneVr[0];
        VrP[1] = g[i].y - coneVr[1];
        VrP[2] = g[i].z - coneVr[2];

        gamma = acosf(dot(VP, walls[0].axis)/norm(VP));
        gammar = acosf(dot(VrP, walls[0].axis)/norm(VrP));

        if((gammar > walls[0].theta/2.) && (gamma < walls[0].theta/2.) )
        {
            VE_len = norm(VP) * sinf(M_PI/2 - (walls[0].theta/2. - gamma)) / sinf(M_PI/2. - walls[0].theta/2.);
            AB_len = norm(VP) * sinf(walls[0].theta/2 - gamma);
            gap = AB_len - g[i].radius;
            if(gap < 0.)
            {
                id = 3;
                float coneE[3] = {0};
                coneE[0] = walls[0].coneV[0] + VE_len*walls[0].axis[0];
                coneE[1] = walls[0].coneV[1] + VE_len*walls[0].axis[1];
                coneE[2] = walls[0].coneV[2] + VE_len*walls[0].axis[2];

                nij[0] = coneE[0] - g[i].x;
                nij[1] = coneE[1] - g[i].y;
                nij[2] = coneE[2] - g[i].z;
                if(norm(nij) != 0.)
                {
                    normalize(nij);
                }
            }
        }

    }

    else if( (1 - g[i].radius/(walls[0].height + walls[1].height) )< SF && SF<= 1)
    {
        float coeff = g[i].radius/sinf(walls[0].theta/2);
        coneVr[0] = walls[0].coneV[0] + coeff*walls[0].axis[0];
        coneVr[1] = walls[0].coneV[1] + coeff*walls[0].axis[1];
        coneVr[2] = walls[0].coneV[2] + coeff*walls[0].axis[2];

        VP[0] = g[i].x - walls[0].coneV[0];
        VP[1] = g[i].y - walls[0].coneV[1];
        VP[2] = g[i].z - walls[0].coneV[2];

        VrP[0] = g[i].x - coneVr[0];
        VrP[1] = g[i].y - coneVr[1];
        VrP[2] = g[i].z - coneVr[2];

        gamma = acosf(dot(VP, walls[0].axis)/norm(VP));
        gammar = acosf(dot(VrP, walls[0].axis)/norm(VrP));

        if(gammar <walls[0].theta/2.)
        {
            // lower plane
            float temp[3] = {0};
            temp[0] = walls[0].coneV[0] - walls[0].bot[0];
            temp[1] = walls[0].coneV[1] - walls[0].bot[1];
            temp[2] = walls[0].coneV[2] - walls[0].bot[2];
            float h_ = norm(temp);
            gap = h_ - dot(VP, walls[0].axis) - g[i].radius;
            if(gap < 0.)
            {
                id = 4;
                nij[0] = -1 * walls[0].axis[0];
                nij[1] = -1 * walls[0].axis[1];
                nij[2] = -1 * walls[0].axis[2];
            }
        }
        else if(gammar > walls[0].theta/2. && gamma < walls[0].theta/2.)
        {
            //low edge
            // 1: checking gap with lower plane
            float temp2[3] = {0};
            temp2[0] = walls[0].coneV[0] - walls[0].bot[0];
            temp2[1] = walls[0].coneV[1] - walls[0].bot[1];
            temp2[2] = walls[0].coneV[2] - walls[0].bot[2];
            float h_1 = norm(temp2);
            gap_e1 = h_1 - dot(VP, walls[0].axis) - g[i].radius;
            //checking lower wall
            VE_len = norm(VP) * sinf(M_PI/2 - (walls[0].theta/2. - gamma)) / sinf(M_PI/2. - walls[0].theta/2.);
            AB_len = norm(VP) * sinf(walls[0].theta/2 - gamma);
            gap_e2 = AB_len - g[i].radius;
            if(gap_e1 < 0. && gap_e2 < 0. )
            {
                id = 5;
                nij_e1[0] = -1 * walls[0].axis[0];
                nij_e1[1] = -1 * walls[0].axis[1];
                nij_e1[2] = -1 * walls[0].axis[2];

                float coneE[3] = {0};
                coneE[0] = walls[0].coneV[0] + VE_len*walls[0].axis[0];
                coneE[1] = walls[0].coneV[1] + VE_len*walls[0].axis[1];
                coneE[2] = walls[0].coneV[2] + VE_len*walls[0].axis[2];

                nij_e2[0] = coneE[0] - g[i].x;
                nij_e2[1] = coneE[1] - g[i].y;
                nij_e2[2] = coneE[2] - g[i].z;

                if(norm(nij_e2) > 0.){
//                    nij_e2[0] = nij_e2[0]/norm(nij_e2);
//                    nij_e2[1] = nij_e2[1]/norm(nij_e2);
//                    nij_e2[2] = nij_e2[2]/norm(nij_e2);
                    normalize(nij_e2);
                }
            }
        }

    }

    if(id != -1 && id == 5)
    {
        wallKernel( i, g, gap_e1, nij_e1,dt);
        wallKernel( i, g, gap_e2, nij_e2,dt);
        return ;

    }

     if(id != -1 && id != 5)
    {
        wallKernel( i,g, gap, nij,dt);
        return ;
    }

}


void set_debug_sample();
void set_sample();
void setWalls();
void identify_bonds();
void output_grains(int numfile, int num, grains* g);
void identify_bonds(DemParams h_params);
void write_txt(grains *g, int size);
int main()
{
    const int timeStep = 200000;
    const int updteGrid = 1;
    const int outputFreq = 5000;
    const int terminalTimeStepInterval = 1000;

    //Timestep:
    dt_crt = 1e-5;
    dt = dt_crt;
    std::cout<<"simulation dt: "<<dt<<std::endl;

    DemParams h_params;
    h_params.gx = 0.;
    h_params.gy = -9.81;
    h_params.gz = 0.;
    h_params.en =  0.2;
    h_params.E = 1e6;
    h_params.ks_to_kn = 0.5;
    h_params.kr_to_kn = 0.1;
    h_params.ko_to_kn = 0.1;
    h_params.mu = 0.8;
    h_params.mur = 0.f;
    h_params.muo = 0.f;
    h_params.nus_to_nun = 0.2;
    h_params.nur_to_nun = 0.1;
    h_params.nuo_to_nun = 0.1;

    //creating sparse matrix
    cudaMemcpyToSymbol(params,&h_params, sizeof(DemParams));

    // host: setting samples and walls
    set_sample();
//    set_debug_sample();

    setWalls();
    cylwall.r = r_big;
    cylwall.top[0]=0.;
    cylwall.top[1]=4.;
    cylwall.top[2]=0.;
    cylwall.bot[0] = 2 * top[0];
    cylwall.bot[1] = 2 * top[1];
    cylwall.bot[2] = 2 * top[2];

    h_[0] = top[0] - bot[0];
    h_[1] = top[1] - bot[1];
    h_[2] = top[2] - bot[2];

    const int GRAINS = g_vec.size();
    std::cout<<"Number of generated particles: "<<GRAINS<<std::endl;
    grains *g= new grains[GRAINS];
    std::copy(g_vec.begin(), g_vec.end(), g);

    //device: setting samples and walls
    grains *d_g;
    cone *d_walls;
    cylinder *d_cylwall;
    cudaMalloc((void**)& d_g, GRAINS * sizeof(grains));
    cudaMalloc((void**)& d_walls, 2 * sizeof(cone));
    cudaMalloc((void**)& d_cylwall, sizeof(cylinder));

    cudaMemcpy(d_g, g, GRAINS * sizeof(grains), cudaMemcpyHostToDevice);
    cudaMemcpy(d_walls, cowall, 2 * sizeof(cone), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cylwall, &cylwall, sizeof(cylinder), cudaMemcpyHostToDevice);



    //device: LC algorithm
//    mesh resolution and initilizing LC line 1306 pablo code
    x_min = bot[0] - 2 * r_big;
    x_max = top[0] + 2 * r_big;

    y_min = bot[1];
    y_max = top[1] * 6;

    z_min = bot[2] - 2 * r_big;
    z_max = top[2] + 2 * r_big;

    float mesh_resol = 3 * rmax;
    const int mesh_x = floor((x_max - x_min)/mesh_resol) + 1;
    const int mesh_y = floor((y_max - y_min)/mesh_resol) + 1;
    const int mesh_z = floor((z_max - z_min)/mesh_resol) + 1;

    int *d_head;
    int *d_list;

    cudaMalloc((void**)& d_head, mesh_x * mesh_y * mesh_z * sizeof(int));
    cudaMalloc((void**)& d_list, GRAINS * sizeof(int));

    cudaMemset(d_head, -1, mesh_x * mesh_y * mesh_z * sizeof(int));
    cudaMemset(d_list, -1, GRAINS * sizeof(int));

    const int nThreadsPerBlocks  = 64;
    const int nBlocks = (GRAINS + nThreadsPerBlocks -1)/nThreadsPerBlocks;
    sortingParticles_cuda<<<1,1>>>(d_g, GRAINS,d_head, d_list,mesh_resol,mesh_x,mesh_y,mesh_z, x_min,y_min,z_min, x_max, y_max, z_max);

    for(int step = 0; step< timeStep; step++)
    {

        if(step % terminalTimeStepInterval == 0)
        {
            std::cout<<"timestep: "<<step<<std::endl;
        }

        //device: update LC Grid
        if(step%updteGrid == 0)
        {
            cudaMemset(d_head, -1, mesh_x*mesh_y*mesh_z * sizeof(int));
            cudaMemset(d_list, -1, GRAINS * sizeof(int));
            sortingParticles_cuda<<<1,1>>>(d_g, GRAINS,d_head, d_list,mesh_resol,mesh_x,mesh_y,mesh_z, x_min,y_min,z_min, x_max, y_max, z_max);
        }

        //device: pre force cuda
        preForce_cuda<<<nBlocks,nThreadsPerBlocks>>>(d_g,dt, GRAINS);
        //device: sphere-sphere interactions
        verletList<<<nBlocks,nThreadsPerBlocks>>>(d_g, GRAINS, dt, d_head, d_list,mesh_resol,mesh_x,mesh_y,mesh_z, x_min,y_min,z_min, x_max,y_max,z_max);
        //device: sphere-wall interactions
        wallForce<<<nBlocks,nThreadsPerBlocks>>>(d_g, d_walls,d_cylwall, dt, GRAINS );
        //device: post force
        postForce_cuda<<<nBlocks,nThreadsPerBlocks>>>(d_g,dt, GRAINS);

        if(step%outputFreq == 0)
        {
            cudaMemcpy(g, d_g, GRAINS * sizeof(grains), cudaMemcpyDeviceToHost);
            output_grains(step, (int)GRAINS, g);
            if(step != 0) write_txt(g, GRAINS);
        }



    }
    //device: free memory
    cudaDeviceSynchronize();
    cudaFree(d_g);
    cudaFree(d_walls);
    cudaFree(d_cylwall);
    cudaFree(d_head);
    cudaFree(d_list);
    delete [] g;

    std::cout << "end of execution form CPU!" << std::endl;
    return 0;

}



void set_debug_sample()
{
    const int ng = 3;
    float R[ng];
    grains g0;
    for(int i=0;i<ng;i++) {g_vec.push_back(g0);}

    for(int i = 0 ; i < ng ; i++) {
        g_vec[i].id = i;
        g_vec[i].radius = 0.002;
        g_vec[i].mass = (4. / 3.) * M_PI *rhos*g_vec[i].radius*g_vec[i].radius*g_vec[i].radius;
        g_vec[i].inertia = (2. / 5.) * g_vec[i].mass * g_vec[i].radius * g_vec[i].radius;
        g_vec[i].vx = 0.;
        g_vec[i].vy = 0.;
        g_vec[i].vz = 0.;
        g_vec[i].ax = 0. ;
        g_vec[i].ay = 0. ;
        g_vec[i].az = 0. ;
        g_vec[i].vthx = 0.;
        g_vec[i].vthy = 0.;
        g_vec[i].vthz = 0.;
        g_vec[i].athx = 0. ;
        g_vec[i].athy = 0. ;
        g_vec[i].athz = 0. ;


    }
    //position_grain1 = np.array([0, 0.1, 0])
    //position_grain2 = np.array([0.009998, 0.1, 0])
    g_vec[0].x =0.009;
    g_vec[0].y = 0.04;
    g_vec[0].z = 0.01;

//    g_vec[1].x =0.01499;
    g_vec[1].x =0.015;
    g_vec[1].y = 0.04;
    g_vec[1].z = 0.01;

    g_vec[2].x = 0.01;
    g_vec[2].y = 0.004;
    g_vec[2].z = 0.01;

}



void output_grains(int numfile, int num, grains *g)
{
    int i;
    char fname[25]; // file name
    sprintf(fname,"VTK/grains%04d.vtk",numfile);


    std::ofstream fog(fname, std::ofstream::out);
    if (fog)
    {
//        std::vector<double> g_v = {g[i].vx,g[i].vy,g[i].vz};
        fog.precision(5); fog << std::scientific;
        fog << "# vtk DataFile Version 3.0" << std::endl;
        fog << "Particle data" << std::endl;
        fog << "ASCII" << std::endl;
        fog << "DATASET POLYDATA" << std::endl;

        fog << "POINTS " << num << " float" << std::endl;
        for(i=0;i<num;i++) fog << g[i].x <<" "<< g[i].y <<" "<< g[i].z << std::endl;
        fog << "POINT_DATA " << num << std::endl;
        fog << "SCALARS radius float" << std::endl;
        fog << "LOOKUP_TABLE default" << std::endl;
        for(i=0;i<num;i++) fog << g[i].radius << std::endl;
        fog << "SCALARS velocity float" << std::endl;
        fog << "LOOKUP_TABLE default" << std::endl;
        for(i=0;i<num;i++){
            std::vector<float> g_v = {g[i].vx,g[i].vy,g[i].vz};
            fog <<norm(g_v)<< std::endl;
        }




    }
}


void set_sample()
{
    float old_g_number = 0;
#ifdef importOldSample
    std::ifstream file_("sample.txt");
    int id;
    float x_,y_,z_, r_;

    if(file_.is_open()){
        while(file_>>id>>x_>>y_>>z_>>r_){
            //std::cout<<id<<" "<<x_<<" "<<y_<<" "<<z_<<" "<<r_<<'\n'<<std::endl;
            grains g0;
            g_vec.push_back(g0);


            g_vec[id].id = id;
            g_vec[id].radius = r_;
            g_vec[id].mass = 4. / 3. * M_PI * rhos * g_vec[id].radius * g_vec[id].radius * g_vec[id].radius;
            g_vec[id].inertia = 2. / 5. * g_vec[id].mass * g_vec[id].radius * g_vec[id].radius;
            g_vec[id].vx = 0. ;
            g_vec[id].vy = 0. ;
            g_vec[id].vz = 0. ;
            g_vec[id].ax = 0. ;
            g_vec[id].ay = 0. ;
            g_vec[id].az = 0. ;
            g_vec[id].x = x_ ;
            g_vec[id].y = y_ ;
            g_vec[id].z = z_ ;
            g_vec[id].vthx = 0.;
            g_vec[id].vthy = 0.;
            g_vec[id].vthz = 0.;
            g_vec[id].athx = 0. ;
            g_vec[id].athy = 0. ;
            g_vec[id].athz = 0. ;


        }

    }
    file_.close();
    old_g_number = g_vec.size();
#endif


#ifdef creatNewSample
//    std::cout<<"size of grains: "<<sizeof(g)/sizeof(grains)<<std::endl;
    const int ng = nbgrains;
    float R[ng];
    grains g0;
    for(int i=0;i<nbgrains;i++) {g_vec.push_back(g0);}

    srand(time(NULL));
    for (int l = 0 ; l<row_number;l++){

        for (int i = 0 ; i<nbg_row; i++){
            float Smax = M_PI * rmax * rmax;
            float Smin = M_PI * rmin * rmin;
            float S = Smin + (i/(float(nbg_row) -1))*(Smax-Smin);
            R[l * nbg_row + i] = sqrtf(S/M_PI);

        }

        // swapping 10*ng times
        for(int k = 0 ; k < 10*nbg_row ; k++) {
            int i = rand()%nbg_row;
            int j = rand()%nbg_row;
            float tmpRi = R[l * nbg_row +i];
            R[l * nbg_row + i] = R[l * nbg_row +j];
            R[l * nbg_row + j] = tmpRi;
        }

        for(int i = 0 ; i < nbg_row ; i++) {
            int index = l * nbg_row +i+old_g_number;
            float PV = 0.1 * rmin * 1000;


            g_vec[index].id = index ;

//        std::cout<<"here is g id: "<<g_vec[index].id<<std::endl;
            g_vec[index].radius = R[l * nbg_row +i];
            g_vec[index].mass = 4. / 3. * M_PI *rhos*g_vec[index].radius*g_vec[index].radius*g_vec[index].radius;
            g_vec[index].inertia = 2. / 5.*g_vec[index].mass*g_vec[index].radius*g_vec[index].radius;
            g_vec[index].vx = PV*(((float) rand() / RAND_MAX) - 0.5);
            g_vec[index].vy = PV*(((float) rand() / RAND_MAX) - 0.5);
            g_vec[index].vz = PV*(((float) rand() / RAND_MAX) - 0.5);
            g_vec[index].ax = 0. ;
            g_vec[index].ay = 0. ;
            g_vec[index].az = 0. ;
            g_vec[index].athx = 0.;
            g_vec[index].athy = 0.;
            g_vec[index].athy = 0.;
            g_vec[index].vthx = 0.;
            g_vec[index].vthy = 0.;
            g_vec[index].vthz = 0.;

            int column = i%ngw;
            int row = i/ngw;
            if(row%2 == 0){
                g_vec[index].x = -1* pow(2,0.5)/2*0.015 + rmax + 2*column*rmax  - 0.001;
            }
            else{
                g_vec[index].x = -1* pow(2,0.5)/2*0.015 + 2*rmax + 2*column*rmax  - 0.001;
            }
            g_vec[index].z =   -0.01  + 2*row*rmax ;
            g_vec[index].y = first_row_height;



        }

        first_row_height -= (2 * rmax + rmin * (((float) rand() / (RAND_MAX)) + 1));

    }
#endif

}


void setWalls()
{

    //lower cone
    cowall[0].id = 0;
    cowall[0].r_bot = r_big;
    cowall[0].r_top = r_small;

    cowall[0].top[0] = top[0];
    cowall[0].top[1] = top[1];
    cowall[0].top[2] = top[2];

    cowall[0].bot[0] = bot[0];
    cowall[0].bot[1] = bot[1];
    cowall[0].bot[2] = bot[2];

    cowall[0].height = (float)top[1];

    cowall[0].axis[0] = 0.f;
    cowall[0].axis[1] = -1.f;
    cowall[0].axis[2] = 0.f;

//   float y = walls[0].height * (r_s / r_b) / (1 - (r_s / r_b));
    float y =  cowall[0].height * (r_small / r_big) / (1 - (r_small / r_big));

    cowall[0].coneV[0] = top[0] - cowall[0].axis[0] * y;
    cowall[0].coneV[1] = top[1] - cowall[0].axis[1] * y;
    cowall[0].coneV[2] = top[2] - cowall[0].axis[2] * y;

    cowall[0].alpha = atan2f(cowall[0].height, r_big - r_small);
    cowall[0].theta = M_PI - 2 * cowall[0].alpha;


   // upper cone
    cowall[1].id = 1;
    cowall[1].r_bot = r_small;
    cowall[1].r_top = r_big;

    cowall[1].top[0] = 2*top[0];
    cowall[1].top[1] = 2*top[1];
    cowall[1].top[2] = 2*top[2];

    cowall[1].bot[0] = top[0];
    cowall[1].bot[1] = top[1];
    cowall[1].bot[2] = top[2];

    cowall[1].height = top[1];

    cowall[1].axis[0] = 0.f;
    cowall[1].axis[1] = 1.f;
    cowall[1].axis[2] = 0.f;

    cowall[1].coneV[0] = top[0] - cowall[1].axis[0] * y;
    cowall[1].coneV[1] = top[1] - cowall[1].axis[1] * y;
    cowall[1].coneV[2] = top[2] - cowall[1].axis[2] * y;

    cowall[1].alpha = atan2f(cowall[1].height, r_big - r_small);
    cowall[1].theta = M_PI - 2 * cowall[1].alpha;

}

void write_txt(grains *g, int size)
{
    std::ofstream file_;
    file_.open("sample.txt");
    for(int i=0;i<size;i++){
        file_<<g[i].id<<" "<<g[i].x<<" "<<g[i].y<<" "<<g[i].z<<" "<<g[i].radius<<'\n'<<std::endl;
    }
    file_.close();

}

#ifdef cohesion
void identify_bonds(DemParams par){
    float nb = 0;
    for(int i=0; i<nbgrains; i++) {
        for(int j=i+1; j<nbgrains; j++){
            float xij, yij, zij, cij, dn;
            xij = g[i].x - g[j].x;
            yij = g[i].y - g[j].y;
            zij = g[i].z - g[j].z;
            cij = sqrtf(xij*xij + yij*yij + zij*zij);
            dn = cij - (g[i].radius + g[j].radius);
//            std::cout<<"i: "<<i<<" j: "<<j<<" dn: "<<dn<<std::endl;
            if(dn < 0.){

                float kn = 2. * par.E * (g[i].radius * g[j].radius)/(g[i].radius + g[j].radius);
                float m_eff = g[i].mass * g[j].mass/(g[i].mass + g[j].mass);
                float damp = -1 * logf(par.en)/sqrtf(logf(par.en) * logf(par.en) + M_PI* M_PI);
                float nun = 0.;

                kn = 1 * kn;
                float ks = 0.5 * kn;
                float kr = 2e-4 * kn;
                float ko = 2e-4* kn;
                float nus = 0.5 * nun;
                float nur = 0.1 * nun;
                float nuo = 0.1 * nun;
                bond b0;
                b0.gi = i;
                b0.gj = j;
                b0.isActive = true;
                b0.dn_init = dn;
                b0.ut[0] = 0.;
                b0.ut[1] = 0.;
                b0.ut[2] = 0.;
                b0.ur[0] = 0.;
                b0.ur[1] = 0.;
                b0.ur[2] = 0.;
                b0.uo[0] = 0.;
                b0.uo[1] = 0.;
                b0.uo[2] = 0.;
                b0.fn[0] = 0.;
                b0.fn[1] = 0.;
                b0.fn[2] = 0.;
                b0.fs[0] = 0.;
                b0.fs[1] = 0.;
                b0.fs[2] = 0.;
                b0.torque[0] = 0.;
                b0.torque[1] = 0.;
                b0.torque[2] = 0.;
                b0.torsion[0] = 0.;
                b0.torsion[1] = 0.;
                b0.torsion[2] = 0.;
                b0.rupture = -1.;
                b0.kn = kn;
                b0.ks = ks;
                b0.kr = kr;
                b0.ko = ko;
                b0.nun = nun;
                b0.nus = nus;
                b0.nur = nur;
                b0.nuo = nuo;

                b.push_back(b0);
                indexi.push_back(i);
                indexj.push_back(j);
                indexb.push_back(nb);
                nb++;


            }


        }
    }
}


void sparseMatrix(sMatrix smatrix, bond *barray){
    for (int k = 0 ; k < b.size(); k++){
        smatrix.row[k] = indexi[k];
        smatrix.column[k] = indexj[k];
        smatrix.value[k] = indexb[k];
        barray[k] = b[k];
    }
}
#endif



