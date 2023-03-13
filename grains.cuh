//
// Created by Mohammad Sanayei on 13.07.21.
//

#ifndef DEM_GPU_GRAINS_CUH
#define DEM_GPU_GRAINS_CUH
#include"contact_history.cuh"

//template<int N>
struct grains{
    int id;
    float radius;
    float x,y,z;
    float thx,thy,thz;
    float vx,vy,vz;
    float vthx,vthy,vthz;
    float ax,ay,az;
    float athx,athy,athz;
    float mass;
    float inertia;
    //contact oldinfo[N];
    //int bool_oldinfo[N];


};
#endif //DEM_GPU_GRAINS_CUH
