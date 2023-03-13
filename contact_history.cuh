//
// Created by msanayei on 7/17/2021.
//

#ifndef DEM_GPU_CONTACT_HISTORY_CUH
#define DEM_GPU_CONTACT_HISTORY_CUH
struct contact{
    int id;
    float ut[3];
    float ur[3];
    float uo[3];
};
#endif //DEM_GPU_CONTACT_HISTORY_CUH
