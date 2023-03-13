//
// Created by Mohammad Sanayei on 03.08.21.
//

#ifndef INC_3_CUDA_SETTLING_PARTICLES_CONEWALL_CUH
#define INC_3_CUDA_SETTLING_PARTICLES_CONEWALL_CUH

struct cone{
    int id;
    float alpha;
    float theta;
    float height;
    float r_top;
    float r_bot;

    float top[3];
    float bot[3];

    float axis[3];
    float coneV[3];
};
#endif //INC_3_CUDA_SETTLING_PARTICLES_CONEWALL_CUH
