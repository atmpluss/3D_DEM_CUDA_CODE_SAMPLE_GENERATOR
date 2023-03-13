//
// Created by msanayei on 11/10/2021.
//

#ifndef INC_3_CUDA_SETTLING_PARTICLES_CONE_DIMENSIONS_CUH
#define INC_3_CUDA_SETTLING_PARTICLES_CONE_DIMENSIONS_CUH
//small cone:
float top[3] = {0.0, 0.081, 0.0};
float bot[3] = {0.,0.,0.};
float h_[3] = {0};
const float r_big = 0.025;
const float r_small = 0.015;

//medium cone:
//float top[3] = {0.0, 0.108, 0.0};
//float bot[3] = {0.,0.,0.};
//float h_[3] = {0};
//const float r_big = 0.0425;
//const float r_small = 0.028;

//large cone:
// float top[3] = {0.0, 0.15, 0.0};
// float bot[3] = {0.,0.,0.};
// float h_[3] = {0};
// const float r_big = 0.06;
// const float r_small = 0.0395;
#endif //INC_3_CUDA_SETTLING_PARTICLES_CONE_DIMENSIONS_CUH
