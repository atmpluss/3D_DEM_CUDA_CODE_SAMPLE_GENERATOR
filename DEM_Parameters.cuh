//
// Created by msanayei on 11/10/2021.
//

#ifndef INC_3_CUDA_SETTLING_PARTICLES_DEM_PARAMETERS_CUH
#define INC_3_CUDA_SETTLING_PARTICLES_DEM_PARAMETERS_CUH
struct DemParams{
    float E,en,mu,mur,muo, gx,gy,gz;
    float ks_to_kn,kr_to_kn,ko_to_kn,
            nus_to_nun,nur_to_nun,nuo_to_nun;

};
#endif //INC_3_CUDA_SETTLING_PARTICLES_DEM_PARAMETERS_CUH
