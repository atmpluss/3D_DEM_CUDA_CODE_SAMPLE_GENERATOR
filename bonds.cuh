//
// Created by Mohammad Sanayei on 13.07.21.
//

#ifndef DEM_GPU_BONDS_CUH
#define DEM_GPU_BONDS_CUH
struct bond{
//    static float b_c;
//    static float y_n ;
//    static float y_s;
//    static float y_r;
//    static float y_o; // we can put is as constant memory for cuda
    int gi;
    int gj;
    bool isActive;
    float dn_init;
    float  ut[3];
    float  ur[3];
    float  uo[3];
    float  fn[3];
    float  fs[3];
    float  torque[3];
    float  torsion[3];
    float rupture;
    float kn, ks, kr,ko;
    float nun, nus,nur,nuo;

//#ifdef BOND_DAMAGE
//    double ydam;
//    double eta;
//    double dam;
//#endif
};

#endif //DEM_GPU_BONDS_CUH
