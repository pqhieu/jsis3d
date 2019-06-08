#include "cnpy.h"
#include "densecrf.h"


const float confidence = 0.5;


int main(int argc, char* argv[])
{
    if (argc < 2) return 0;

    std::string fname = std::string(argv[1]);
    cnpy::npz_t data = cnpy::npz_load(fname);

    cnpy::NpyArray coords = data["coords"];
    cnpy::NpyArray points = data["points"];
    cnpy::NpyArray pred = data["pred"];

    int N = coords.shape[0];
    int M = 13;

    // Find number of instance labels
    int L = 0;
    int64_t* labels = pred.data<int64_t>();
    for (int i = 1; i < pred.num_vals; i += 2)
        if (labels[i] > L) L = labels[i];
    L = L + 1;

    DenseCRF crf_s(N, M);
    DenseCRF crf_i(N, L);

    // Semantic unary
    MatrixXf unary_s(M, N);
    float n_energy = -log((1.0 - confidence) / (M - 1));
    float p_energy = -log(confidence);
    unary_s.fill(n_energy);
    for (int i = 0; i < N; ++i) {
        int l = labels[i * 2 + 0];
        unary_s(l, i) = p_energy;
    }
    crf_s.setUnaryEnergy(unary_s);

    MatrixXf unary_i(L, N);
    n_energy = -log((1.0 - confidence) / (L - 1));
    p_energy = -log(confidence);
    unary_i.fill(n_energy);
    for (int i = 0; i < N; ++i) {
        int l = labels[i * 2 + 1];
        unary_i(l, i) = p_energy;
    }
    crf_i.setUnaryEnergy(unary_i);

    float sigma_spatial = 0.2f;
    float sigma_color = 0.04f;

    MatrixXf gaussian(3, N);
    for (int i = 0; i < N; ++i) {
        float* p = coords.data<float>() + i * 3;
        gaussian(0, i) = p[0] / sigma_spatial;
        gaussian(1, i) = p[1] / sigma_spatial;
        gaussian(2, i) = p[2] / sigma_spatial;
    }
    crf_s.addPairwiseEnergy(gaussian, new PottsCompatibility(3.0));
    crf_i.addPairwiseEnergy(gaussian, new PottsCompatibility(3.0));

    MatrixXf appearance(6, N);
    for (int i = 0; i < N; ++i) {
        float* p = coords.data<float>() + i * 3;
        float* c = points.data<float>() + i * 9 + 3;
        appearance(0, i) = p[0] / sigma_spatial;
        appearance(1, i) = p[1] / sigma_spatial;
        appearance(2, i) = p[2] / sigma_spatial;
        appearance(3, i) = c[0] / sigma_color;
        appearance(4, i) = c[1] / sigma_color;
        appearance(5, i) = c[2] / sigma_color;
    }
    crf_s.addPairwiseEnergy(appearance, new PottsCompatibility(10.0));
    crf_i.addPairwiseEnergy(appearance, new PottsCompatibility(10.0));

    MatrixXf Q_i = crf_i.startInference(), t1, t2;
    MatrixXf Q_s = crf_s.startInference(), t3, t4;
    VectorXs instances, semantics;
    for(int it = 0; it < 5; ++it) {
        crf_i.stepInference(Q_i, t1, t2);
        instances = crf_i.currentMap(Q_i);
        crf_s.addHigherOrderEnergy(instances, 20.0);
        crf_s.stepInference(Q_s, t3, t4);
        semantics = crf_s.currentMap(Q_s);
    }

    for (int i = 0; i < N; ++i) {
        labels[i * 2 + 0] = semantics[i];
        labels[i * 2 + 1] = instances[i];
    }

    cnpy::npz_save(argv[1], "pred", pred.data<int64_t>(), pred.shape, "w");
    return 0;
}
