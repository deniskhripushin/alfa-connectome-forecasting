#include <stdint.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

typedef struct {
    const float *mean;
    const float *std;
    float *engineered;
    double *ema;
    float *ring;
} History;

static float clip(float x) {
    if (x < -10.0f) return -10.0f;
    if (x > 10.0f) return 10.0f;
    return x;
}

API int history_version(void) {
    return 1;
}

API int history_step(History *state, const float *input, int step) {
    for (int i = 0; i < 112; ++i) {
        union { float f; uint32_t u; } value;
        value.f = input[i];
        if ((value.u & 0x7f800000u) == 0x7f800000u) return 0;
    }
    const int previous = ((step + 19) % 20) * 112;
    const int current = (step % 20) * 112;
    const double alpha0 = 2.0 / 21.0, alpha1 = 2.0 / 101.0;
    const double beta0 = 1.0 - alpha0, beta1 = 1.0 - alpha1;
    for (int i = 0; i < 112; ++i) {
        const float centered = input[i] - state->mean[i];
        const float z = clip(centered / state->std[i]);
        const float d1 = step >= 1 ? z - state->ring[previous + i] : 0.0f;
        const float d20 = step >= 20 ? z - state->ring[current + i] : 0.0f;
        /* Match the float32 initialization, then separate float64 EMA operations. */
        const float initial0 = (float)beta0 * z;
        const float initial1 = (float)beta1 * z;
        const double old0 = step == 0 ? (double)initial0 : state->ema[i] * beta0;
        const double old1 = step == 0 ? (double)initial1 : state->ema[112 + i] * beta1;
        const double next0 = old0 + alpha0 * (double)z;
        const double next1 = old1 + alpha1 * (double)z;
        state->ema[i] = next0;
        state->ema[112 + i] = next1;
        state->engineered[i] = z;
        state->engineered[112 + i] = clip(d1);
        state->engineered[224 + i] = clip(d20);
        state->engineered[336 + i] = clip((float)((double)z - next0));
        state->engineered[448 + i] = clip((float)((double)z - next1));
        state->ring[current + i] = z;
    }
    return 1;
}
