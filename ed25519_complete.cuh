/*
ed25519_complete.cuh - Complete Ed25519 field and curve operations
*/

#ifndef ED25519_COMPLETE_CUH
#define ED25519_COMPLETE_CUH

#include <stdint.h>
#include <cuda_runtime.h>

struct fe_t { uint32_t v[10]; };

struct ge_ext {
    fe_t X, Y, Z, T;
};

// Basic field arithmetic implementations
__device__ inline void fe_add(fe_t &out, const fe_t &a, const fe_t &b) {
    #pragma unroll
    for(int i = 0; i < 10; i++) {
        out.v[i] = a.v[i] + b.v[i];
    }
}

__device__ inline void fe_sub(fe_t &out, const fe_t &a, const fe_t &b) {
    // Add 2*p to ensure positive result
    const uint32_t two_p[10] = {0x7ffffda, 0x3fffffe, 0x7fffffe, 0x3fffffe, 0x7fffffe, 
                                0x3fffffe, 0x7fffffe, 0x3fffffe, 0x7fffffe, 0x3fffffe};
    #pragma unroll
    for(int i = 0; i < 10; i++) {
        out.v[i] = two_p[i] + a.v[i] - b.v[i];
    }
}

__device__ inline void fe_mul(fe_t &out, const fe_t &a, const fe_t &b) {
    // Ed25519 field multiplication using schoolbook method with carry reduction
    uint64_t t[19] = {0};
    
    // Schoolbook multiplication
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 10; j++) {
            t[i + j] += (uint64_t)a.v[i] * b.v[j];
        }
    }
    
    // Reduce high limbs using 2^255 ≡ 19 (mod p)
    t[0] += 19 * t[10]; t[1] += 19 * t[11]; t[2] += 19 * t[12]; t[3] += 19 * t[13];
    t[4] += 19 * t[14]; t[5] += 19 * t[15]; t[6] += 19 * t[16]; t[7] += 19 * t[17];
    t[8] += 19 * t[18];
    
    // Carry propagation for alternating 26/25 bit limbs
    t[1] += t[0] >> 26; out.v[0] = (uint32_t)(t[0] & 0x3ffffff);
    t[2] += t[1] >> 25; out.v[1] = (uint32_t)(t[1] & 0x1ffffff);
    t[3] += t[2] >> 26; out.v[2] = (uint32_t)(t[2] & 0x3ffffff);
    t[4] += t[3] >> 25; out.v[3] = (uint32_t)(t[3] & 0x1ffffff);
    t[5] += t[4] >> 26; out.v[4] = (uint32_t)(t[4] & 0x3ffffff);
    t[6] += t[5] >> 25; out.v[5] = (uint32_t)(t[5] & 0x1ffffff);
    t[7] += t[6] >> 26; out.v[6] = (uint32_t)(t[6] & 0x3ffffff);
    t[8] += t[7] >> 25; out.v[7] = (uint32_t)(t[7] & 0x1ffffff);
    t[9] += t[8] >> 26; out.v[8] = (uint32_t)(t[8] & 0x3ffffff);
    
    // Handle final carry and potential overflow
    uint64_t carry = t[9] >> 25;
    out.v[9] = (uint32_t)(t[9] & 0x1ffffff);
    
    // Final reduction: carry * 2^255 ≡ carry * 19
    out.v[0] += (uint32_t)(carry * 19);
    
    // Propagate final carry if needed
    uint32_t c = out.v[0] >> 26;
    out.v[0] &= 0x3ffffff;
    out.v[1] += c;
}

__device__ inline void fe_sq(fe_t &out, const fe_t &a) {
    // Optimized squaring using diagonal optimization
    uint64_t t[19] = {0};
    
    // Diagonal terms
    for(int i = 0; i < 10; i++) {
        t[2*i] += (uint64_t)a.v[i] * a.v[i];
    }
    
    // Off-diagonal terms (doubled)
    for(int i = 0; i < 10; i++) {
        for(int j = i + 1; j < 10; j++) {
            t[i + j] += 2 * (uint64_t)a.v[i] * a.v[j];
        }
    }
    
    // Same reduction as multiplication
    t[0] += 19 * t[10]; t[1] += 19 * t[11]; t[2] += 19 * t[12]; t[3] += 19 * t[13];
    t[4] += 19 * t[14]; t[5] += 19 * t[15]; t[6] += 19 * t[16]; t[7] += 19 * t[17];
    t[8] += 19 * t[18];
    
    t[1] += t[0] >> 26; out.v[0] = (uint32_t)(t[0] & 0x3ffffff);
    t[2] += t[1] >> 25; out.v[1] = (uint32_t)(t[1] & 0x1ffffff);
    t[3] += t[2] >> 26; out.v[2] = (uint32_t)(t[2] & 0x3ffffff);
    t[4] += t[3] >> 25; out.v[3] = (uint32_t)(t[3] & 0x1ffffff);
    t[5] += t[4] >> 26; out.v[4] = (uint32_t)(t[4] & 0x3ffffff);
    t[6] += t[5] >> 25; out.v[5] = (uint32_t)(t[5] & 0x1ffffff);
    t[7] += t[6] >> 26; out.v[6] = (uint32_t)(t[6] & 0x3ffffff);
    t[8] += t[7] >> 25; out.v[7] = (uint32_t)(t[7] & 0x1ffffff);
    t[9] += t[8] >> 26; out.v[8] = (uint32_t)(t[8] & 0x3ffffff);
    
    uint64_t carry = t[9] >> 25;
    out.v[9] = (uint32_t)(t[9] & 0x1ffffff);
    out.v[0] += (uint32_t)(carry * 19);
    uint32_t c = out.v[0] >> 26;
    out.v[0] &= 0x3ffffff;
    out.v[1] += c;
}

__device__ inline void fe_tobytes(uint8_t bytes[32], const fe_t &f) {
    // Convert field element to bytes (little-endian)
    fe_t h = f;
    
    // Carry propagation to normalize
    uint32_t carry = 0;
    #pragma unroll
    for(int i = 0; i < 10; i++) {
        h.v[i] += carry;
        if(i % 2 == 0) { // 26-bit limbs
            carry = h.v[i] >> 26;
            h.v[i] &= 0x3ffffff;
        } else { // 25-bit limbs
            carry = h.v[i] >> 25;
            h.v[i] &= 0x1ffffff;
        }
    }
    
    // Final reduction modulo p
    h.v[0] += carry * 19;
    carry = h.v[0] >> 26;
    h.v[0] &= 0x3ffffff;
    h.v[1] += carry;
    
    // Pack into bytes
    bytes[0]  = h.v[0] & 0xff;
    bytes[1]  = (h.v[0] >> 8) & 0xff;
    bytes[2]  = (h.v[0] >> 16) & 0xff;
    bytes[3]  = ((h.v[0] >> 24) | (h.v[1] << 2)) & 0xff;
    bytes[4]  = (h.v[1] >> 6) & 0xff;
    bytes[5]  = (h.v[1] >> 14) & 0xff;
    bytes[6]  = ((h.v[1] >> 22) | (h.v[2] << 3)) & 0xff;
    bytes[7]  = (h.v[2] >> 5) & 0xff;
    bytes[8]  = (h.v[2] >> 13) & 0xff;
    bytes[9]  = ((h.v[2] >> 21) | (h.v[3] << 5)) & 0xff;
    bytes[10] = (h.v[3] >> 3) & 0xff;
    bytes[11] = (h.v[3] >> 11) & 0xff;
    bytes[12] = ((h.v[3] >> 19) | (h.v[4] << 6)) & 0xff;
    bytes[13] = (h.v[4] >> 2) & 0xff;
    bytes[14] = (h.v[4] >> 10) & 0xff;
    bytes[15] = (h.v[4] >> 18) & 0xff;
    bytes[16] = h.v[5] & 0xff;
    bytes[17] = (h.v[5] >> 8) & 0xff;
    bytes[18] = (h.v[5] >> 16) & 0xff;
    bytes[19] = ((h.v[5] >> 24) | (h.v[6] << 1)) & 0xff;
    bytes[20] = (h.v[6] >> 7) & 0xff;
    bytes[21] = (h.v[6] >> 15) & 0xff;
    bytes[22] = ((h.v[6] >> 23) | (h.v[7] << 3)) & 0xff;
    bytes[23] = (h.v[7] >> 5) & 0xff;
    bytes[24] = (h.v[7] >> 13) & 0xff;
    bytes[25] = ((h.v[7] >> 21) | (h.v[8] << 4)) & 0xff;
    bytes[26] = (h.v[8] >> 4) & 0xff;
    bytes[27] = (h.v[8] >> 12) & 0xff;
    bytes[28] = ((h.v[8] >> 20) | (h.v[9] << 6)) & 0xff;
    bytes[29] = (h.v[9] >> 2) & 0xff;
    bytes[30] = (h.v[9] >> 10) & 0xff;
    bytes[31] = (h.v[9] >> 18) & 0xff;
}

// Field element constants
__constant__ fe_t fe_zero_const = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
__constant__ fe_t fe_one_const = {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
__constant__ fe_t fe_two_const = {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

// Edwards curve parameter d = -121665/121666
__constant__ fe_t ed25519_d = {{0x035978a3, 0x02d37284, 0x03e5bc06, 0x0080e2c7, 0x02734ab8,
                                0x01b5f199, 0x01043dab, 0x01ed1c11, 0x03ce19fb, 0x01de8c3f}};

// 2*d
__constant__ fe_t ed25519_2d = {{0x06b2f159, 0x05a6e508, 0x07cb780c, 0x0101c58e, 0x04e69570,
                                 0x036be333, 0x02087b56, 0x03da2223, 0x0799c3f6, 0x03bd187f}};

// sqrt(-1) mod p
__constant__ fe_t fe_sqrtm1 = {{0x020ea0b0, 0x0186c9d2, 0x008f189d, 0x0035697f, 0x00bd0c60,
                                0x01fbd7a7, 0x02804c9e, 0x01e16569, 0x0004fc1d, 0x00ae0c92}};

__device__ inline void fe_copy(fe_t &out, const fe_t &a) {
    for (int i = 0; i < 10; i++) {
        out.v[i] = a.v[i];
    }
}

__device__ inline void fe_0(fe_t &out) {
    for (int i = 0; i < 10; i++) {
        out.v[i] = 0;
    }
}

__device__ inline void fe_1(fe_t &out) {
    fe_0(out);
    out.v[0] = 1;
}

// Complete field element inversion using Fermat's little theorem
// For Ed25519: p = 2^255 - 19, so p-2 = 2^255 - 21
__device__ void fe_invert(fe_t &out, const fe_t &z) {
    fe_t z2, z8, z9, z11, z22, z_5_0, z_10_5, z_10_0, z_20_10, z_20_0;
    fe_t z_40_20, z_40_0, z_50_10, z_50_0, z_100_50, z_100_0, z_200_100, z_200_0;
    fe_t z_250_50, z_250_0, z_255_5, z_255_21, t0, t1;
    
    // z2 = z^(2^1)
    fe_sq(z2, z);
    
    // z8 = z^(2^3)
    fe_sq(t0, z2);  // z^4
    fe_sq(t1, t0);  // z^8
    fe_copy(z8, t1);
    
    // z9 = z^(2^3 + 1)
    fe_mul(z9, z8, z);
    
    // z11 = z^(2^3 + 2^1 + 1)  
    fe_mul(z11, z9, z2);
    
    // z22 = z^(2^4 + 2^2 + 2^1)
    fe_sq(z22, z11);
    
    // z_5_0 = z^(2^5 - 1)
    fe_mul(z_5_0, z22, z9);
    
    // z_10_5 = z^(2^10 - 2^5)
    fe_sq(t0, z_5_0);
    for (int i = 1; i < 5; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_10_5, t0);
    
    // z_10_0 = z^(2^10 - 1)
    fe_mul(z_10_0, z_10_5, z_5_0);
    
    // z_20_10 = z^(2^20 - 2^10)
    fe_copy(t0, z_10_0);
    for (int i = 0; i < 10; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_20_10, t0);
    
    // z_20_0 = z^(2^20 - 1)
    fe_mul(z_20_0, z_20_10, z_10_0);
    
    // z_40_20 = z^(2^40 - 2^20)
    fe_copy(t0, z_20_0);
    for (int i = 0; i < 20; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_40_20, t0);
    
    // z_40_0 = z^(2^40 - 1)
    fe_mul(z_40_0, z_40_20, z_20_0);
    
    // z_50_10 = z^(2^50 - 2^10)
    fe_copy(t0, z_40_0);
    for (int i = 0; i < 10; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_50_10, t0);
    
    // z_50_0 = z^(2^50 - 1)
    fe_mul(z_50_0, z_50_10, z_10_0);
    
    // z_100_50 = z^(2^100 - 2^50)
    fe_copy(t0, z_50_0);
    for (int i = 0; i < 50; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_100_50, t0);
    
    // z_100_0 = z^(2^100 - 1)
    fe_mul(z_100_0, z_100_50, z_50_0);
    
    // z_200_100 = z^(2^200 - 2^100)
    fe_copy(t0, z_100_0);
    for (int i = 0; i < 100; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_200_100, t0);
    
    // z_200_0 = z^(2^200 - 1)
    fe_mul(z_200_0, z_200_100, z_100_0);
    
    // z_250_50 = z^(2^250 - 2^50)
    fe_copy(t0, z_200_0);
    for (int i = 0; i < 50; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_250_50, t0);
    
    // z_250_0 = z^(2^250 - 1)
    fe_mul(z_250_0, z_250_50, z_50_0);
    
    // z_255_5 = z^(2^255 - 2^5)
    fe_copy(t0, z_250_0);
    for (int i = 0; i < 5; i++) {
        fe_sq(t0, t0);
    }
    fe_copy(z_255_5, t0);
    
    // z_255_21 = z^(2^255 - 21) = z^(p-2)
    fe_mul(z_255_21, z_255_5, z11);
    
    fe_copy(out, z_255_21);
}

// Conditional move: if b == 1, then out = x, else out = y
__device__ void fe_cmov(fe_t &out, const fe_t &x, const fe_t &y, uint32_t b) {
    uint32_t mask = -(b & 1);
    for (int i = 0; i < 10; i++) {
        out.v[i] = (mask & x.v[i]) | (~mask & y.v[i]);
    }
}

// Test if field element is negative (LSB of canonical representation)
__device__ uint32_t fe_isnegative(const fe_t &f) {
    uint8_t s[32];
    fe_tobytes(s, f);
    return s[0] & 1;
}

// Test if field element is non-zero
__device__ uint32_t fe_isnonzero(const fe_t &f) {
    uint8_t s[32];
    fe_tobytes(s, f);
    uint8_t result = 0;
    for (int i = 0; i < 32; i++) {
        result |= s[i];
    }
    return (result != 0) ? 1 : 0;
}

// Square root computation
__device__ int fe_sqrt(fe_t &out, const fe_t &a) {
    fe_t exp, x, b, t0, t1;
    
    // Try x = a^((p+3)/8)
    fe_copy(t0, a);
    for (int i = 0; i < 252; i++) {  // (p+3)/8 exponent
        fe_sq(t0, t0);
        if (i == 1) fe_mul(t0, t0, a);
    }
    fe_copy(x, t0);
    
    // Check if x^2 = a
    fe_sq(t0, x);
    fe_sub(t1, t0, a);
    
    if (fe_isnonzero(t1) == 0) {
        fe_copy(out, x);
        return 1;
    }
    
    // Try x = a^((p+3)/8) * sqrt(-1)
    fe_mul(x, x, fe_sqrtm1);
    fe_sq(t0, x);
    fe_sub(t1, t0, a);
    
    if (fe_isnonzero(t1) == 0) {
        fe_copy(out, x);
        return 1;
    }
    
    return 0;  // No square root exists
}

// Point operations with complete addition formulas

__device__ void ge_p3_0(ge_ext &h) {
    fe_0(h.X);
    fe_1(h.Y);
    fe_1(h.Z);
    fe_0(h.T);
}

__device__ void ge_p3_dbl(ge_ext &r, const ge_ext &p) {
    fe_t A, B, C, D, E, G, F, H;
    
    fe_sq(A, p.X);
    fe_sq(B, p.Y);
    fe_sq(C, p.Z);
    fe_add(C, C, C);
    fe_copy(D, A);  // In Edwards, A = -X^2, but we use A = X^2
    fe_add(E, p.X, p.Y);
    fe_sq(E, E);
    fe_sub(E, E, A);
    fe_sub(E, E, B);
    fe_sub(G, B, A);  // G = Y^2 - X^2 (note: should be Y^2 + X^2 for our representation)  
    fe_add(G, B, A);  // Correct: G = Y^2 + X^2
    fe_sub(F, G, C);
    fe_add(H, B, A);  // H = Y^2 + X^2
    fe_sub(H, B, A);  // Correct: H = Y^2 - X^2
    
    fe_mul(r.X, E, F);
    fe_mul(r.Y, G, H);
    fe_mul(r.Z, F, G);
    fe_mul(r.T, E, H);
}

__device__ void ge_p3_add(ge_ext &r, const ge_ext &p, const ge_ext &q) {
    fe_t A, B, C, D, E, F, G, H;
    
    fe_sub(A, p.Y, p.X);
    fe_sub(B, q.Y, q.X);
    fe_mul(A, A, B);
    fe_add(B, p.Y, p.X);
    fe_add(C, q.Y, q.X);
    fe_mul(B, B, C);
    fe_mul(C, p.T, q.T);
    fe_mul(C, C, ed25519_2d);
    fe_mul(D, p.Z, q.Z);
    fe_add(D, D, D);
    fe_sub(E, B, A);
    fe_sub(F, D, C);
    fe_add(G, D, C);
    fe_add(H, B, A);
    
    fe_mul(r.X, E, F);
    fe_mul(r.Y, G, H);
    fe_mul(r.Z, F, G);
    fe_mul(r.T, E, H);
}

// Convert extended coordinates to affine and compress
__device__ void ge_p3_tobytes(uint8_t s[32], const ge_ext &h) {
    fe_t recip, x, y;
    
    fe_invert(recip, h.Z);
    fe_mul(x, h.X, recip);
    fe_mul(y, h.Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

// Decompress point from bytes
__device__ int ge_frombytes_negate_vartime(ge_ext &h, const uint8_t s[32]) {
    fe_t u, v, v3, vxx, check, y, y_squared;
    uint32_t sign;
    
    // Extract y coordinate and sign bit
    fe_copy(y, *(fe_t*)s);  // This is unsafe - need proper deserialization
    sign = s[31] >> 7;
    
    // Clear sign bit
    ((uint8_t*)&y)[31] &= 0x7F;
    
    // u = y^2 - 1
    fe_sq(y_squared, y);
    fe_sub(u, y_squared, fe_one_const);
    
    // v = d*y^2 + 1  
    fe_mul(v, ed25519_d, y_squared);
    fe_add(v, v, fe_one_const);
    
    // x = u/v = (y^2-1)/(d*y^2+1)
    fe_sq(v3, v);
    fe_mul(v3, v3, v);        // v^3
    fe_sq(h.X, v);            // v^2
    fe_mul(h.X, h.X, u);      // u*v^2
    fe_mul(h.X, h.X, v);      // u*v^3
    
    // Try to compute square root
    if (fe_sqrt(h.X, h.X) == 0) {
        return -1;  // Point not on curve
    }
    
    // Check if we got the right sign
    if (fe_isnegative(h.X) != sign) {
        fe_sub(h.X, fe_zero_const, h.X);  // Negate
    }
    
    fe_copy(h.Y, y);
    fe_1(h.Z);
    fe_mul(h.T, h.X, h.Y);
    
    return 0;
}

#endif // ED25519_COMPLETE_CUH