/*
sha512.cuh - Device-side SHA512 implementation for CUDA
Based on FIPS 180-4 specification
*/

#ifndef SHA512_CUH
#define SHA512_CUH

#include <stdint.h>
#include <cuda_runtime.h>

struct SHA512_CTX {
    uint64_t state[8];
    uint64_t count[2];
    uint8_t buffer[128];
};

// SHA512 constants
__constant__ uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// Rotate right
__device__ __forceinline__ uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// SHA512 functions
__device__ __forceinline__ uint64_t Ch(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ ((~x) & z);
}

__device__ __forceinline__ uint64_t Maj(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint64_t Sigma0(uint64_t x) {
    return rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39);
}

__device__ __forceinline__ uint64_t Sigma1(uint64_t x) {
    return rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41);
}

__device__ __forceinline__ uint64_t sigma0(uint64_t x) {
    return rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7);
}

__device__ __forceinline__ uint64_t sigma1(uint64_t x) {
    return rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6);
}

__device__ void sha512_init(SHA512_CTX *ctx) {
    ctx->count[0] = ctx->count[1] = 0;
    
    // Initial hash values
    ctx->state[0] = 0x6a09e667f3bcc908ULL;
    ctx->state[1] = 0xbb67ae8584caa73bULL;
    ctx->state[2] = 0x3c6ef372fe94f82bULL;
    ctx->state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->state[4] = 0x510e527fade682d1ULL;
    ctx->state[5] = 0x9b05688c2b3e6c1fULL;
    ctx->state[6] = 0x1f83d9abfb41bd6bULL;
    ctx->state[7] = 0x5be0cd19137e2179ULL;
}

__device__ void sha512_transform(SHA512_CTX *ctx, const uint8_t data[128]) {
    uint64_t W[80];
    uint64_t a, b, c, d, e, f, g, h;
    uint64_t T1, T2;
    
    // Prepare message schedule
    for (int t = 0; t < 16; t++) {
        W[t] = ((uint64_t)data[t*8 + 0] << 56) | ((uint64_t)data[t*8 + 1] << 48) |
               ((uint64_t)data[t*8 + 2] << 40) | ((uint64_t)data[t*8 + 3] << 32) |
               ((uint64_t)data[t*8 + 4] << 24) | ((uint64_t)data[t*8 + 5] << 16) |
               ((uint64_t)data[t*8 + 6] << 8)  | ((uint64_t)data[t*8 + 7] << 0);
    }
    
    for (int t = 16; t < 80; t++) {
        W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16];
    }
    
    // Initialize working variables
    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];
    
    // Main loop
    for (int t = 0; t < 80; t++) {
        T1 = h + Sigma1(e) + Ch(e, f, g) + K[t] + W[t];
        T2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    
    // Update hash values
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

__device__ void sha512_update(SHA512_CTX *ctx, const uint8_t *data, uint64_t len) {
    uint64_t index = (ctx->count[0] >> 3) & 0x7F;
    
    ctx->count[0] += len << 3;
    if (ctx->count[0] < (len << 3)) {
        ctx->count[1]++;
    }
    ctx->count[1] += len >> 61;
    
    uint64_t partLen = 128 - index;
    
    if (len >= partLen) {
        for (uint64_t i = 0; i < partLen; i++) {
            ctx->buffer[index + i] = data[i];
        }
        sha512_transform(ctx, ctx->buffer);
        
        for (uint64_t i = partLen; i + 128 <= len; i += 128) {
            sha512_transform(ctx, &data[i]);
        }
        
        index = 0;
        len -= partLen;
        data += partLen;
        len = len % 128;
    }
    
    for (uint64_t i = 0; i < len; i++) {
        ctx->buffer[index + i] = data[i];
    }
}

__device__ void sha512_final(SHA512_CTX *ctx, uint8_t digest[64]) {
    uint8_t bits[16];
    uint64_t index, padLen;
    
    // Save bit count
    for (int i = 0; i < 8; i++) {
        bits[i] = (ctx->count[1] >> (56 - i * 8)) & 0xFF;
        bits[i + 8] = (ctx->count[0] >> (56 - i * 8)) & 0xFF;
    }
    
    // Pad to 896 mod 1024 bits
    index = (ctx->count[0] >> 3) & 0x7F;
    padLen = (index < 112) ? (112 - index) : (240 - index);
    
    uint8_t padding[128];
    padding[0] = 0x80;
    for (uint64_t i = 1; i < padLen; i++) {
        padding[i] = 0x00;
    }
    
    sha512_update(ctx, padding, padLen);
    sha512_update(ctx, bits, 16);
    
    // Store digest
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            digest[i * 8 + j] = (ctx->state[i] >> (56 - j * 8)) & 0xFF;
        }
    }
}

// Convenience function for one-shot hashing
__device__ void sha512_hash(const uint8_t *input, uint64_t len, uint8_t output[64]) {
    SHA512_CTX ctx;
    sha512_init(&ctx);
    sha512_update(&ctx, input, len);
    sha512_final(&ctx, output);
}

#endif // SHA512_CUH