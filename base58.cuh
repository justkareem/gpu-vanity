/*
base58.cuh - Proper Base58 encoding for CUDA devices
*/

#ifndef BASE58_CUH
#define BASE58_CUH

#include <stdint.h>
#include <cuda_runtime.h>

// Base58 alphabet (Bitcoin/Solana variant)
__constant__ char base58_alphabet[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Big integer structure for Base58 calculations
struct BigInt {
    uint32_t limbs[9];  // 32-bit limbs, enough for 256-bit numbers
    int length;
};

__device__ void bigint_from_bytes(BigInt &num, const uint8_t bytes[32]) {
    // Initialize
    for (int i = 0; i < 9; i++) {
        num.limbs[i] = 0;
    }
    num.length = 8;  // We need 8 limbs for 256 bits
    
    // Load bytes in big-endian order
    for (int i = 0; i < 32; i++) {
        int limb_idx = i / 4;
        int byte_idx = i % 4;
        num.limbs[limb_idx] |= ((uint32_t)bytes[i]) << (24 - 8 * byte_idx);
    }
    
    // Find actual length (remove leading zeros)
    while (num.length > 1 && num.limbs[num.length - 1] == 0) {
        num.length--;
    }
}

__device__ bool bigint_is_zero(const BigInt &num) {
    for (int i = 0; i < num.length; i++) {
        if (num.limbs[i] != 0) return false;
    }
    return true;
}

// Divide BigInt by 58 in place, return remainder
__device__ uint32_t bigint_div58(BigInt &num) {
    uint64_t carry = 0;
    
    // Divide from most significant to least significant limb
    for (int i = num.length - 1; i >= 0; i--) {
        uint64_t temp = carry * (1ULL << 32) + num.limbs[i];
        num.limbs[i] = (uint32_t)(temp / 58);
        carry = temp % 58;
    }
    
    // Adjust length if leading limb became zero
    while (num.length > 1 && num.limbs[num.length - 1] == 0) {
        num.length--;
    }
    
    return (uint32_t)carry;
}

// Full Base58 encoding
__device__ int base58_encode(char *output, const uint8_t input[32]) {
    BigInt num;
    bigint_from_bytes(num, input);
    
    // Count leading zero bytes in input
    int leading_zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (input[i] == 0) {
            leading_zeros++;
        } else {
            break;
        }
    }
    
    // Convert to base58
    char temp[45];  // Maximum possible length
    int digit_count = 0;
    
    while (!bigint_is_zero(num)) {
        uint32_t remainder = bigint_div58(num);
        temp[digit_count++] = base58_alphabet[remainder];
    }
    
    // Add leading '1's for zero bytes
    int total_length = leading_zeros + digit_count;
    
    // Output leading '1's
    for (int i = 0; i < leading_zeros; i++) {
        output[i] = '1';
    }
    
    // Reverse the digits (they come out in reverse order)
    for (int i = 0; i < digit_count; i++) {
        output[leading_zeros + i] = temp[digit_count - 1 - i];
    }
    
    output[total_length] = '\0';
    return total_length;
}

// Fast prefix matching (compute only first N characters)
__device__ bool base58_check_prefix(const uint8_t input[32], const char *prefix, int prefix_len) {
    if (prefix_len <= 0 || prefix_len > 10) return false;
    
    BigInt num;
    bigint_from_bytes(num, input);
    
    // Count leading zero bytes
    int leading_zeros = 0;
    for (int i = 0; i < 32; i++) {
        if (input[i] == 0) {
            leading_zeros++;
        } else {
            break;
        }
    }
    
    // Generate exactly prefix_len characters
    char result[11];  // Max 10 chars + null terminator
    int pos = 0;
    
    // Add leading '1's if needed
    while (pos < prefix_len && pos < leading_zeros) {
        result[pos] = '1';
        pos++;
    }
    
    // Generate remaining digits
    while (pos < prefix_len && !bigint_is_zero(num)) {
        uint32_t remainder = bigint_div58(num);
        result[pos] = base58_alphabet[remainder];
        pos++;
    }
    
    // If we couldn't generate enough digits (number too small), pad with '1'
    while (pos < prefix_len) {
        result[pos] = '1';
        pos++;
    }
    
    result[prefix_len] = '\0';
    
    // Reverse the non-leading-zero part
    if (prefix_len > leading_zeros) {
        int start = leading_zeros;
        int end = prefix_len - 1;
        while (start < end) {
            char temp = result[start];
            result[start] = result[end];
            result[end] = temp;
            start++;
            end--;
        }
    }
    
    // Compare with target prefix
    for (int i = 0; i < prefix_len; i++) {
        if (result[i] != prefix[i]) {
            return false;
        }
    }
    
    return true;
}

// Alternative implementation using lookup table for small prefixes
__device__ bool base58_quick_prefix_check(const uint8_t input[32], const char *prefix, int prefix_len) {
    // For very short prefixes, we can use modular arithmetic shortcuts
    if (prefix_len == 1) {
        // Find what remainder mod 58 would give us this character
        int target_digit = -1;
        for (int i = 0; i < 58; i++) {
            if (base58_alphabet[i] == prefix[0]) {
                target_digit = i;
                break;
            }
        }
        if (target_digit == -1) return false;
        
        // Quick check: does the input mod 58 match?
        BigInt num;
        bigint_from_bytes(num, input);
        
        // Handle leading zeros
        if (input[0] == 0 && prefix[0] == '1') return true;
        if (input[0] == 0 && prefix[0] != '1') return false;
        if (input[0] != 0 && prefix[0] == '1') return false;
        
        uint32_t remainder = bigint_div58(num);
        return remainder == target_digit;
    }
    
    // For longer prefixes, fall back to full method
    return base58_check_prefix(input, prefix, prefix_len);
}

#endif // BASE58_CUH