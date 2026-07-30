#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reference_canonical_core.h"

namespace {

int hex_nibble(const char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    throw std::invalid_argument("hex input must be lowercase");
}

std::vector<std::uint8_t> decode_hex(const std::string& encoded) {
    if ((encoded.size() % 2U) != 0U) {
        throw std::invalid_argument("hex input length must be even");
    }
    std::vector<std::uint8_t> output;
    output.reserve(encoded.size() / 2U);
    for (std::size_t index = 0; index < encoded.size(); index += 2U) {
        output.push_back(static_cast<std::uint8_t>(
            (hex_nibble(encoded[index]) << 4)
            | hex_nibble(encoded[index + 1U])));
    }
    return output;
}

std::string sha256(const std::vector<std::uint8_t>& message) {
    nf_u8 digest[32];
    if (!nf_reference_sha256(
            message.empty() ? nullptr : message.data(),
            static_cast<nf_u64>(message.size()),
            digest)) {
        throw std::runtime_error("canonical core rejected hash input");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const nf_u8 value : digest) {
        output << std::setw(2) << static_cast<unsigned int>(value);
    }
    return output.str();
}

bool flag(const std::string& value) {
    if (value == "1") {
        return true;
    }
    if (value == "0") {
        return false;
    }
    throw std::invalid_argument("state flags must be 0 or 1");
}

}  // namespace

int main(const int argc, const char* const argv[]) {
    try {
        if (argc == 3 && std::string(argv[1]) == "hash") {
            std::cout << sha256(decode_hex(argv[2])) << '\n';
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "state") {
            const bool numeric_eligible = flag(argv[2]);
            const bool all_promoted = flag(argv[3]);
            const bool any_research_override = flag(argv[4]);
            std::cout
                << (
                    nf_reference_staging_authorized(
                        numeric_eligible ? 1U : 0U,
                        all_promoted ? 1U : 0U,
                        any_research_override ? 1U : 0U)
                    ? "authorized-for-staging"
                    : "identity-fallback")
                << '\n';
            return 0;
        }
        std::cerr
            << "usage: reference_product_chain_conformance "
               "hash <canonical_hex> | "
               "state <numeric> <promoted> <research>\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
