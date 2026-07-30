#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "reference_canonical_core.h"
#include "reference_srgb_icc_profile_v1.h"

namespace {

using Profile = std::array<std::uint8_t, NF_SRGB_ICC_PROFILE_V1_SIZE>;

Profile profile() {
    Profile output{};
    if (nf_srgb_icc_profile_copy_v1(output.data(), output.size()) != 1) {
        throw std::runtime_error("profile copy failed");
    }
    return output;
}

std::string hash(const Profile& value) {
    nf_u8 digest[32];
    if (!nf_reference_sha256(value.data(), value.size(), digest)) {
        throw std::runtime_error("profile hash failed");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const nf_u8 byte : digest) {
        output << std::setw(2) << static_cast<unsigned int>(byte);
    }
    return output.str();
}

std::string hex4(const Profile& value, const std::size_t offset) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < 4U; ++index) {
        output
            << std::setw(2)
            << static_cast<unsigned int>(value[offset + index]);
    }
    return output.str();
}

bool negative_contract() {
    Profile output{};
    output.fill(0xa5U);
    const Profile original = output;
    if (
        nf_srgb_icc_profile_copy_v1(nullptr, output.size()) != 0
        || output != original
        || nf_srgb_icc_profile_copy_v1(
            output.data(),
            output.size() - 1U) != 0
        || output != original
    ) {
        return false;
    }
    return nf_srgb_icc_profile_copy_v1(
        output.data(),
        output.size()) == 1;
}

}  // namespace

int main(const int argc, const char* const argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "usage: reference_srgb_icc_profile_conformance "
                         "hash | header | negative\n";
            return 2;
        }
        const std::string command(argv[1]);
        const Profile value = profile();
        if (command == "hash") {
            std::cout << hash(value) << '\n';
            return 0;
        }
        if (command == "header") {
            std::cout
                << value.size() << ' '
                << hex4(value, 12U) << ' '
                << hex4(value, 16U) << ' '
                << hex4(value, 20U) << ' '
                << hex4(value, 36U) << '\n';
            return 0;
        }
        if (command == "negative") {
            const bool passed = negative_contract();
            std::cout << (passed ? "pass" : "fail") << '\n';
            return passed ? 0 : 1;
        }
        std::cerr << "unknown command\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
