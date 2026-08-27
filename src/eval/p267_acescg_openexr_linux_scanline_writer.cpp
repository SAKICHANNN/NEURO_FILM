// Private P267 WSL2/Linux ordered-scanline ACEScg OpenEXR research writer.

#include <ImfChannelList.h>
#include <ImfChromaticitiesAttribute.h>
#include <ImfCompression.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <ImfVecAttribute.h>

#include <ImathVec.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path output;
    int width = 0;
    int height = 0;
    int row_block = 0;
    std::uint32_t period = 0;
    bool inject_before_publish = false;
};

int parse_positive_int(const char* text, const char* name) {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed, 10);
    if (consumed != std::string(text).size() || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<int>(value);
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--inject-before-publish") {
            args.inject_before_publish = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing option value: " + option);
        }
        const char* value = argv[++i];
        if (option == "--output") {
            args.output = std::filesystem::u8path(value);
        } else if (option == "--width") {
            args.width = parse_positive_int(value, "width");
        } else if (option == "--height") {
            args.height = parse_positive_int(value, "height");
        } else if (option == "--row-block") {
            args.row_block = parse_positive_int(value, "row-block");
        } else if (option == "--period") {
            args.period = static_cast<std::uint32_t>(parse_positive_int(value, "period"));
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (args.output.empty() || args.width <= 0 || args.height <= 0 ||
        args.row_block <= 0 || args.period < 2) {
        throw std::invalid_argument("incomplete writer arguments");
    }
    if (args.row_block != 16) {
        throw std::invalid_argument("P267 row block must equal 16");
    }
    return args;
}

float sample_value(int channel, std::uint32_t x, std::uint32_t y, std::uint32_t period) {
    std::uint32_t code = 0;
    float scale = 0.0F;
    float offset = 0.0F;
    if (channel == 0) {
        code = (x + 3U * y) % period;
        scale = 4.0F / static_cast<float>(period - 1U);
        offset = -0.25F;
    } else if (channel == 1) {
        code = (5U * x + 7U * y) % period;
        scale = 2.0F / static_cast<float>(period - 1U);
    } else {
        code = (x ^ (13U * y)) % period;
        scale = 16.0F / static_cast<float>(period - 1U);
    }
    return static_cast<float>(code) * scale + offset;
}

void atomic_replace(
    const std::filesystem::path& source,
    const std::filesystem::path& target) {
    std::error_code error;
    std::filesystem::rename(source, target, error);
    if (error) {
        throw std::runtime_error("atomic publish failed: " + error.message());
    }
}

void write_file(const Arguments& args) {
    const auto parent = args.output.parent_path();
    if (parent.empty() || !std::filesystem::is_directory(parent)) {
        throw std::invalid_argument("output parent must already exist");
    }
    const std::filesystem::path temp_path(args.output.string() + ".p267.tmp");
    if (std::filesystem::exists(temp_path)) {
        throw std::runtime_error("temporary output already exists");
    }

    try {
        OPENEXR_IMF_NAMESPACE::Header header(args.width, args.height);
        header.compression() = OPENEXR_IMF_NAMESPACE::ZIP_COMPRESSION;
        header.channels().insert(
            "R", OPENEXR_IMF_NAMESPACE::Channel(OPENEXR_IMF_NAMESPACE::FLOAT));
        header.channels().insert(
            "G", OPENEXR_IMF_NAMESPACE::Channel(OPENEXR_IMF_NAMESPACE::FLOAT));
        header.channels().insert(
            "B", OPENEXR_IMF_NAMESPACE::Channel(OPENEXR_IMF_NAMESPACE::FLOAT));
        header.insert(
            "chromaticities",
            OPENEXR_IMF_NAMESPACE::ChromaticitiesAttribute(
                OPENEXR_IMF_NAMESPACE::Chromaticities(
                    IMATH_NAMESPACE::V2f(0.713F, 0.293F),
                    IMATH_NAMESPACE::V2f(0.165F, 0.830F),
                    IMATH_NAMESPACE::V2f(0.128F, 0.044F),
                    IMATH_NAMESPACE::V2f(0.32168F, 0.33767F))));
        header.insert(
            "adoptedNeutral",
            OPENEXR_IMF_NAMESPACE::V2fAttribute(
                IMATH_NAMESPACE::V2f(0.32168F, 0.33767F)));

        OPENEXR_IMF_NAMESPACE::OutputFile output(temp_path.string().c_str(), header);
        const std::size_t pixel_stride = 3U * sizeof(float);
        const std::size_t row_stride = static_cast<std::size_t>(args.width) * pixel_stride;
        std::vector<float> rows(
            static_cast<std::size_t>(args.row_block) *
            static_cast<std::size_t>(args.width) * 3U);

        for (int y0 = 0; y0 < args.height; y0 += args.row_block) {
            const int count = std::min(args.row_block, args.height - y0);
            for (int local_y = 0; local_y < count; ++local_y) {
                const std::uint32_t y = static_cast<std::uint32_t>(y0 + local_y);
                for (int x_int = 0; x_int < args.width; ++x_int) {
                    const std::uint32_t x = static_cast<std::uint32_t>(x_int);
                    const std::size_t base =
                        (static_cast<std::size_t>(local_y) *
                             static_cast<std::size_t>(args.width) +
                         static_cast<std::size_t>(x_int)) *
                        3U;
                    rows[base] = sample_value(0, x, y, args.period);
                    rows[base + 1U] = sample_value(1, x, y, args.period);
                    rows[base + 2U] = sample_value(2, x, y, args.period);
                }
            }

            char* logical_origin = reinterpret_cast<char*>(rows.data()) -
                                   static_cast<std::ptrdiff_t>(y0) *
                                       static_cast<std::ptrdiff_t>(row_stride);
            OPENEXR_IMF_NAMESPACE::FrameBuffer frame_buffer;
            frame_buffer.insert(
                "R",
                OPENEXR_IMF_NAMESPACE::Slice(
                    OPENEXR_IMF_NAMESPACE::FLOAT,
                    logical_origin,
                    pixel_stride,
                    row_stride));
            frame_buffer.insert(
                "G",
                OPENEXR_IMF_NAMESPACE::Slice(
                    OPENEXR_IMF_NAMESPACE::FLOAT,
                    logical_origin + sizeof(float),
                    pixel_stride,
                    row_stride));
            frame_buffer.insert(
                "B",
                OPENEXR_IMF_NAMESPACE::Slice(
                    OPENEXR_IMF_NAMESPACE::FLOAT,
                    logical_origin + 2U * sizeof(float),
                    pixel_stride,
                    row_stride));
            output.setFrameBuffer(frame_buffer);
            output.writePixels(count);
        }
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temp_path, ignored);
        throw;
    }

    if (args.inject_before_publish) {
        std::filesystem::remove(temp_path);
        throw std::runtime_error("injected failure before publish");
    }
    atomic_replace(temp_path, args.output);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        write_file(parse_arguments(argc, argv));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "p267 writer error: " << error.what() << '\n';
        return 2;
    }
}
