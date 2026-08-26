#include "dng_hue_sat_map.h"
#include "dng_reference.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

template <typename T> bool read_value (std::ifstream &stream, T &value)
    { return static_cast<bool> (stream.read (reinterpret_cast<char *> (&value), sizeof (T))); }

int main (int argc, char **argv)
    {
    if (argc != 3) return 2;
    std::ifstream input (argv [1], std::ios::binary);
    std::ofstream output (argv [2], std::ios::binary | std::ios::trunc);
    char magic [8];
    if (!input.read (magic, 8) || std::string (magic, 8) != "P243HS01") return 3;
    uint32 h, s, v, count;
    if (!read_value (input, h) || !read_value (input, s) || !read_value (input, v) || !read_value (input, count)) return 4;
    dng_hue_sat_map lut;
    lut.SetDivisions (h, s, v);
    for (uint32 vi = 0; vi < v; ++vi)
        for (uint32 hi = 0; hi < h; ++hi)
            for (uint32 si = 0; si < s; ++si)
                {
                dng_hue_sat_map::HSBModify entry;
                if (!read_value (input, entry.fHueShift) || !read_value (input, entry.fSatScale) || !read_value (input, entry.fValScale)) return 5;
                lut.SetDelta (hi, si, vi, entry);
                }
    std::vector<real32> r (count), g (count), b (count), dr (count), dg (count), db (count);
    for (uint32 i = 0; i < count; ++i)
        if (!read_value (input, r [i]) || !read_value (input, g [i]) || !read_value (input, b [i])) return 6;
    char extra;
    if (input.read (&extra, 1)) return 7;
    RefBaselineHueSatMap (r.data (), g.data (), b.data (), dr.data (), dg.data (), db.data (), count, lut, nullptr, nullptr, false);
    output.write ("P243HO01", 8);
    output.write (reinterpret_cast<const char *> (&count), sizeof (count));
    for (uint32 i = 0; i < count; ++i)
        {
        output.write (reinterpret_cast<const char *> (&dr [i]), sizeof (real32));
        output.write (reinterpret_cast<const char *> (&dg [i]), sizeof (real32));
        output.write (reinterpret_cast<const char *> (&db [i]), sizeof (real32));
        }
    return output ? 0 : 8;
    }
