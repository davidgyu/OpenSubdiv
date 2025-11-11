//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#define HDR_X_MAJOR (1 << 0)
#define HDR_Y_MAJOR (1 << 1)
#define HDR_X_DEC   (1 << 2)
#define HDR_Y_DEC   (1 << 3)

struct HdrInfo
{
    char magic[64];
    char format[64];
    double exposure;
    int width;
    int height;
    char flag;
    int scanLength;
    int scanWidth;
};

extern unsigned char *loadHdr(const char *filename, HdrInfo *info, bool convertToFloat);
