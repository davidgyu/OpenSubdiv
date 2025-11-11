//
//   Copyright 2019 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "arg_utils.h"

#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
parseIntArg(const char* argString, int dfltValue = 0) {
    char *argEndptr;
    int argValue = (int) strtol(argString, &argEndptr, 10);
    if (*argEndptr != 0) {
        printf("Warning: non-integer option parameter '%s' ignored\n", 
			   argString);
        argValue = dfltValue;
    }
    return argValue;
}


ArgOptions::ArgOptions() 
    : _adaptive(true)
    , _fullscreen(false)
    , _level(2)
    , _objsAreAnim(false)
    , _yup(false)
    , _repeatCount(0)
    , _defaultScheme(kCatmark)
{
}

void
ArgOptions::Parse(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {

        if (strstr(argv[i], ".obj")) {
            _objFiles.push_back(argv[i]);
        } else if (!strcmp(argv[i], "-a")) {
            _adaptive = true;
        } else if (!strcmp(argv[i], "-u")) {
            _adaptive = false;
        } else if (!strcmp(argv[i], "-l")) {
            if (++i < argc) _level = parseIntArg(argv[i], 2);
        } else if (!strcmp(argv[i], "-c")) {
            if (++i < argc) _repeatCount = parseIntArg(argv[i], 0);
        } else if (!strcmp(argv[i], "-f")) {
            _fullscreen = true;
        } else if (!strcmp(argv[i], "-yup")) {
            _yup = true;
        } else if (!strcmp(argv[i], "-anim")) {
            _objsAreAnim = true;
        } else if (!strcmp(argv[i], "-bilinear")) {
            _defaultScheme = kBilinear;
        } else if (!strcmp(argv[i], "-catmark")) {
            _defaultScheme = kCatmark;
        } else if (!strcmp(argv[i], "-loop")) {
            _defaultScheme = kLoop;
        } else {
            _remainingArgs.push_back(argv[i]);
        }

    }
}

void 
ArgOptions::PrintUnrecognizedArgWarning(const char *arg) const
{
    printf("Warning: unrecognized argument '%s' ignored\n", arg);
}

void 
ArgOptions::PrintUnrecognizedArgsWarnings() const
{
    for(size_t i = 0; i < _remainingArgs.size(); ++i) {
        PrintUnrecognizedArgWarning(_remainingArgs[i]);
    }
}

size_t
ArgOptions::AppendObjShapes(std::vector<ShapeDesc>& shapes, bool warn) const
{
    size_t originalShapesSize = shapes.size();

    for (size_t i = 0; i < GetObjFiles().size(); ++i) {
        std::ifstream ifs(GetObjFiles()[i]);
        if (ifs) {
            std::stringstream ss;
            ss << ifs.rdbuf();
            ifs.close();
            std::string str = ss.str();
            shapes.push_back(ShapeDesc(
                        GetObjFiles()[i], str.c_str(),
                        GetDefaultScheme()));
        } else if (warn) {
            printf("Warning: cannot open shape file '%s'\n",
                   GetObjFiles()[i]);
        }
    }
    return shapes.size() - originalShapesSize;
}
