//
//   Copyright 2019 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef ARG_UTILS_H
#define ARG_UTILS_H

#include "shape_utils.h"

#include <vector>

class ArgOptions 
{
public:

    ArgOptions();

    // Uses argc and argv to initialize the members of this object.
    void Parse(int argc, char **argv);

    // Prints out unrecognized argument warnings for each argument left in
    // remainingArgs
    void PrintUnrecognizedArgsWarnings() const;

    // Print unrecognized warning for arg
    void PrintUnrecognizedArgWarning(const char *arg) const;


    // Accessors to parsed arguments
    //

    bool GetAdaptive() const { return _adaptive; }

    bool GetFullScreen() const { return _fullscreen; }

    int GetLevel() const { return _level; }

    bool GetObjsAreAnim() const { return _objsAreAnim; }

    bool GetYUp() const { return _yup; }

    int GetRepeatCount() const { return _repeatCount; }
    
    Scheme GetDefaultScheme() const { return _defaultScheme; }

    const std::vector<const char *> GetObjFiles() const { return _objFiles; }

    const std::vector<const char *> GetRemainingArgs() const {
        return _remainingArgs; }


    // Operations on parsed arguments
    //

    size_t AppendObjShapes(std::vector<ShapeDesc>& shapes,
                           bool warn = true) const;

private:

    bool _adaptive;

    bool _fullscreen;

    int _level;

    bool _objsAreAnim;

    bool _yup;

    int _repeatCount;

    Scheme _defaultScheme;

    // .obj files that we've parsed
    std::vector<const char *> _objFiles;

    // Remaining args that we have not parsed, in order that they've appeared
    std::vector<const char *> _remainingArgs;

};

#endif // COMMON_ARGS_H
