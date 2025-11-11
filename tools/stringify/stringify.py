#
#   Copyright 2022 Pixar
#
#    Licensed under the terms set forth in the LICENSE.txt file available at
#    https://opensubdiv.org/license.
#

from __future__ import print_function
import sys

def stringify(s):
    withinStringConstant = False

    result = ""
    for i in range(len(s)-1):
        # escape double quotes
        if s[i] == '"':
            result += '\\'
            withinStringConstant = not withinStringConstant

        if s[i] == '\\' and i == len(s)-2:
            return '"' + result + '"\n'

        # escape backslash
        if withinStringConstant and s[i] == '\\':
            result += '\\'

        result += s[i]

    return '"' + result + '\\n"\n'

def stringifyFile(inputFilename, outputFilename):
    with open(inputFilename, "r") as inputFile, \
         open(outputFilename, "w") as outputFile:
        for line in inputFile:
            outputFile.write(stringify(line))
        outputFile.write('"\\n"\n')

if len(sys.argv) != 3:
    print("Usage: stringify input-file output-file")
    sys.exit(1)

stringifyFile(sys.argv[1], sys.argv[2])

