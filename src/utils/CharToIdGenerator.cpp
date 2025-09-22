#include "utils/CharToIdGenerator.h"


CharToIdGenerator::CharToIdGenerator():
    _trash("abcd")
    //_trash("abcdefghijklmnopqrstuvwxyz")
{
    // lowercase letters is irrelevant information, uppercase is of interest
}

char CharToIdGenerator::randChar(const std::string& source) const {
    return source[_rand() % source.length()];
}

std::string CharToIdGenerator::randTrash(int maxLen) const {
    std::string s;
    int randLen = _rand() % maxLen;
    for (int n = 0; n < randLen; ++n)
        s += randChar(_trash);
    return s;
}

std::string CharToIdGenerator::generateSimple() {
    // makes simple sequences 1A and 2B with trash between them
    std::string ret;
    int tick = 0;
    for (int i = 0; i < 50; ++i) {
        ret += randTrash(4);
        if (tick++ % 2 == 0)
            ret += "1A";
        else
            ret += "2B";
    }
    return ret;
}

std::string CharToIdGenerator::generateComplex() {
    std::string ret;
    const std::string upperExceptAB = "CDEFGHIJKLMNOPQRSTUVWXYZ";
    // A and B are rules to get symbol from left or right of the char, 1 and 2 are requests
    for (int i = 0; i < 50; ++i) {
        ret += randTrash(5);
        if (_rand() % 2) {
            char x = randChar(upperExceptAB);
            ret += x;
            ret += 'A';
            ret += randTrash(5);
            ret += '1';
            ret += x;
        } else {
            char x = randChar(upperExceptAB);
            ret += 'B';
            ret += x;
            ret += randTrash(5);
            ret += '2';
            ret += x;
        }
    }
    return ret;
}