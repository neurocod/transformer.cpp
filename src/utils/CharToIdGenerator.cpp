#include "utils/CharToIdGenerator.h"


CharToIdTextGenerator::CharToIdTextGenerator() 
    : _trash("abcdefghijklmnopqrstuvwxyz") {
    // lowercase letters is irrelevant information, uppercase is of interest
}

char CharToIdTextGenerator::randChar(const std::string& source) const {
    return source[_rand() % source.length()];
}

std::string CharToIdTextGenerator::randTrash(int maxLen) const {
    std::string s;
    int randLen = _rand() % maxLen;
    for (int n = 0; n < randLen; ++n)
        s += randChar(_trash);
    return s;
}

std::string CharToIdTextGenerator::generateSimple() {
    // makes simple sequences 1A and 2B with trash between them
    std::string ret;
    for (int i = 0; i < 50; ++i) {
        ret += randTrash();
        if (_rand() % 2)
            ret += "1A";
        else
            ret += "2B";
    }
    return ret;
}

std::string CharToIdTextGenerator::generateComplex() {
    std::string ret;
    const std::string upperExceptAB = "CDEFGHIJKLMNOPQRSTUVWXYZ";
    // A and B are rules to get symbol from left or right of the char, 1 and 2 are requests
    for (int i = 0; i < 50; ++i) {
        ret += randTrash();
        if (_rand() % 2) {
            char x = randChar(upperExceptAB);
            ret += x;
            ret += 'A';
            ret += randTrash();
            ret += '1';
            ret += x;
        } else {
            char x = randChar(upperExceptAB);
            ret += 'B';
            ret += x;
            ret += randTrash();
            ret += '2';
            ret += x;
        }
    }
    return ret;
}