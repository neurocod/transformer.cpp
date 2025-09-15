#pragma once

class CharToIdTextGenerator {
public:
    CharToIdTextGenerator();
    virtual ~CharToIdTextGenerator() {}

    std::string generateSimple();
    std::string generateComplex();

protected:
    mutable std::mt19937 _rand;
    const std::string _trash;
    
    char randChar(const std::string& source) const;
    std::string randTrash(int maxLen = 5) const;
};