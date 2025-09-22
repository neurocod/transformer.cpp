#pragma once

class CharToIdGenerator {
public:
    CharToIdGenerator();
    virtual ~CharToIdGenerator() {}

    std::string generateSimple();
    std::string generateComplex();

protected:
    mutable std::mt19937 _rand;
    const std::string _trash;
    
    char randChar(const std::string& source) const;
    std::string randTrash(int maxLen) const;
};