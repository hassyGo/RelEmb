#pragma once

#include "Vocabulary.hpp"

class Params{
public:
  static void save(const std::string& fileName, Vocabulary& voc);
  static void load(const std::string& fileName, Vocabulary& voc);
};
