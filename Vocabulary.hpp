#pragma once

#include "Matrix.hpp"
#include "Utils.hpp"
#include <unordered_map>
#include <string>
#include <vector>

class Vocabulary{
public:
  const int wordDim;
  const int contextSize;
  const int scoreDim;

  MatD nounVector;
  MatD contextVector;

  MatD scoreVector;
  MatD scoreBias;

  std::unordered_map<std::string, int> nounIndices;
  std::vector<std::string> nounStr;
  std::vector<double> nounDiscardProb;

  std::unordered_map<std::string, int> contextIndices;
  std::vector<std::string> contextStr;
  std::vector<double> contextDiscardProb;
  std::vector<int> contextWordDistribution;
  int nullIndex;
  int unkIndex;

  Vocabulary();
  Vocabulary(const int dim, const int numContext, const int vocSize, const std::string& nounList, const std::string& wordList);
  
  void outputBinary();
  std::vector<int> getKNNIndices(const MatD& wordVector, int k = 1);
  void wordKnn(int k);
  void loadPretrainedVector(const std::string& file);
  void loadVocabulary(const std::string& nounList, const std::string& wordList, int vocSize);
  void initWordVectors();
};
