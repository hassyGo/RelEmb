#pragma once

#include "Vocabulary.hpp"
#include "Rand.hpp"
#include <vector>

class TrainingData{
public:
  TrainingData(int ln, int rn): leftNoun(ln), rightNoun(rn) {};
  ~TrainingData(){
    std::vector<int>().swap(this->between);
    std::vector<int>().swap(this->before);
    std::vector<int>().swap(this->after);
  };

  static const int BETWEEN_NUM = 10;
  static const int BEFORE_AFTER_NUM = 5;
  
  //basic info.
  int leftNoun;
  int rightNoun;
  std::vector<int> between;
  std::vector<int> before;
  std::vector<int> after;

  //negative sampling info.
  static int numNegative;

  static bool readUnlabeled(std::vector<TrainingData*>& data, Vocabulary& voc, const std::string& fileName, const bool gzip = false);
  static void trainNEG(std::vector<TrainingData*>& data, Vocabulary& voc, const double learningRate, const double shrink, const int numThreads = 1);

private:
  static void* threadFunc(void* id);
  static void trainBetween(TrainingData* sample, Vocabulary& voc, double learningRate, Rand* rnd);
};
