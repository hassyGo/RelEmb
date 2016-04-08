#include "Vocabulary.hpp"
#include "TrainingData.hpp"
#include "Params.hpp"
#include <sstream>
#include <iostream>

/*
  Arguments
  argv[1]: dimensionality
  argv[2]: learningRate
  argv[3]: model file name
  argv[4]: context size
  argv[5]: vocabulary size
  argv[6]: # of threads
  argv[7]: list of files for training
  argv[8]: list of nouns used for training
  argv[9]: list of words used for training
  argv[10]: # of negative samples
*/

int main(int argc, char** argv){
  const int dim = atoi(argv[1]);
  const double learningRate = atof(argv[2]);
  const std::string model = (std::string)argv[3];
  const int contextSize = atoi(argv[4]);
  const int vocSize = atoi(argv[5]);
  const int numThreads = atoi(argv[6]);
  const std::string fileList = (std::string)argv[7];
  const std::string nounList = (std::string)argv[8];
  const std::string wordList = (std::string)argv[9];
  const bool gzip = true; //gzip?
  Vocabulary voc(dim, contextSize, vocSize, nounList, wordList);
  
  Eigen::initParallel();
  TrainingData::numNegative = atoi(argv[10]);
  printf("Context size: %d + %d\n", voc.contextSize, voc.contextSize);
  printf("Negative sampling: %d\n", TrainingData::numNegative);
  
  std::ifstream ifs(fileList.c_str());
  std::vector<std::string> files;
  
  for (std::string line; std::getline(ifs, line); ){
    files.push_back(line);
  }

  double lr = learningRate;
  const double shrink = learningRate/files.size();  

  for (auto it = files.begin(); it != files.end(); ++it){
    std::vector<TrainingData*> data;

    TrainingData::readUnlabeled(data, voc, *it, gzip);
    
    printf("training ... %s\n", it->c_str());
    printf("(# of training samples: %zd, learning rate: %f/%f)\n", data.size(), lr, learningRate);
    std::random_shuffle(data.begin(), data.end());
    TrainingData::trainNEG(data, voc, lr, shrink, numThreads);
    lr -= shrink;
    
    for (int j = 0; j < (int)data.size(); ++j){
      delete data[j];
    }
    
    std::vector<TrainingData*>().swap(data);
  }

  Params::save(model, voc);

  return 0;
}
