#include "Vocabulary.hpp"
#include "Rand.hpp"
#include <fstream>
#include <boost/bind.hpp>
#include <boost/algorithm/string.hpp>
#include <cstdio>
#include <iostream>
#include <omp.h>

Vocabulary::Vocabulary():
  wordDim(-1),
  contextSize(-1),
  scoreDim(-1)
{
}

Vocabulary::Vocabulary(const int dim, const int numContext, const int vocSize, const std::string& nounList, const std::string& wordList):
  wordDim(dim),
  contextSize(numContext),
  scoreDim(this->wordDim*(2+this->contextSize*2+2))
{
  printf("embedding size: %d\n", dim);

  this->loadVocabulary(nounList, wordList, vocSize);
  this->initWordVectors();
}

void Vocabulary::wordKnn(int k){
  printf("KNN words of words\n");

  for (std::string line; std::getline(std::cin, line) && line != "q"; ){
    if (!this->nounIndices.count(line)){
      continue;
    }

    std::vector<int> res = this->getKNNIndices(this->nounVector.col(this->nounIndices.at(line)), k);

    printf("norm: %f\n", this->nounVector.col(this->nounIndices.at(line)).norm());

    for (int i = 0; i < (int)res.size(); ++i){
      printf("(%.5f) %s\n", Utils::cosDis(this->nounVector.col(res[i]), this->nounVector.col(this->nounIndices.at(line))), this->nounStr[res[i]].c_str());
    }

    printf("\n");
  }
}

std::vector<int> Vocabulary::getKNNIndices(const MatD& wordVector, int k){
  std::vector<std::pair<int, double> > distances(this->nounStr.size());
  std::vector<int> indices;
  
  for (int i = 0; i < (int)distances.size(); ++i){
    MatD target = this->nounVector.col(i);
    
    distances[i] = std::pair<int, double>(i, Utils::cosDis(wordVector, target));
  }
  
  std::sort(distances.begin(), distances.end(), boost::bind(&std::pair<int, double>::second, _1) > boost::bind(&std::pair<int, double>::second, _2));
  
  for (int i = 0; (int)indices.size() < k; ++i){
    indices.push_back(distances[i].first);
  }
  
  return indices;
}

void Vocabulary::loadVocabulary(const std::string& nounList, const std::string& wordList, int vocSize){
  std::ifstream ifsNoun(nounList.c_str());
  std::ifstream ifsWord(wordList.c_str());
  int index = 0;
  double totalFreq = 0.0;

  assert(ifsNoun && ifsWord);
  
  //noun
  for (std::string line; std::getline(ifsNoun, line);){
    std::string n = line.substr(0, line.find(" "));
    int freq = atoi(line.substr(line.find(" ")+1).c_str());
    
    totalFreq += freq;

    if (index < vocSize){
      this->nounStr.push_back(n);
      this->nounDiscardProb.push_back(freq);
      assert(!this->nounIndices.count(n));
      this->nounIndices[n] = index++;
    }
  }

  for (int i = 0; i < (int)this->nounDiscardProb.size(); ++i){
    this->nounDiscardProb[i] /= totalFreq;
    this->nounDiscardProb[i] = 1.0-sqrt(1.0e-05/this->nounDiscardProb[i]);
  }
  
  index = 0;
  totalFreq = 0.0;

  //context
  for (std::string line; std::getline(ifsWord, line);){
    if (line.find("__") == std::string::npos){
      //continue;
    }

    std::string c = line.substr(0, line.find(" "));
    int freq = atoi(line.substr(line.find(" ")+1).c_str());
    
    totalFreq += freq;

    if (index < vocSize){
      this->contextStr.push_back(c);
      this->contextDiscardProb.push_back(freq);
      assert(!this->contextIndices.count(c));
      this->contextIndices[c] = index;
      
      int loop = pow(freq, 0.75);
      
      for (int i = 0; i < loop; ++i){
	this->contextWordDistribution.push_back(index);
      }
      
      ++index;
    }
  }

  //for null pad
  this->nullIndex = index;
  this->unkIndex = this->nullIndex+1;
  this->contextStr.push_back("NULL");
  this->contextStr.push_back("UNK");

  for (int i = 0; i < (int)this->contextDiscardProb.size(); ++i){
    this->contextDiscardProb[i] /= totalFreq;
    this->contextDiscardProb[i] = 1.0-sqrt(1.0e-05/this->contextDiscardProb[i]);
  }
  
  printf("# of noun words    : %zd\n", this->nounStr.size());
  printf("# of context words : %zd\n", this->contextStr.size());
}

void Vocabulary::outputBinary(){
  std::ofstream ofs("/home/hassy/data_local/rnn/wordParams.bin", std::ios::out|std::ios::binary);

  exit(1);
}

void Vocabulary::initWordVectors(){
  this->scoreVector = MatD::Zero(this->scoreDim, this->contextStr.size()-2);
  this->scoreBias = MatD::Zero(1, this->contextStr.size()-2);

  Rand rnd;
  double r = 1.0/this->wordDim;

  this->nounVector = MatD(this->wordDim, this->nounStr.size());
  this->contextVector = MatD(this->wordDim, this->contextStr.size());

  rnd.gauss(this->nounVector, r);
  rnd.gauss(this->contextVector, r);
}

void Vocabulary::loadPretrainedVector(const std::string& file){
  std::ifstream ifs1(file.c_str());
  std::ifstream ifs2((file+".score").c_str());

  assert(ifs1 && ifs2);

  for (std::string line; std::getline(ifs1, line); ){
    std::vector<std::string> res;

    boost::split(res, line, boost::is_space());

    std::unordered_map<std::string, int>::iterator it = this->nounIndices.find(res[0]);

    if (it != this->nounIndices.end()){
      for (int i = 0; i < this->wordDim; ++i){
	this->nounVector.coeffRef(i, it->second) = atof(res[i+1].c_str());
      }
    }

    it = this->contextIndices.find(res[0]);

    if (it != this->contextIndices.end()){
      for (int i = 0; i < this->wordDim; ++i){
	this->contextVector.coeffRef(i, it->second) = atof(res[i+1].c_str());
      }
    }
  }
  
  for (std::string line; std::getline(ifs2, line); ){
    std::vector<std::string> res;

    boost::split(res, line, boost::is_space());

    std::unordered_map<std::string, int>::iterator it = this->contextIndices.find(res[0]);

    if (it != this->contextIndices.end()){
      for (int i = 0; i < this->wordDim; ++i){
	this->scoreVector.coeffRef(i, it->second) = atof(res[i+1].c_str());
      }
    }
  }
}
