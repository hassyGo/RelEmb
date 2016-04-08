#include "Params.hpp"
#include "TrainingData.hpp"
#include <fstream>

#define INF_NAN(x) assert(!isnan(x) && !isinf(x))

void Params::save(const std::string& fileName, Vocabulary& voc){
  std::ofstream ofs(fileName.c_str(), std::ios::out|std::ios::binary);
  float val = 0.0;

  assert(ofs);

  for (int i = 0; i < voc.nounVector.cols(); ++i){
    for (int j = 0; j < voc.nounVector.rows(); ++j){
      val = voc.nounVector.coeff(j, i);
      INF_NAN(val);
      ofs.write((char*)&val, sizeof(float));
    }
  }

  //between
  for (int i = 0; i < voc.contextVector.cols(); ++i){
    for (int j = 0; j < voc.contextVector.rows(); ++j){
      val = voc.contextVector.coeff(j, i);
      INF_NAN(val);
      ofs.write((char*)&val, sizeof(float));
    }
  }

  for (int i = 0; i < voc.scoreVector.cols(); ++i){
    for (int j = 0; j < voc.scoreVector.rows(); ++j){
      val = voc.scoreVector.coeff(j, i);
      INF_NAN(val);
      ofs.write((char*)&val, sizeof(float));
    }
  }

  //bais
  for (int i = 0; i < voc.scoreBias.cols(); ++i){
    val = voc.scoreBias.coeff(0, i);
    INF_NAN(val);
    ofs.write((char*)&val, sizeof(float));
  }

  printf("Saved parameters at \"%s\"\n", fileName.c_str());
}

void Params::load(const std::string& fileName, Vocabulary& voc){
  std::ifstream ifs(fileName.c_str(), std::ios::in|std::ios::binary);
  float val = 0.0;

  assert(ifs);

  for (int i = 0; i < voc.nounVector.cols(); ++i){
    for (int j = 0; j < voc.nounVector.rows(); ++j){
      ifs.read((char*)&val, sizeof(float));
      voc.nounVector.coeffRef(j, i) = val;
      INF_NAN(val);
    }
  }

  //between
  for (int i = 0; i < voc.contextVector.cols(); ++i){
    for (int j = 0; j < voc.contextVector.rows(); ++j){
      ifs.read((char*)&val, sizeof(float));
      voc.contextVector.coeffRef(j, i) = val;
      INF_NAN(val);
    }
  }

  for (int i = 0; i < voc.scoreVector.cols(); ++i){
    for (int j = 0; j < voc.scoreVector.rows(); ++j){
      ifs.read((char*)&val, sizeof(float));
      voc.scoreVector.coeffRef(j, i) = val;
      INF_NAN(val);
    }
  }

  //bais
  for (int i = 0; i < voc.scoreBias.cols(); ++i){
    ifs.read((char*)&val, sizeof(float));
    voc.scoreBias.coeffRef(0, i) = val;
    INF_NAN(val);
  }
}
