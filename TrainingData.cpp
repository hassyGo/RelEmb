#include "TrainingData.hpp"
#include <fstream>
#include <cassert>
#include <unordered_map>
#include <iostream>
#include <boost/algorithm/string.hpp>

int TrainingData::numNegative = 1;

bool TrainingData::readUnlabeled(std::vector<TrainingData*>& data, Vocabulary& voc, const std::string& fileName, const bool gzip){
  std::istream* is;

  if (gzip){
    is = (std::istream*)Utils::gzipIstream(fileName.c_str());
  }
  else {
    is = new std::ifstream(fileName.c_str());
  }

  if (!is){
    return false;
  }

  for (std::string line; std::getline(*is, line);){
    int beg = 0;
    bool e1 = false, e2 = false, betw = false, befo = false;
    int e1Index = -1, e2Index = -1;
    std::unordered_map<std::string, int>::iterator it;

    for (int i = 0, len = line.length(); i < len; ++i){
      if (i == len-1 || line[i+1] == ' '){
	int begTmp = beg;

	beg = i+2;

	if (e1Index == -1){
	  it = voc.nounIndices.find(line.substr(begTmp, i-begTmp+1));
	  e1Index = (it != voc.nounIndices.end()) ? it->second : voc.unkIndex;

	  e1 = true;
	  continue;
	}

	if (e1){
	  it = voc.nounIndices.find(line.substr(begTmp, i-begTmp+1));
	  e2Index = (it != voc.nounIndices.end()) ? it->second : voc.unkIndex;

	  e1 = false;
	  e2 = true;
	  data.push_back(new TrainingData(e1Index, e2Index));
	  data.back()->between.reserve(voc.contextSize*2+TrainingData::BETWEEN_NUM);
	  data.back()->before.reserve(TrainingData::BEFORE_AFTER_NUM);
	  data.back()->after.reserve(TrainingData::BEFORE_AFTER_NUM);

	  for (int j = 0; j < voc.contextSize; ++j){
	    data.back()->between.push_back(voc.nullIndex);
	  }

	  continue;
	}

	if (e2){
	  //<BETWEEN>
	  if (!betw && line[begTmp] == '<' && line[begTmp+1] == 'B'){
	    betw = true;
	    continue;
	  }
	  //<BEFORE>
	  if (betw && line[begTmp] == '<' && line[begTmp+1] == 'B'){
	    e2 = false;
	    continue;
	  }

	  it = voc.contextIndices.find(line.substr(begTmp, i-begTmp+1));
	  data.back()->between.push_back((it != voc.contextIndices.end()) ? it->second : voc.unkIndex);

	  continue;
	}

	if (betw){
	  //<AFTER>
	  if (!befo && line[begTmp] == '<' && line[begTmp+1] == 'A'){
	    befo = true;
	    betw = false;
	    //break;
	    continue;
	  }

	  it = voc.contextIndices.find(line.substr(begTmp, i-begTmp+1));

	  //only known words
	  if (it != voc.contextIndices.end()){
	    data.back()->before.push_back(it->second);
	  }

	  continue;
	}

	if (befo){
	  it = voc.contextIndices.find(line.substr(begTmp, i-begTmp+1));

	  //only known words
	  if (it != voc.contextIndices.end()){
	    data.back()->after.push_back(it->second);
	  }

	  continue;
	}
      }
    }

    //reverse the order
    for (int j = 0, len = data.back()->before.size(), tmp; j < len/2; ++j){
      tmp = data.back()->before[j];
      data.back()->before[j] = data.back()->before[len-1-j];
      data.back()->before[len-1-j] = tmp;
    }
    
    for (int j = 0; j < voc.contextSize; ++j){
      data.back()->between.push_back(voc.nullIndex);
    }
  }

  delete is;

  return true;
}
