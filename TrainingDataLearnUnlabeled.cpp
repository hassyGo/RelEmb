#include "TrainingData.hpp"
#include "Sigmoid.hpp"
#include <pthread.h>

class Arg{
public:
  Arg(std::vector<TrainingData*>& data, Vocabulary& voc): data(data), voc(voc) {};

  std::vector<TrainingData*>& data;
  Vocabulary& voc;
  Rand* r;
  int beg, end;
  double lr;
  double shr;
};

//training by Negative Sampling
void TrainingData::trainNEG(std::vector<TrainingData*>& data, Vocabulary& voc, const double learningRate, const double shrink, const int numThreads){
  static std::vector<Rand*> rgen;
  static double totalTime = 0.0;
  std::vector<Arg*> args(numThreads);
  clock_t startTime = clock();
  int step = data.size()/numThreads;

  if (rgen.empty()){
    for (int i = 0; i < numThreads; ++i){
      rgen.push_back(new Rand(Rand::r_.next()));
    }
  }
  
  pthread_t pt[numThreads];

  for (int i = 0; i < numThreads; ++i){
    args[i] = new Arg(data, voc);
    args[i]->r = rgen[i];
    args[i]->beg = i*step;
    args[i]->end = (i == numThreads-1 ? data.size()-1 : (i+1)*step-1);
    args[i]->lr = learningRate;
    args[i]->shr = shrink/(args[i]->end-args[i]->beg+1);
    
    pthread_create(&pt[i], 0, TrainingData::threadFunc, (void*)args[i]);
  }

  for (int i = 0; i < numThreads; ++i){
    pthread_join(pt[i], 0);
  }

  for (int i = 0; i < (int)args.size(); ++i){
    delete args[i];
  }
  
  std::vector<Arg*>().swap(args);

  double localTime = (double)(clock()-startTime)/CLOCKS_PER_SEC;
  totalTime += localTime;

  printf("processing time: %f sec (total: %f sec)\n", localTime, totalTime);
}

void* TrainingData::threadFunc(void* arg){
  Arg* a = (Arg*)arg;

  for (int i = a->beg; i <= a->end; ++i){
    if (a->data[i]->leftNoun == a->voc.unkIndex ||
	a->data[i]->rightNoun == a->voc.unkIndex ||
	//sub-sampling
	a->voc.nounDiscardProb[a->data[i]->leftNoun] > a->r->zero2one() ||
	a->voc.nounDiscardProb[a->data[i]->rightNoun] > a->r->zero2one()){
      continue;
    }

    TrainingData::trainBetween(a->data[i], a->voc, a->lr, a->r);
    a->lr -= a->shr;
  }

  pthread_exit(0);
}

void TrainingData::trainBetween(TrainingData* sample, Vocabulary& voc, double learningRate, Rand* rnd){
  MatD gradLeftNoun, gradRightNoun;
  MatD gradPrev(voc.wordDim, voc.contextSize), gradNext(voc.wordDim, voc.contextSize);
  MatD gradBefore, gradAfter;
  MatD beforeVec(voc.wordDim, 1), afterVec(voc.wordDim, 1);
  double lrTmp;

  for (int j = voc.contextSize, end = sample->between.size()-voc.contextSize; j < end; ++j){
    //sub-sampling
    if (sample->between[j] == voc.unkIndex || voc.contextDiscardProb[sample->between[j]] > rnd->zero2one()){
      continue;
    }

    std::unordered_map<int, int> negativeHist;

    int positive = sample->between[j];
    int left = sample->leftNoun;
    int right = sample->rightNoun;
    int beforeNum = sample->before.size();
    int afterNum = sample->after.size();

    beforeVec.setZero();
    afterVec.setZero();
    
    //process the positive sample
    double deltaPos =
      voc.nounVector.col(left).dot(voc.scoreVector.block( 0*voc.wordDim, positive, voc.wordDim, 1))+
      voc.nounVector.col(right).dot(voc.scoreVector.block(1*voc.wordDim, positive, voc.wordDim, 1));

    for (int i = 1; i <= voc.contextSize; ++i){
      deltaPos +=
	(voc.contextVector.col(sample->between[j-i]).dot(voc.scoreVector.block((i+1)*voc.wordDim, positive, voc.wordDim, 1))+
	 voc.contextVector.col(sample->between[j+i]).dot(voc.scoreVector.block((i+1+voc.contextSize)*voc.wordDim, positive, voc.wordDim, 1)));
    }

    for (int i = 0; i < beforeNum; ++i){
      beforeVec += voc.contextVector.col(sample->before[i]);
    }
    for (int i = 0; i < afterNum; ++i){
      afterVec += voc.contextVector.col(sample->after[i]);
    }
    beforeVec *= (1.0/beforeNum);
    afterVec *= (1.0/afterNum);

    if (beforeNum > 0){
      deltaPos +=
	beforeVec.dot(voc.scoreVector.block((2+2*voc.contextSize)*voc.wordDim, positive, voc.wordDim, 1));
    }
    if (afterNum > 0){
      deltaPos +=
	afterVec.dot(voc.scoreVector.block((2+2*voc.contextSize+1)*voc.wordDim, positive, voc.wordDim, 1));
    }

    deltaPos = Sigmoid::sigmoid(deltaPos+voc.scoreBias.coeff(0, positive))-1.0;
    lrTmp = learningRate*deltaPos;
    
    gradLeftNoun = deltaPos*voc.scoreVector.block(0, positive, voc.wordDim, 1);
    gradRightNoun = deltaPos*voc.scoreVector.block(voc.wordDim, positive, voc.wordDim, 1);
    voc.scoreVector.block(0*voc.wordDim, positive, voc.wordDim, 1) -= lrTmp*voc.nounVector.col(left);
    voc.scoreVector.block(1*voc.wordDim, positive, voc.wordDim, 1) -= lrTmp*voc.nounVector.col(right);

    for (int i = 1; i <= voc.contextSize; ++i){
      gradPrev.col(i-1) = deltaPos*voc.scoreVector.block((i+1)*voc.wordDim, positive, voc.wordDim, 1);
      voc.scoreVector.block((i+1)*voc.wordDim, positive, voc.wordDim, 1) -= lrTmp*voc.contextVector.col(sample->between[j-i]);

      gradNext.col(i-1) = deltaPos*voc.scoreVector.block((i+1+voc.contextSize)*voc.wordDim, positive, voc.wordDim, 1);
      voc.scoreVector.block((i+1+voc.contextSize)*voc.wordDim, positive, voc.wordDim, 1) -= lrTmp*voc.contextVector.col(sample->between[j+i]);
    }

    if (beforeNum > 0){
      gradBefore = deltaPos*voc.scoreVector.block((2+2*voc.contextSize)*voc.wordDim, positive, voc.wordDim, 1);
      voc.scoreVector.block((2+2*voc.contextSize)*voc.wordDim, positive, voc.wordDim, 1) -= lrTmp*beforeVec;
    }
    if (afterNum > 0){
      gradAfter = deltaPos*voc.scoreVector.block((2+2*voc.contextSize+1)*voc.wordDim, positive, voc.wordDim, 1);
      voc.scoreVector.block((2+2*voc.contextSize+1)*voc.wordDim, positive, voc.wordDim, 1) -= lrTmp*afterVec;
    }

    voc.scoreBias.coeffRef(0, positive) -= lrTmp;

    //process K negative samples
    for (int k = 0; k < TrainingData::numNegative; ++k){
      int negative = positive;

      while (negative == positive || negativeHist.count(negative)){
	negative = voc.contextWordDistribution[(rnd->next() >> 16)%voc.contextWordDistribution.size()];
      }
      
      negativeHist[negative] = 1;
      
      double deltaNeg =
	voc.nounVector.col(left).dot(voc.scoreVector.block( 0*voc.wordDim, negative, voc.wordDim, 1))+
	voc.nounVector.col(right).dot(voc.scoreVector.block(1*voc.wordDim, negative, voc.wordDim, 1));

      for (int i = 1; i <= voc.contextSize; ++i){
	deltaNeg +=
	  (voc.contextVector.col(sample->between[j-i]).dot(voc.scoreVector.block((i+1)*voc.wordDim, negative, voc.wordDim, 1))+
	   voc.contextVector.col(sample->between[j+i]).dot(voc.scoreVector.block((i+1+voc.contextSize)*voc.wordDim, negative, voc.wordDim, 1)));
      }

      if (beforeNum > 0){
	deltaNeg +=
	  beforeVec.dot(voc.scoreVector.block((2+2*voc.contextSize)*voc.wordDim, negative, voc.wordDim, 1));
      }
      if (afterNum > 0){
	deltaNeg +=
	  afterVec.dot(voc.scoreVector.block((2+2*voc.contextSize+1)*voc.wordDim, negative, voc.wordDim, 1));
      }

      deltaNeg = Sigmoid::sigmoid(deltaNeg+voc.scoreBias.coeff(0, negative));
      lrTmp = learningRate*deltaNeg;
    
      gradLeftNoun  += deltaNeg*voc.scoreVector.block(0*voc.wordDim, negative, voc.wordDim, 1);
      gradRightNoun += deltaNeg*voc.scoreVector.block(1*voc.wordDim, negative, voc.wordDim, 1);
      voc.scoreVector.block(0*voc.wordDim, negative, voc.wordDim, 1) -= lrTmp*voc.nounVector.col(left);
      voc.scoreVector.block(1*voc.wordDim, negative, voc.wordDim, 1) -= lrTmp*voc.nounVector.col(right);

      for (int i = 1; i <= voc.contextSize; ++i){
	gradPrev.col(i-1) += deltaNeg*voc.scoreVector.block((i+1)*voc.wordDim, negative, voc.wordDim, 1);
	voc.scoreVector.block((i+1)*voc.wordDim, negative, voc.wordDim, 1) -= lrTmp*voc.contextVector.col(sample->between[j-i]);

	gradNext.col(i-1) += deltaNeg*voc.scoreVector.block((i+1+voc.contextSize)*voc.wordDim, negative, voc.wordDim, 1);
	voc.scoreVector.block((i+1+voc.contextSize)*voc.wordDim, negative, voc.wordDim, 1) -= lrTmp*voc.contextVector.col(sample->between[j+i]);
      }

      if (beforeNum > 0){
	gradBefore += deltaNeg*voc.scoreVector.block((2+2*voc.contextSize)*voc.wordDim, negative, voc.wordDim, 1);
	voc.scoreVector.block((2+2*voc.contextSize)*voc.wordDim, negative, voc.wordDim, 1) -= lrTmp*beforeVec;
      }
      if (afterNum > 0){
	gradAfter += deltaNeg*voc.scoreVector.block((2+2*voc.contextSize+1)*voc.wordDim, negative, voc.wordDim, 1);
	voc.scoreVector.block((2+2*voc.contextSize+1)*voc.wordDim, negative, voc.wordDim, 1) -= lrTmp*afterVec;
      }

      voc.scoreBias.coeffRef(0, negative) -= lrTmp;
    }

    voc.nounVector.col(left) -= learningRate*gradLeftNoun;
    voc.nounVector.col(right) -= learningRate*gradRightNoun;

    for (int i = 1; i <= voc.contextSize; ++i){
      voc.contextVector.col(sample->between[j-i]) -= learningRate*gradPrev.col(i-1);
      voc.contextVector.col(sample->between[j+i]) -= learningRate*gradNext.col(i-1);
    }

    if (beforeNum > 0){
      gradBefore *= (learningRate/beforeNum);

      for (int i = 0; i < beforeNum; ++i){
	voc.contextVector.col(sample->before[i]) -= gradBefore;
      }
    }
    if (afterNum > 0){
      gradAfter *= (learningRate/afterNum);

      for (int i = 0; i < afterNum; ++i){
	voc.contextVector.col(sample->after[i]) -= gradAfter;
      }
    }
  }
}
