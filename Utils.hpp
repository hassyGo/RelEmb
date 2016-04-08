#pragma once

#include "Matrix.hpp"
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

namespace Utils{
  inline double cosDis(const MatD& a, const MatD& b){
    return a.col(0).dot(b.col(0))/(a.norm()*b.norm());
  }

  inline boost::iostreams::filtering_istream* gzipIstream(const std::string& fileName){
    boost::iostreams::filtering_istream* is = new boost::iostreams::filtering_istream();
  
    is->push(boost::iostreams::gzip_decompressor());
    is->push(boost::iostreams::file_source(fileName));
    return is;
  }
}
