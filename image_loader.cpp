#include "image_loader.h"

#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"

#include <ostream>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

typedef itk::ImageFileReader< ImageType > ImageReader;
typedef itk::ImageSeriesReader< ImageType > ImageSeriesReader;


ImageType::Pointer ImageLoader::load(const std::string filename) throw(ImageLoadingException)
{
  std::cerr << "Loading " << filename << std::endl;
  try
  {
    boost::filesystem::path path(filename);
    if(boost::filesystem::is_directory(path))
    {
      std::cerr << "\t" << filename << " is a folder" << std::endl;
      return loadImageSerie(filename);
    } else {
      std::cerr << "\t" << filename << " is a file" << std::endl;
      return loadImage(filename);
    }
  } catch(boost::filesystem::filesystem_error & ex) {
    std::stringstream err;
    err << filename << " cannot be read" << std::endl;
    throw new ImageLoadingException(err.str());
  }
}

ImageType::Pointer ImageLoader::loadImage(const std::string filename) throw(ImageLoadingException)
{
  typename ImageReader::Pointer reader = ImageReader::New();

  reader->SetFileName(filename);

  try {
    reader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    throw ImageLoadingException(err.what());
  }

  return reader->GetOutput();
}

ImageType::Pointer ImageLoader::loadImageSerie(const std::string filename) throw(ImageLoadingException)
{
  typename ImageSeriesReader::Pointer reader = ImageSeriesReader::New();

  typename ImageSeriesReader::FileNamesContainer filenames;

  try
  {
    boost::filesystem::path path(filename);
    boost::filesystem::directory_iterator end_iter;
    boost::regex pattern(".*\\.((?:png)|(?:bmp)|(?:jpe?g))", boost::regex::icase);

    for( boost::filesystem::directory_iterator dir_iter(path) ; dir_iter != end_iter ; ++dir_iter)
    {
      boost::smatch match;
      if( !boost::regex_match( dir_iter->path().filename().string(), match, pattern ) ) continue;

      std::cerr << "\tLoading " << boost::filesystem::absolute(dir_iter->path()).string() << std::endl;
      filenames.push_back(boost::filesystem::absolute(dir_iter->path()).string());
    }
  }
  catch(boost::filesystem::filesystem_error e) {
    std::stringstream err;
    err << "Something wrong happened while listing the files in " << filename;
    throw ImageLoadingException(err.str());
  }

  std::sort(filenames.begin(), filenames.end());

  reader->SetFileNames(filenames);

  try {
    reader->Update();
  }
  catch( itk::ExceptionObject & err )
  {
    throw ImageLoadingException(err.what());
  }

  return reader->GetOutput();
}

