#include "image_loader.h"

#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"

#include <ostream>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include "Logger.h"

typedef itk::ImageFileReader< ImageType > ImageReader;
typedef itk::ImageSeriesReader< ImageType > ImageSeriesReader;


ImageType::Pointer ImageLoader::load(const std::string filename)
{
	log4cxx::LoggerPtr logger = Logger::getInstance();

	LOG4CXX_INFO(logger, "Loading \"" << filename << "\"");

	try
	{
		boost::filesystem::path path(filename);

		if(boost::filesystem::exists(path)) {
			if(boost::filesystem::is_directory(path))
			{
				LOG4CXX_DEBUG(logger, "\"" << filename << "\"" << " is a folder");

				return loadImageSerie(filename);
			} else {
				LOG4CXX_DEBUG(logger, "\"" << filename << "\"" << " is a file");

				return loadImage(filename);
			}
		} else {
			std::stringstream err;
			err << "\"" << filename << "\" does not exists";

			LOG4CXX_FATAL(logger, err.str());

			throw ImageLoadingException(err.str());
		}
	} catch(boost::filesystem::filesystem_error &ex) {
		std::stringstream err;
		err << filename << " cannot be read (" << ex.what() << ")" << std::endl;
		throw ImageLoadingException(err.str());
	}
}

ImageType::Pointer ImageLoader::loadImage(const std::string filename)
{
	typename ImageReader::Pointer reader = ImageReader::New();

	reader->SetFileName(filename);

	try {
		reader->Update();
	}
	catch( itk::ExceptionObject &ex )
	{
		std::stringstream err;
		err << "ITK is unable to load the image \"" << filename << "\" (" << ex.what() << ")";

		throw ImageLoadingException(err.str());
	}

	return reader->GetOutput();
}

ImageType::Pointer ImageLoader::loadImageSerie(const std::string filename)
{
	typename ImageSeriesReader::Pointer reader = ImageSeriesReader::New();

	typename ImageSeriesReader::FileNamesContainer filenames;

	log4cxx::LoggerPtr logger = Logger::getInstance();

	try
	{
		boost::filesystem::path path(filename);
		boost::filesystem::directory_iterator end_iter;
		boost::regex pattern(".*\\.((?:png)|(?:bmp)|(?:jpe?g))", boost::regex::icase);

		for( boost::filesystem::directory_iterator dir_iter(path) ; dir_iter != end_iter ; ++dir_iter)
		{
			boost::smatch match;
			if( !boost::regex_match( dir_iter->path().filename().string(), match, pattern ) ) continue;

			LOG4CXX_DEBUG(logger, "Loading \"" << boost::filesystem::absolute(dir_iter->path()).string() << "\"");

			filenames.push_back(boost::filesystem::absolute(dir_iter->path()).string());
		}
	}
	catch(boost::filesystem::filesystem_error &ex) {
		std::stringstream err;
		err << filename << " cannot be read (" << ex.what() << ")" << std::endl;

		throw ImageLoadingException(err.str());
	}

	std::sort(filenames.begin(), filenames.end());

	reader->SetFileNames(filenames);

	try {
		reader->Update();
	}
	catch( itk::ExceptionObject &ex )
	{
		std::stringstream err;
		err << "ITK is unable to load the image serie located in \"" << filename << "\" (" << ex.what() << ")";

		throw ImageLoadingException(err.str());
	}

	return reader->GetOutput();
}

