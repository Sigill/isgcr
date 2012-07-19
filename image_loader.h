#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdexcept>

#include "common.h"

#include "itkImageFileReader.h"
#include "itkImageSeriesReader.h"

class ImageLoadingException : public std::runtime_error
{
public:
  ImageLoadingException ( const std::string &err ) : std::runtime_error (err) {}
};


class ImageLoader
{
private:
  typedef itk::ImageFileReader< ImageType > ImageReader;
  typedef itk::ImageSeriesReader< ImageType > ImageSeriesReader;

public:
  /**
   * Load an image either as a single file or as a serie of files.
   * @param[in] filename The file to load of the folder containing the files. Must exists.
   */
  static ImageType::Pointer load(const std::string filename) throw(ImageLoadingException);

private:
  /**
   * Load an image as a single file.
   * @param[in] filename The file to load. Must exists.
   */
  static ImageType::Pointer loadImage(const std::string filename) throw(ImageLoadingException);

  /**
   * Load an image as a serie of files.
   * @param[in] filename The folder containing the files. Must be a directory.
   */
  static ImageType::Pointer loadImageSerie(const std::string filename) throw(ImageLoadingException);

};

#endif /* IMAGE_LOADER_H */
