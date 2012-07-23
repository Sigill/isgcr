#ifndef HARALICK_H
#define HARALICK_H


#include <stdexcept>

class HaralickImageException : public std::runtime_error
{
public:
  HaralickImageException ( const std::string &err ) : std::runtime_error (err) {}
};

NormalizedHaralickImage::Pointer load_texture_image(const std::string filename, const unsigned int _posterizationLevel, const unsigned int _windowRadius);

#endif /* HARALICK_H */
