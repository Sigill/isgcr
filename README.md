# ISGCR (Image Segmentation using Graph-based Classifier Regularization)

This tool is the result of the PhD I achieved in december 2014.

Basically, this tool uses supervised classifiers to produce a initial segmentation of an image which is then regularized using a graph-based regularization process. This tool supports 2D and 3D images and can use any kind of numerical pixel descriptor (for example, texture descriptors). It uses neural networks and SVM (SVM support is currently limited, since the tool does not offer a way to tune the parameters), and is able to perform multi-class segmentation.

You can read more about the whole process in [this](http://link.springer.com/chapter/10.1007/978-3-642-40261-6_37) article (also available [here](https://hal.archives-ouvertes.fr/hal-01027467/)), or in my [PhD thesis](http://www.theses.fr/2013TOUR4050) (in french).

## How to build

It has only been used on Linux systems, but should work on any Unix systems. It should also work on other operating systems, but getting every dependency to work may be painful.

This tool is written in C++, relies on [CMake](http://www.cmake.org/) for the build process, and is based on several open source projects:

* The [ITK](http://www.itk.org/) Segmentation and Registration Toolkit, for image IO.
* [FANN](http://leenissen.dk/fann/wp/), for neural networks (you will have to use the `static_libs_and_improved_cmake_support` branch of my [fork](https://github.com/Sigill/fann)).
* [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/), for SVM.
* [Tulip](http://tulip.labri.fr/TulipDrupal/), for graphs.
* [This](https://github.com/Sigill/tulip-plugin-grid3d-import) Tulip plugin, for creating 3D grids.
* [This](https://github.com/Sigill/tulip-plugin-grid3d-import) and [this](https://github.com/Sigill/tulip-plugin-image3d) Tulip plugins, for transforming images to graphs.
* [This](https://github.com/Sigill/tulip-plugin-rof-regularization) Rudin-Osher-Fatemi graph regularization plugin for Tulip.

Except for LIBSVM (which is often available in your package manager), you will have to produce you own builds of all those tools an libraries.

It has last been tested with ITK 4.5, Tulip 4.5, LIBSVM 3.1, but it may work with newer versions.

Then, use CMake and specify the path for all dependencies.

## How to use

    $ ./isgcr -h
    Usage: ./isgcr [options]
    Command line parameters:
      -h [ --help ]                         Produce help message.
      --debug                               Enables debug mode (will export 
                                            graphs).
      -i [ --input-image ] arg              Input image.
      -r [ --roi ] arg                      Region of interest.
      -E [ --export-dir ] arg               Export directory.
      -e [ --export-interval ] arg (=0)     Export interval during regularization.
      -n [ --num-iter ] arg (=0)            Number of iterations for the 
                                            regularization.
      --lambda arg (=1)                     Lambda parameter for regularization.
      --classifier-type arg (=0)            Type of classifier. (ann or svm)
      --classifier-training-image arg       An image from which the texture is 
                                            learned (use --classifier-training-imag
                                            e-class to define the regions to 
                                            learn). Multiple images can be 
                                            specified. If no image is specified, 
                                            the input image will be used.
      --classifier-training-image-class arg Defines a class to be learned from a 
                                            binary image. At least 2 values 
                                            required. If multiple images are used, 
                                            they must have the same number of 
                                            classes.
      --classifier-config-dir arg           Directory containing the classifier 
                                            configuration files.
      --ann-hidden-layer arg (=3)           Number of neurons per hidden layer 
                                            (default: one layer of 3 neurons).
      --ann-learning-rate arg (=0.1)        Learning rate of the neural networks.
      --ann-max-epoch arg (=1000)           Maximum number of training iterations 
                                            for the neural networks.
      --ann-mse-target arg (=0.0001)        Mean squared error targeted by the 
                                            neural networks training algorithm.
      --ann-validation-image arg            The images to use to validate the 
                                            training of the neural network (use 
                                            --ann-validation-image-class to define 
                                            associated classes). Multiple images 
                                            can be specified. They mush have the 
                                            same number of components per pixels 
                                            than the images on which the neural 
                                            network is trained.
      --ann-validation-image-class arg      Defines the classes of the images used 
                                            to validate de training of the neural 
                                            network. If multiple images are used, 
                                            they must have as much classes as the 
                                            images on which the neural network is 
                                            trained.
      --ann-build-validation-from-training arg (=0.333)
                                            The percentage of elements from the 
                                            training-set to extract to build the 
                                            validation-set.

Your input and every training image should be a vector image using floating point values (using for example the [MetaImage](http://www.itk.org/Wiki/ITK/MetaIO/Documentation) file format. [this tool](https://github.com/Sigill/ImageFeaturesComputer) can help you produce such images.

The region of interest (specified with the `--roi` option) and every "class" image must be black and white images (white pixels indicating the region of interest or the pixels belonging to the class). Use lossless image file formats (but not BMP, because ITK does (or used to) not supports binary BMP images.

### How to segment an image

This command will produce a segmentation of the `input.mha` image using a classifier implemented as a neural network (a multi-layer perceptron containing a single hidden layer of 3 neurons). The classifier is trained using the `input.mha` image, each class to learn being described by the `class1.png` and `class2.png` images. The result will be written in the `output_dir/` directory.

    ./isgcr -i input.mha -E output_dir/ --classifier-type ann --classifier-training-image-class class1.png class2.png --ann-hidden-layer 3

### How to train and save a classifier

This command will train a neural network classifier (see the previous section for the purpose of the parameters) and save its configuration in the `ann_config/` directory.

    ./isgcr --classifier-training-image training.mha --classifier-type ann --classifier-training-image-class class1.png class2.png --ann-hidden-layer 3 --classifier-config-dir ann_config/

Note: the `--classifier-config-dir` can also be used when performing a segmentation, it will allow you to have a preview of what the classifier has learned.

### How to segment an image using a pre-trained classifier

    ./isgcr -i input.mha -E output_dir --classifier-type ann --classifier-config-dir ann_config/

Of course, the number of descriptors in the input image and the type of classifier must match the parameters used when the classifier was trained.

## License

This tool is released under the terms of the MIT License. See the LICENSE.txt file for more details.