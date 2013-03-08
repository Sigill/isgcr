#include "cli_parser.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <vector>

#include "ParseUtils.h"

#include "log4cxx/logger.h"

namespace po = boost::program_options;

void validate(boost::any& v, const std::vector<std::string>& values, PositiveInteger*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	unsigned int value;
	if( ParseUtils::ParseUInt(value, s.data(), 10) && value >= 0 )
	{
		v = boost::any(PositiveInteger(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, StrictlyPositiveInteger*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	unsigned int value;
	if( ParseUtils::ParseUInt(value, s.data(), 10) && value > 0 )
	{
		v = boost::any(StrictlyPositiveInteger(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, Float*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	float value;
	if( ParseUtils::ParseFloat(value, s.data()) )
	{
		v = boost::any(Float(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, Double*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	double value;
	if( ParseUtils::ParseDouble(value, s.data()) )
	{
		v = boost::any(Double(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

void validate(boost::any& v, const std::vector<std::string>& values, Percentage*, int)
{
	po::validators::check_first_occurrence(v);
	const std::string& s = po::validators::get_single_string(values);

	float value;
	if( ParseUtils::ParseFloat(value, s.data()) && (value >= 0) && (value <= 100) )
	{
		v = boost::any(Percentage(value));
	} else {
		throw po::invalid_option_value(s);
	}
}

template< typename TElemType >
std::ostream &operator<<(std::ostream &s, const std::vector< TElemType >& v)
{
	std::stringstream ss;

	if( !v.empty() )
	{
		typename std::vector< TElemType >::const_iterator it = v.begin(), begin = v.begin(), end = v.end();

		for( ; it < end; ++it )
		{
			if(it != begin)
				ss << " ";
			ss << *it;
		}
	}

	return s << ss.str();
}

CliParser::CliParser()
{}

CliParser::ParseResult CliParser::parse_argv(int argc, char ** argv)
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	LOG4CXX_INFO(logger, "Parsing command line options");

	po::options_description desc("Command line parameters");
	desc.add_options()
		("help,h",
			"Produce help message.")
		("debug",
			"Enables debug mode (will export graphs).")
		("input-image,i",
			po::value< std::string >(&(this->input_image)),
			"Input image.")
		("roi,r",
			po::value< std::string >(&(this->region_of_interest))->default_value(""),
			"Region of interest.")
		("export-dir,E",
			po::value< std::string >(&(this->export_dir))->default_value(""),
			"Export directory.")
		("export-interval,e",
			po::value< PositiveInteger >(&(this->export_interval))->default_value(0),
			"Export interval during regularization.")
		("num-iter,n",
			po::value< PositiveInteger >(&(this->num_iter))->default_value(0),
			"Number of iterations for the regularization.")
		("lambda1",
			po::value< Double >(&(this->lambda1))->default_value(1.0),
			"Lambda 1 parameter for regularization.")
		("lambda2",
			po::value< Double >(&(this->lambda2))->default_value(1.0),
			"Lambda 2 parameter for regularization.")
		("ann-image",
			po::value< std::vector< std::string > >(&(this->ann_images))->multitoken(),
			"An image from which the texture is learned (use --ann-image-class to define the regions to learn). Multiple images can be specified. If no image is specified, the input image will be used.")
		("ann-image-class",
			po::value< std::vector< std::string > >(&(this->ann_images_classes))->multitoken(),
			"Defines a class to be learned from a binary image. At least 2 values required. If multiple images are used, they must have the same number of classes.")
		("ann-config-dir",
			po::value< std::string >(&(this->ann_config_dir))->default_value(""),
			"Directory containing the neural networks configuration files.")
		("ann-hidden-layer",
			po::value< HiddenLayerVector >(&(this->ann_hidden_layers))->multitoken()->default_value(HiddenLayerVector(1, 3)),
			"Number of neurons per hidden layer (default: one layer of 3 neurons).")
		("ann-learning-rate",
			po::value< Float >(&(this->ann_learning_rate))->default_value(0.1),
			"Learning rate of the neural networks.")
		("ann-max-epoch",
			po::value< StrictlyPositiveInteger >(&(this->ann_max_epoch))->default_value(1000),
			"Maximum number of training iterations for the neural networks.")
		("ann-mse-target",
			po::value< Float >(&(this->ann_mse_target))->default_value(0.0001),
			"Mean squared error targeted by the neural networks training algorithm.")
		("ann-validation-image",
			po::value< std::vector< std::string > >(&(this->ann_validation_images))->multitoken(),
			"The images to use to validate the training of the neural network (use --ann-validation-image-class to define associated classes). Multiple images can be specified. They mush have the same number of components per pixels than the images on which the neural network is trained.")
		("ann-validation-image-class",
			po::value< std::vector< std::string > >(&(this->ann_validation_images_classes))->multitoken(),
			"Defines the classes of the images used to validate de training of the neural network. If multiple images are used, they must have as much classes as the images on which the neural network is trained.")
		("ann-build-validation-from-training",
			po::value< Float_0_100 >(&(this->ann_validation_training_ratio))->default_value(0.0f),
			"The percentage of elements from the training-set to extract to build the validation-set.")
		;

	po::variables_map vm;

	try {
		po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

		// Handling --help before notify() in order to allow ->required()
		// http://stackoverflow.com/questions/5395503/required-and-optional-arguments-using-boost-library-program-options#answer-5517755
		if (vm.count("help")) {
			std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
			std::cout << desc;
			return EXIT;
		}

		po::notify(vm);
	} catch(po::error &err) {
		throw CliException(err.what());
	}

	this->debug = vm.count("debug");

	check_ann_parameters(vm);

	if( !this->input_image.empty() ) {
		check_regularization_parameters(vm);
	} else {
		if(this->ann_config_dir.empty()) {
			throw CliException("The directory where to save the neural networks configuration is not specified.");
		}
	}

	print_ann_parameters();

	if( !this->input_image.empty() )
		print_regularization_parameters();

	return CONTINUE;
}

const bool CliParser::get_debug() const
{
	return this->debug;
}

const std::string CliParser::get_input_image() const
{
	return this->input_image;
}

const std::string CliParser::get_export_dir() const
{
	return this->export_dir;
}

const std::string CliParser::get_region_of_interest() const
{
	return this->region_of_interest;
}

const int CliParser::get_export_interval() const
{
	return this->export_interval;
}

const int CliParser::get_num_iter() const
{
	return this->num_iter;
}

const double CliParser::get_lambda1() const {
	return this->lambda1;
}

const double CliParser::get_lambda2() const {
	return this->lambda2;
}

const std::string CliParser::get_ann_config_dir() const
{
	return this->ann_config_dir;
}

const std::vector<std::string> CliParser::get_ann_images() const
{
	return this->ann_images;
}

const std::vector<std::string> CliParser::get_ann_images_classes() const
{
	return this->ann_images_classes;
}

const std::vector< unsigned int > CliParser::get_ann_hidden_layers() const {
	return std::vector< unsigned int >(this->ann_hidden_layers.begin(), this->ann_hidden_layers.end());
}

const float CliParser::get_ann_learning_rate() const {
	return this->ann_learning_rate;
}

const unsigned int CliParser::get_ann_max_epoch() const {
	return this->ann_max_epoch;
}

const float CliParser::get_ann_mse_target() const {
	return this->ann_mse_target;
}

const std::vector<std::string> CliParser::get_ann_validation_images() const
{
	return this->ann_validation_images;
}

const std::vector<std::string> CliParser::get_ann_validation_images_classes() const
{
	return this->ann_validation_images_classes;
}

const float CliParser::get_ann_validation_training_ratio() const {
	return this->ann_validation_training_ratio;
}

/*
 * If there is no image classes, the classifier must be loaded from a stored configuration (no training will be performed):
 *     Throw an exception if no directory for the sorted configuration is provided.
 * Otherwise, (we need to find everything to train the classifiers):
 *     If there is no training image provided, we will use the input-image that will be processed.
 *         Throw an exception if there is no input-image.
 *         Set the number of images and classes.
 *     Otherwise, (at least one training-image is provided):
 *         Throw an exception if the number of training-image is not coherent with the number of classes
 *         (there must be at least two classes per image, and the total number of image-classes must
 *         be a multiple of the number of training-images).
 *
 *     If there is at least one validation-image provided:
 *         Throw an exception if there is an option asking to build the validation-set from the training-set.
 *         Throw an exception if the number of validation-classes is not coherent with the number of
 *         validation-images or if the number of validation classes is different from tne number of training-classes.
 */
void CliParser::check_ann_parameters(po::variables_map &vm) {
	if(this->ann_images_classes.empty()) {
		/*
		 * If no class is specified, the neural network
		 * must be loaded from a stored configuration.
		 */
		if(this->ann_config_dir.empty()) {
			throw CliException("You must either load the neural network from a stored configuration or specify"
			                   " the image(s) and classes to use for training.");
		}
	} else {
		/*
		 * Some classes have been provided.
		 */
		int number_of_images, number_of_classes, number_of_classes_per_image;

		if(this->ann_images.empty()) {
			/*
			 * If no learning image have been provided,
			 * we use the input image of the algorithm.
			 */
			if(this->input_image.empty()) {
				throw CliException("No image can be used to train the neural networks. You need to either specify"
				                   "a list of images to use to train the neural networks, or provide an input image"
				                   "for the algorithm.");
			}

			number_of_images            = 1;
			number_of_classes           = this->ann_images_classes.size();
			number_of_classes_per_image = number_of_classes;
		} else {
			/*
			 * We need to have the same number of classes (at least 2) for every
			 * image used for the training of the neural networks.
			 */
			number_of_images            = this->ann_images.size();
			number_of_classes           = this->ann_images_classes.size();
			number_of_classes_per_image = number_of_classes / number_of_images;

			if((number_of_classes % number_of_images != 0) || // Enough class for each image
			   (number_of_classes_per_image < 2))             // At least two class per image
			{
				throw CliException("You need to provide the same number of classes (at least two) "
						"for every image used for learning.");
			}
		}

		if(!this->ann_validation_images.empty()) {
			if(vm.count("ann-build-validation-from-training")) {
				throw CliException("You can't use the --ann-build-validation-from-training option when you specify the classes for the validation-set.");
			}

			const int number_of_validation_images = this->ann_validation_images.size(),
			          number_of_validation_classes = this->ann_validation_images_classes.size(),
			          number_of_classes_per_validation_image = number_of_validation_classes / number_of_validation_images;

			if((number_of_validation_classes % number_of_validation_images != 0) ||     // Enough class for each image
			   (number_of_classes_per_validation_image < 2) ||                          // At least two class per image
			   (number_of_classes_per_validation_image != number_of_classes_per_image)) // As many classes as the for the images used in training
			{
				throw CliException("You have an invalid number of classes for the validation images.");
			}
		}
	}
}

void CliParser::check_regularization_parameters(po::variables_map &vm) {
	if(this->export_dir.empty())
		throw CliException("You need to provide an export directory.");
}

void CliParser::print_ann_parameters() {
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	if(this->ann_images_classes.empty()) {
		if(!this->ann_config_dir.empty()) {
			LOG4CXX_INFO(logger, "The neural networks will be loaded from: " << this->ann_config_dir);
		}
	} else {
		std::vector< std::string > training_images = (this->ann_images.size() == 0 ? std::vector< std::string >(1, this->input_image) : this->ann_images);
		const int number_of_images = training_images.size();
		const int number_of_classes = this->ann_images_classes.size();
		const int number_of_classes_per_image = number_of_classes / number_of_images;

		LOG4CXX_INFO(logger, "Neural networks parameters:");
		LOG4CXX_INFO(logger, number_of_images << " images provided for training.");

		for(int j = 0; j < number_of_images; ++j) {
			LOG4CXX_INFO(logger, "\tImage " << (j + 1) << ": " << training_images[j]);

			for(int i = 0; i < number_of_classes_per_image; ++i) {
				LOG4CXX_INFO(logger, "\t\tClass " << (i + 1) << ": " << this->ann_images_classes[j * number_of_classes_per_image + i]);
			}
		}

		std::stringstream m;
		m << this->ann_hidden_layers;
		LOG4CXX_INFO(logger, "\tNumber of hidden neurons per layer: " << m.str());

		LOG4CXX_INFO(logger, "\tLearning rate: " << this->ann_learning_rate);
		LOG4CXX_INFO(logger, "\tMaximum number of iterations: " << this->ann_max_epoch.value);
		LOG4CXX_INFO(logger, "\tMean squared error targeted: " << this->ann_mse_target);
	}
}

void CliParser::print_regularization_parameters() {
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	LOG4CXX_INFO(logger, "Regularization parameters:");
	LOG4CXX_INFO(logger,    "\tInput image: "          << this->input_image);
	LOG4CXX_INFO(logger,    "\tExport directory: "     << this->export_dir);
	LOG4CXX_INFO(logger,    "\tRegion of interest: "   << this->region_of_interest);
	LOG4CXX_INFO(logger,    "\tExport interval: "      << this->export_interval);
	LOG4CXX_INFO(logger,    "\tNumber of iterations: " << this->num_iter);
	LOG4CXX_INFO(logger,    "\tLambda1: "              << this->lambda1);
	LOG4CXX_INFO(logger,    "\tLambda2: "              << this->lambda2);
}
