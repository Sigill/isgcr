#ifndef _CLI_OPTIONS_H
#define _CLI_OPTIONS_H

#include <boost/program_options.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <boost/regex.hpp>

#include "boost_program_options_types.h"

namespace po = boost::program_options;

class CliException : public std::runtime_error
{
  public:
      CliException ( const std::string &err ) : std::runtime_error (err) {}
};

class CliParser
{
public:
	enum ParseResult {
		CONTINUE = 0,
		EXIT
	};

	enum ClassifierType {
		NONE = 0,
		ANN,
		SVM
	};

	CliParser();

	/**
	 * Parses argv.
	 *
	 * \return A ParseResult if the arguments are successfully parsed.
	 *
	 * \throw A CliException if the arguments cannot be parsed.
	 */
	ParseResult parse_argv(int argc, char ** argv);

	const bool get_debug() const;

	const std::string get_input_image() const;
	const std::string get_region_of_interest() const;
	const std::string get_export_dir() const;
	const int         get_export_interval() const;
	const int         get_num_iter() const;
	const double      get_lambda1() const;
	const double      get_lambda2() const;

	const ClassifierType get_classifier_type() const;

	const std::vector< std::string >  get_classifier_training_images() const;
	const std::vector< std::string >  get_classifier_training_images_classes() const;
	const std::string                 get_classifier_config_dir() const;

	const std::vector< unsigned int > get_ann_hidden_layers() const;
	const float                       get_ann_learning_rate() const;
	const unsigned int                get_ann_max_epoch() const;
	const float                       get_ann_mse_target() const;
	const std::vector< std::string >  get_ann_validation_images() const;
	const std::vector< std::string >  get_ann_validation_images_classes() const;
	const float                       get_ann_validation_training_ratio() const;

private:
	typedef std::vector< StrictlyPositiveInteger > HiddenLayerVector;

	bool debug;

	std::string     input_image;
	std::string     region_of_interest;
	std::string     export_dir;
	PositiveInteger export_interval;
	PositiveInteger num_iter;
	Double          lambda1;
	Double          lambda2;

	ClassifierType             classifier_type;
	std::vector< std::string > classifier_training_images;
	std::vector< std::string > classifier_training_images_classes;
	std::string                classifier_config_dir;

	HiddenLayerVector           ann_hidden_layers;
	Float                       ann_learning_rate;
	StrictlyPositiveInteger     ann_max_epoch;
	Float                       ann_mse_target;
	std::vector< std::string >  ann_validation_images;
	std::vector< std::string >  ann_validation_images_classes;
	Percentage                  ann_validation_training_ratio;

	StrictlyPositiveInteger     svm_number_folds;

	void check_config_or_training_set(po::variables_map &vm);
	void check_ann_validation_set(po::variables_map &vm);
	void check_regularization_parameters(po::variables_map &vm);

	void print_classifier_parameters();
	void print_ann_parameters();
	void print_regularization_parameters();
};

#endif /* _CLI_OPTIONS_H */
