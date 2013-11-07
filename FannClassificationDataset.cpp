#include "FannClassificationDataset.h"

#include <cmath>
#include <log4cxx/logger.h>



float round(float r) {
	return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
}



FannClassificationDataset::FannClassificationDataset(ClassificationDataset<fann_type> &classificationDataset) :
	m_Container()
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	const int numberOfClasses = classificationDataset.getNumberOfClasses();

	if(0 == numberOfClasses)
		throw FannClassificationDatasetException("Cannot build a FannClassificationDataset from an empty ClassificationDataset.");

	m_FeaturesLength = classificationDataset.getInputSize();

	if(0 == m_FeaturesLength)
		throw FannClassificationDatasetException("The ClassificationDataset used to construct the FannClassificationDataset contains empty features.");

	const int numberOfClassifiers = (numberOfClasses == 2 ? 1 : numberOfClasses);

	m_Container.reserve(numberOfClassifiers);

	unsigned int total_number_of_elements = 0;
	{
		for(int i = 0; i < numberOfClasses; ++i)
		{
			const ClassificationDataset<fann_type>::Class &c = classificationDataset.getClass(i);

			if(c.empty()) {
				std::stringstream err;
				err << "The class #" << i << " is empty.";
				throw FannClassificationDatasetException(err.str());
			}

			total_number_of_elements += c.size();

			LOG4CXX_DEBUG(logger, "Class #" << i << ": " << c.size() << " elements");
		}
	}

	// Creating one data set that will be used to initialized the others
	FannDataset *training_data = fann_create_train(total_number_of_elements, m_FeaturesLength, 1);
	fann_type **training_data_input_it = training_data->input;
	fann_type **training_data_output_it = training_data->output;

	for(int i = 0; i < numberOfClasses; ++i)
	{
		const ClassificationDataset<fann_type>::Class &current_class = classificationDataset.getClass(i);

		ClassificationDataset<fann_type>::Class::const_iterator current_raw_class_it = current_class.begin(), current_raw_class_end = current_class.end();

		while(current_raw_class_it != current_raw_class_end)
		{
			// Copying the features
			std::copy(
					current_raw_class_it->begin(),
					current_raw_class_it->end(), 
					*training_data_input_it
					);

			// Don't care about the output right now
			**training_data_output_it = 0;

			++current_raw_class_it;
			++training_data_input_it;
			++training_data_output_it;
		}
	}

	// Storing the first one
	m_Container.push_back( boost::shared_ptr< FannDataset >( training_data, fann_destroy_train ) );

	// Storing copies of the first one
	for(int i = 1; i < numberOfClassifiers; ++i) {
		m_Container.push_back( boost::shared_ptr< FannDataset >( fann_duplicate_train_data(training_data), fann_destroy_train ) );
	}

	// Set the desired output of a class to 1 in the dataset representing this class
	int class_start = 0, current_class_size;
	for(int i = 0; i < numberOfClassifiers; ++i) {
		current_class_size = classificationDataset.getClass(i).size();

		fann_type *output_start = *(m_Container[i]->output) + class_start;

		std::fill(output_start, output_start + current_class_size, 1);

		class_start += current_class_size;
	}
}



FannClassificationDataset::FannClassificationDataset(Container &sets, const int featuresLength) :
	m_Container(sets), m_FeaturesLength(featuresLength)
{}



FannClassificationDataset::~FannClassificationDataset() {}



std::pair< boost::shared_ptr< FannClassificationDataset >, boost::shared_ptr< FannClassificationDataset > > 
FannClassificationDataset::split(const float ratio) const
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

	int numberOfDatasets = getNumberOfDatasets();

	Container first_sets;  first_sets.reserve(getNumberOfDatasets());
	Container second_sets; second_sets.reserve(getNumberOfDatasets());

	for(FannDatasetVector::const_iterator it = m_Container.begin(); it < m_Container.end(); ++it) {
		const unsigned int first_set_size  = fann_length_train_data(it->get()),
		                   cut             = round( first_set_size * ratio ),
		                   second_set_size = first_set_size - cut;

		if((0 == cut) || (0 == second_set_size))
			throw FannClassificationDatasetException("Cannot split this FannClassificationDataset. The ratio will ends-up generating an empty set.");

		LOG4CXX_INFO(logger, "\tFirst set will be " << cut << " elements long, second set will be " << second_set_size << " elements long.");

		FannDataset *first_subset  = fann_subset_train_data(it->get(), 0, cut),
		            *second_subset = fann_subset_train_data(it->get(), cut, second_set_size);
		// TODO Check for null

		first_sets.push_back(boost::shared_ptr< FannDataset >(first_subset, fann_destroy_train));
		second_sets.push_back(boost::shared_ptr< FannDataset >(second_subset, fann_destroy_train));
	}

	return std::make_pair(
		new FannClassificationDataset(first_sets, m_FeaturesLength),
		new FannClassificationDataset(second_sets, m_FeaturesLength)
		);
}



const int FannClassificationDataset::getNumberOfDatasets() const
{
	return m_Container.size();
}



const int FannClassificationDataset::getFeaturesLength() const
{
	return m_FeaturesLength;
}



void FannClassificationDataset::shuffle() {
	for(std::vector< boost::shared_ptr< FannDataset > >::const_iterator it = m_Container.begin(); it != m_Container.end(); ++it)
		fann_shuffle_train_data(it->get());
}



FannClassificationDataset::FannDataset* FannClassificationDataset::getSet(const int i) const
{
	return m_Container[i].get();
}
