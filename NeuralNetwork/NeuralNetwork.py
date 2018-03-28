import tensorflow as tf
import DataParser

batch_size=100

def main():
	# Fetch the data
	(train_x, train_y), (test_x, test_y) = DataParser.load_data()
	
	# Feature columns describe how to use the input
	my_feature_columns = []
	for key in train_x.keys():
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))
		
	# Build 2 hidden layer DNN with 10, 10 units respectively
	classifier = tf.estimator.DNNClassifier(
		feature_columns=my_feature_columns,
		# Two hidden layers of 10 nodes each
		hidden_units=[10, 10, 10, 10],
		# The model must choose between 3 classes
		n_classes=3)
		
	# Train the model
	classifier.train(
		input_fn=lambda:DataParser.train_input_fn(train_x, train_y, batch_size),
		steps=1000)
	
	# Evaluate the model
	eval_result = classifier.evaluate(
		input_fn=lambda:DataParser.eval_input_fn(test_x, test_y, batch_size))
		
	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	
	# Generate predictions from the model
	expected = ['Setosa', 'Versicolor', 'Virginica']
	predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
	
	predictions = classifier.predict(
		input_fn=lambda:DataParser.eval_input_fn(predict_x, labels=None, batch_size=batch_size))
		
	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]

		print(template.format(DataParser.SPECIES[class_id], 100 * probability, expec))
	
if __name__ == '__main__':
	main()