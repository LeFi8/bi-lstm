import tensorflow as tf 
  
import numpy as np 

# Load the model
loaded_model = tf.keras.models.load_model('models/bi_lstm')

loaded_model.summary()

user_input = input('Enter the review: ')

while user_input != 'exit':
	# Making predictions 
	sample_text = (
		'''The movie by GeeksforGeeks was so good and the animation are so dope. 
		I would recommend my friends to watch it.'''
	)
	predictions = loaded_model.predict(np.array([sample_text]))
	print(*predictions[0])

	# Print the label based on the prediction 
	if predictions[0] > 0:
		print('The review is positive')
	else:
		print('The review is negative')
	
	user_input = input('Enter the review: ')
