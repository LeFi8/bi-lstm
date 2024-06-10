import tensorflow as tf 
import tensorflow_datasets as tfds 
  
import numpy as np 
import matplotlib.pyplot as plt 

# tf.compat.v1.disable_v2_behavior() 

# Obtain the imdb review dataset from tensorflow datasets 
dataset = tfds.load('imdb_reviews', as_supervised=True) 

# Seperate test and train datasets 
train_dataset, test_dataset = dataset['train'], dataset['test'] 

# Split the test and train data into batches of 32 
# and shuffling the training set 
batch_size = 32
train_dataset = train_dataset.shuffle(10000) 
train_dataset = train_dataset.batch(batch_size) 
test_dataset = test_dataset.batch(batch_size) 


example, label = next(iter(train_dataset)) 
print('Text:\n', example.numpy()[0]) 
print('\nLabel: ', label.numpy()[0]) 


# Using the TextVectorization layer to normalize, split, and map strings 
# to integers. 
encoder = tf.keras.layers.TextVectorization(max_tokens=10000) 
encoder.adapt(train_dataset.map(lambda text, _: text)) 

# Extracting the vocabulary from the TextVectorization layer. 
vocabulary = np.array(encoder.get_vocabulary()) 

# Encoding a test example and decoding it back. 
original_text = example.numpy()[0] 
encoded_text = encoder(original_text).numpy() 
decoded_text = ' '.join(vocabulary[encoded_text]) 

print('original: ', original_text) 
print('encoded: ', encoded_text) 
print('decoded: ', decoded_text)


# Creating the model 
model = tf.keras.Sequential([ 
	encoder, 
	tf.keras.layers.Embedding( 
		len(encoder.get_vocabulary()), 64, mask_zero=True), 
	tf.keras.layers.Bidirectional( 
		tf.keras.layers.LSTM(64, return_sequences=True)), 
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)), 
	tf.keras.layers.Dense(64, activation='sigmoid'), 
	tf.keras.layers.Dense(1) 
]) 

# Summary of the model 
model.summary() 

# Compile the model 
model.compile( 
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
	optimizer=tf.keras.optimizers.Adam(), 
	metrics=['accuracy'] 
)


# Training the model and validating it on test set 
history = model.fit( 
	train_dataset, 
	epochs=5, 
	validation_data=test_dataset, 
)

# Save the model
model.save('models/bi_lstm')
# model.save('models/bi_lstm.keras')

# Plotting the accuracy and loss over time 
# Training history 
history_dict = history.history 

# Seperating validation and training accuracy 
acc = history_dict['accuracy'] 
val_acc = history_dict['val_accuracy'] 

# Seperating validation and training loss 
loss = history_dict['loss'] 
val_loss = history_dict['val_loss'] 

# Plotting 
plt.figure(figsize=(8, 4)) 
plt.subplot(1, 2, 1) 
plt.plot(acc) 
plt.plot(val_acc) 
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend(['Accuracy', 'Validation Accuracy']) 

plt.subplot(1, 2, 2) 
plt.plot(loss) 
plt.plot(val_loss) 
plt.title('Training and Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend(['Loss', 'Validation Loss']) 

plt.show()
