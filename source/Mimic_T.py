from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(1234)

import pickle
import tensorflow_datasets as tfds
import os
import re
import numpy as np
import h5py


def create_padding_mask(x):
	mask = tf.cast(tf.math.equal(x, 0), tf.float32)
	return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
	seq_len = tf.shape(x)[1]
	look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
	padding_mask = create_padding_mask(x)
	return tf.maximum(look_ahead_mask, padding_mask)

def scaled_dot_product_attention(query, key, value, mask):
	matmul_qk = tf.matmul(query, key, transpose_b=True)
	depth = tf.cast(tf.shape(key)[-1], tf.float32)
	logits = matmul_qk / tf.math.sqrt(depth)

	if mask is not None:
		logits += (mask * -1e9)

	attention_weights = tf.nn.softmax(logits, axis=-1)
	output = tf.matmul(attention_weights, value)

	return output

class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, name="multi_head_attention"):
		super(MultiHeadAttention, self).__init__(name=name)
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.query_dense = tf.keras.layers.Dense(units=d_model)
		self.key_dense = tf.keras.layers.Dense(units=d_model)
		self.value_dense = tf.keras.layers.Dense(units=d_model)

		self.dense = tf.keras.layers.Dense(units=d_model)

	def split_heads(self, inputs, batch_size):
		inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(inputs, perm=[0, 2, 1, 3])

	def call(self, inputs):
		query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
		batch_size = tf.shape(query)[0]

		# linear layers
		query = self.query_dense(query)
		key = self.key_dense(key)
		value = self.value_dense(value)

		# split heads
		query = self.split_heads(query, batch_size)
		key = self.split_heads(key, batch_size)
		value = self.split_heads(value, batch_size)

		# scaled dot-product attention
		scaled_attention = scaled_dot_product_attention(query, key, value, mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

		# concatenation of heads
		concat_attention = tf.reshape(scaled_attention,
		                              (batch_size, -1, self.d_model))

		# final linear layer
		outputs = self.dense(concat_attention)

		return outputs

class PositionalEncoding(tf.keras.layers.Layer):

	def __init__(self, position, d_model):
		super(PositionalEncoding, self).__init__()
		self.pos_encoding = self.positional_encoding(position, d_model)

	def get_angles(self, position, i, d_model):
		angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
		return position * angles

	def positional_encoding(self, position, d_model):
		angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],d_model=d_model)

		sines = tf.math.sin(angle_rads[:, 0::2])

		cosines = tf.math.cos(angle_rads[:, 1::2])

		pos_encoding = tf.concat([sines, cosines], axis=-1)
		pos_encoding = pos_encoding[tf.newaxis, ...]
		return tf.cast(pos_encoding, tf.float32)

	def call(self, inputs):
		return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
	inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
	padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

	attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs,'key': inputs,'value': inputs,'mask': padding_mask})
	attention = tf.keras.layers.Dropout(rate=dropout)(attention)
	attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

	outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
	outputs = tf.keras.layers.Dense(units=d_model)(outputs)
	outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
	outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

	return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
	inputs = tf.keras.Input(shape=(None,), name="inputs")
	padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

	embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
	embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
	embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

	outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

	for i in range(num_layers):
		outputs = encoder_layer(
		    units=units,
		    d_model=d_model,
		    num_heads=num_heads,
		    dropout=dropout,
		    name="encoder_layer_{}".format(i),
		)([outputs, padding_mask])

	return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
	inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
	enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
	look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
	padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

	attention1 = MultiHeadAttention(
	  d_model, num_heads, name="attention_1")(inputs={
	      'query': inputs,
	      'key': inputs,
	      'value': inputs,
	      'mask': look_ahead_mask
	  })
	attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

	attention2 = MultiHeadAttention(
	  d_model, num_heads, name="attention_2")(inputs={
	      'query': attention1,
	      'key': enc_outputs,
	      'value': enc_outputs,
	      'mask': padding_mask
	  })
	attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
	attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

	outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
	outputs = tf.keras.layers.Dense(units=d_model)(outputs)
	outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
	outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

	return tf.keras.Model(
	  inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
	  outputs=outputs,
	  name=name)

def decoder(vocab_size,num_layers,units,d_model,num_heads,dropout,name='decoder'):
	inputs = tf.keras.Input(shape=(None,), name='inputs')
	enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
	look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
	padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

	embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
	embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
	embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

	outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

	for i in range(num_layers):
		outputs = decoder_layer(
		    units=units,
		    d_model=d_model,
		    num_heads=num_heads,
		    dropout=dropout,
		    name='decoder_layer_{}'.format(i),
		)(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

	return tf.keras.Model(
	  inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
	  outputs=outputs,
	  name=name)

def transformer(vocab_size,num_layers,units,d_model,num_heads,dropout,name="transformer"):
	inputs = tf.keras.Input(shape=(None,), name="inputs")
	dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

	enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='enc_padding_mask')(inputs)

	look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask,output_shape=(1, None, None),name='look_ahead_mask')(dec_inputs)

	dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),name='dec_padding_mask')(inputs)

	enc_outputs = encoder(vocab_size=vocab_size,num_layers=num_layers,units=units,d_model=d_model,num_heads=num_heads,dropout=dropout,)(inputs=[inputs, enc_padding_mask])

	dec_outputs = decoder(vocab_size=vocab_size,num_layers=num_layers,units=units,d_model=d_model,num_heads=num_heads,dropout=dropout,)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

	outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

	return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps**-1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Mimic_T:

	def __init__(self, model=None, MAX_LENGTH=None):
		self.model = model
		self.MAX_LENGTH = 40
		self.tokenizer = None
		self.VOCAB_SIZE = 0
		self.dataset = None
		self.BATCH_SIZE = 64
		self.BUFFER_SIZE = 20000
		self.NUM_LAYERS = 2
		self.D_MODEL = 256
		self.NUM_HEADS = 8
		self.UNITS = 512
		self.DROPOUT = 0.1
		self.EPOCHS = 1
		self.START_TOKEN = None
		self.END_TOKEN = None
		self.questions = pickle.load(open("joeyData/joeyInputTrain.pkl", "rb")) + pickle.load(open("joeyData/joeyInputTest.pkl", "rb"))
		self.answers = pickle.load(open("joeyData/joeyOutputTrain.pkl", "rb")) + pickle.load(open("joeyData/joeyOutputTest.pkl", "rb"))

	def preprocess_sentence(self,sentence):
		sentence = sentence.lower().strip()
		sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
		sentence = re.sub(r'[" "]+', " ", sentence)
		sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
		sentence = sentence.strip()

		return sentence

	def createtokenizer(self):
		self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(self.questions + self.answers, target_vocab_size=2**13)
		self.START_TOKEN, self.END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
		self.VOCAB_SIZE = self.tokenizer.vocab_size + 2

	def tokenizeData(self):
		tokenized_inputs, tokenized_outputs = [], []
		inputs = self.questions
		outputs = self.answers

		for (sentence1, sentence2) in zip(inputs, outputs):
			# tokenize sentence
			sentence1 = self.START_TOKEN + self.tokenizer.encode(sentence1) + self.END_TOKEN
			sentence2 = self.START_TOKEN + self.tokenizer.encode(sentence2) + self.END_TOKEN
			# check tokenized sentence max length
			if len(sentence1) <= self.MAX_LENGTH and len(sentence2) <= self.MAX_LENGTH:
			  tokenized_inputs.append(sentence1)
			  tokenized_outputs.append(sentence2)

		# pad tokenized sentences
		tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=self.MAX_LENGTH, padding='post')
		tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=self.MAX_LENGTH, padding='post')

		self.questions, self.answers = tokenized_inputs, tokenized_outputs

	def createDataset(self):
		self.dataset = tf.data.Dataset.from_tensor_slices(({'inputs': self.questions,'dec_inputs': self.answers[:, :-1]},{'outputs': self.answers[:, 1:]},))
		self.dataset = self.dataset.cache()
		self.dataset = self.dataset.shuffle(self.BUFFER_SIZE)
		self.dataset = self.dataset.batch(self.BATCH_SIZE)
		self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

	def createModel(self):
		self.model = transformer(vocab_size=self.VOCAB_SIZE,num_layers=self.NUM_LAYERS,units=self.UNITS,d_model=self.D_MODEL,num_heads=self.NUM_HEADS,dropout=self.DROPOUT)

	def loss_function(self, y_true, y_pred):
		y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))

		loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

		mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
		loss = tf.multiply(loss, mask)

		return tf.reduce_mean(loss)

	def accuracy(self, y_true, y_pred):
		y_true = tf.reshape(y_true, shape=(-1, self.MAX_LENGTH - 1))
		return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

	def buildModel(self):
		learning_rate = CustomSchedule(self.D_MODEL)
		optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
		self.model.compile(optimizer=optimizer, loss=self.loss_function, metrics=[self.accuracy])

	def train(self):
		checkpoint_path = "checkpoints/weights2/cp.ckpt"
		checkpoint_dir = os.path.dirname(checkpoint_path)
		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

		self.model.fit(self.dataset, epochs=self.EPOCHS, callbacks=[cp_callback])

	def loadWeights(self, path):
		#self.model.load_weights("checkpoints/joey/cp.ckpt")
		self.model.load_weights(path)

	def evaluate(self,sentence):
		sentence = self.preprocess_sentence(sentence)
		sentence = tf.expand_dims(self.START_TOKEN + self.tokenizer.encode(sentence) + self.END_TOKEN, axis=0)
		output = tf.expand_dims(self.START_TOKEN, 0)
		for i in range(self.MAX_LENGTH):
			predictions = self.model(inputs=[sentence, output], training=False)
			predictions = predictions[:, -1:, :]
			predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
			if tf.equal(predicted_id, self.END_TOKEN[0]):
			  break
			output = tf.concat([output, predicted_id], axis=-1)

		return tf.squeeze(output, axis=0)

	def predict(self,sentence):
		prediction = self.evaluate(sentence)
		predicted_sentence = self.tokenizer.decode(
		  [i for i in prediction if i < self.tokenizer.vocab_size])
		return predicted_sentence

	def processModel(self):
		self.createtokenizer()
		self.tokenizeData()
		self.createDataset()
		self.createModel()
		self.buildModel()

class Mimic_T_Preprocessor:
	def cleanText(self, sentence):
		sentence = sentence.lower().strip()
		sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
		sentence = re.sub(r'[" "]+', " ", sentence)
		sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
		sentence = sentence.strip()
		return sentence

	def cleanTexts(self, sentences):
		return [self.cleanText(sentence) for sentence in sentences]

class Mimic_Trans:
	def __init__(self, preProcessor, mimic_t):
		self.preProcessor = preProcessor
		self.model = mimic_t

	def chat(self, sentence):
		return self.model.predict(sentence)

	@classmethod
	def load(self, loadFile):
		model = Mimic_T(Mimic_T_Preprocessor())
		model.processModel()
		model.loadWeights(loadFile)
		return Mimic_Trans(Mimic_T_Preprocessor(), model)
