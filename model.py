import tensorflow as tf
import numpy as np

class Model:
    def __init__(self):
        # Load the models
        self.model = tf.keras.models.load_model('model.keras')
        # Load the Tokenizer
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open('tokenizer.json').read())
        
    def predict(self, seed_text, next_words=100):
        seed_text = seed_text
        next_words = next_words
        for _ in range(next_words):
            # Tokenize the seed text
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=29, padding='pre')#30 is the max length of the sequence in training
            # Predict the next word
            predicted = self.model.predict(token_list, verbose=0)
            # Get the indices of the words with the top 2 highest probabilities
            top_indices = np.argpartition(-predicted, 2)[0][:2]
            # Get the probabilities of the top 3 words
            top_probs = predicted[0][top_indices]
            # Normalize the probabilities so they sum to 1
            top_probs = top_probs / np.sum(top_probs)
            # Randomly select one of the top 3 words, with higher probability words more likely to be selected
            chosen_index = np.random.choice(top_indices, p=top_probs)
            # Convert the index to the word
            output_word = self.tokenizer.index_word[chosen_index]
            seed_text += " " + output_word
        
        return seed_text 

# Test the model
if __name__ == '__main__':
    model = Model()
    generated_text = model.predict("I am", 50)
    print(generated_text)