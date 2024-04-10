import tensorflow as tf
import numpy as np
import os

class Model:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, 'model')
        self.model = tf.keras.models.load_model(model_path)
        tokenizer_path = os.path.join(dir_path, 'tokenizer.json')
        with open(tokenizer_path) as f:
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
        
    def predict(self, seed_text, next_words=100):
        seed_text = seed_text
        next_words = next_words
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=29, padding='pre')#30 is the max length of the sequence in training
            predicted = self.model.predict(token_list, verbose=0)
            top_indices = np.argpartition(-predicted, 2)[0][:2]
            top_probs = predicted[0][top_indices]
            top_probs = top_probs / np.sum(top_probs)
            chosen_index = np.random.choice(top_indices, p=top_probs)
            output_word = self.tokenizer.index_word[chosen_index]
            seed_text += " " + output_word
        
        return seed_text 

# Test the model
if __name__ == '__main__':
    model = Model()
    generated_text = model.predict("I am", 50)
    print(generated_text)