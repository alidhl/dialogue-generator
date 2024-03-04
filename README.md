# Elden Ring Dialogue Generator

This project is a Recurrent Neural Network (RNN) based dialogue generator, inspired by and trained on dialogue from the game Elden Ring. Utilizing TensorFlow, the model learns patterns and styles from the game's unique dialogue to generate new, game-like conversations. A Gradio interface is integrated to provide an easy and interactive way for users to generate and interact with new dialogues.

## Sample Screenshots
![Screenshot 2024-03-04 211622](https://github.com/alidhl/elden-ring-dialogue-generator/assets/119793124/abc44a5d-1e4f-4504-92eb-ac913a3d9562)
![Screenshot 2024-03-04 211435](https://github.com/alidhl/elden-ring-dialogue-generator/assets/119793124/7ba615f9-cd25-450a-bfe2-d913f809cdac)
![Screenshot 2024-03-04 211519](https://github.com/alidhl/elden-ring-dialogue-generator/assets/119793124/fa014c66-790b-44b6-bb50-d678dd9a3e9b)

## Installation

To set up this project, follow these steps:

1. Clone the repository:
2. Install the required dependencies:
 ```bash
 pip install -r requirements.txt
 ```
## Usage
To run the dialogue generator:

1. Run Main.py
2. Open the provided link in your web browser.
3. Enter your seed text to generate random dialogues.

## Model
The RNN model was built using TensorFlow, particularly leveraging its capabilities for sequence data to model the flow and structure of Elden Ring's dialogue. The model architecture consists of LSTM layers which are well-suited for learning from long sequences of data.

## Performance
The model's performance varies based on the complexity of the input and the training data coverage. Initial tests have shown promising results, with the model able to generate dialogues that hold thematic and stylistic resemblance to Elden Ring's original script. However, it is important to note that the generated dialogues may sometimes lack context or deviate in unexpected ways due to the inherent unpredictability of  recurrent neural network-based text generation.
