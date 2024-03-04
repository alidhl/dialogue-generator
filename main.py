from model import Model
import gradio as gr

model = Model()

interface = gr.Interface(fn=model.predict,
                         inputs=[gr.Textbox(label="Starting Sentence"),
                                 gr.Number(label="Number of Words", maximum=1000, minimum=1)],
                         outputs="text",
                         title="Elden Ring Dialogue Generator",
                         description='Enter a sentence to start the dialogue. The model will predict the next words.',)
interface.launch()