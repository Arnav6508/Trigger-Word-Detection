import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv
load_dotenv()

from inference import model_inference
from utils import convert_to_wav

# --- App Configuration ---
APP_TITLE = "Trigger Word Detection"
APP_DESCRIPTION = "This project detects the trigger word 'activate' in audio input and plays a chime sound."
chime = os.getenv('chime_file')

# --- Gradio Interface Function (Wrapper) ---
def run_model_on_audio(input_audio_path):
    if input_audio_path is None:
        return None, "Please upload a .wav file first."

    status_message = ""
    
    try:
        print(f"Gradio input audio path: {input_audio_path}")
        
        # Convert audio file to wav format
        input_audio_path_wav = convert_to_wav(input_audio_path)

        # Perform audio inference
        output_audio_file, plot_fig = model_inference(input_audio_path_wav, chime)
        
        if output_audio_file and os.path.exists(output_audio_file):
            status_message = "Processing successful!"
            print(f"Returning output file: {output_audio_file}")
            if plot_fig is None:
                status_message += " (Plot generation failed or not available)"
            print(f"Returning output audio: {output_audio_file}, Plot: {'Exists' if plot_fig else 'None'}")

        else: 
            status_message = "Model did not produce an output file or an error occurred internally."
            print("Model function returned None or an empty path.")
            output_audio_file = None

    except Exception as e:
        status_message = f"An unexpected error occurred: {str(e)}"
        print(status_message)

    if not isinstance(plot_fig, plt.Figure) and plot_fig is not None:
        print(f"Warning: plot_fig is not a Figure object before final return. Type: {type(plot_fig)}. Setting to None.")
        if hasattr(plot_fig, 'close') and callable(getattr(plot_fig, 'close')): plt.close(plot_fig) # Try to close if it's a figure
        plot_fig = None

    return output_audio_file, plot_fig, status_message

# --- Gradio Interface Definition ---
iface = gr.Interface(
    fn=run_model_on_audio,
    inputs=gr.Audio(
        type="filepath", # Passes the audio as a file path
        label="Upload your .wav file",
        sources=["upload", "microphone"] # Restrict to file uploads as per your model's nature
    ),
    outputs=[
        gr.Audio(type="filepath", label="Processed Audio Output"),
        gr.Plot(label="Analysis Plots"),
        gr.Textbox(label="Status") # To display messages
    ],
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    flagging_options=None
)


# --- Run the App ---
if __name__ == "__main__":
    print("Starting application...")
    iface.launch(share = True)
    print("App launched. Check your browser for the provided link.")