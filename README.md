md
# Deep-Learning-Project-using-Autoencoder-CNN-LSTM

This project implements a deep learning model for emotion recognition from video using a combination of Autoencoder, CNN, LSTM, and Attention mechanisms.  It includes a Flask server for making predictions and a Streamlit application for real-time video emotion recognition.

## Key Features & Benefits

*   **Emotion Recognition from Video:** Accurately identifies emotions present in video sequences.
*   **Hybrid Model Architecture:** Leverages the strengths of Autoencoders, CNNs, LSTMs, and Attention mechanisms for enhanced performance.
*   **Flask Server:** Provides an API endpoint for making predictions from video data.
*   **Streamlit Application:** Offers a user-friendly interface for real-time video emotion recognition using a webcam.

## Prerequisites & Dependencies

Before running this project, ensure you have the following installed:

*   **Python 3.7+**
*   **PyTorch:** `torch`
*   **Torchvision:** `torchvision`
*   **Flask:** `flask`
*   **Streamlit:** `streamlit`
*   **PIL (Pillow):** `PIL`
*   **OpenCV:** `cv2`
*   **NumPy:** `numpy`

You can install these dependencies using pip:

```bash
pip install torch torchvision flask streamlit Pillow opencv-python numpy
```

## Installation & Setup Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/dhia-eddine-jedidi/Deep-Learning-Project-using-Autoencoder-CNN-LSTM.git
    cd Deep-Learning-Project-using-Autoencoder-CNN-LSTM
    ```

2.  **Download the `best_emotion_model.pt` file**.

    While the provided file list shows it exists, ensure that it's present. If missing and a link is not available replace it with a model trained similarly.

3.  **Optional: Download or prepare the `cremad_videos` dataset.**

    While the repository contains a structure for `cremad_videos`, you need to download or provide the actual video files into this directory.

## Usage Examples

### 1. Flask Server:

1.  Run the `Flask_server.py` file:

    ```bash
    python Flask_server.py
    ```

    This will start the Flask server, which listens for POST requests with video data.

2.  **API Usage (Example):**

    Send a POST request to the server's endpoint (e.g., `http://127.0.0.1:5000/predict`) with the video data in the request body. The server will return a JSON response containing the predicted emotion.

    *Implementation details on how to send the video data is not fully described in the files provided, a more complex implementation might be necessary.*

### 2. Streamlit Application:

1.  Run the `streamlit_app.py` file:

    ```bash
    streamlit run streamlit_app.py
    ```

    This will open the Streamlit application in your web browser.

2.  **Real-time Emotion Recognition:**

    The Streamlit application will access your webcam and display the predicted emotion in real-time.  Adjust the `PRED_INTERVAL` in `streamlit_app.py` to control the prediction frequency.

### 3. Notebooks
1. Open `3D_CNN.ipynb` or `CNN+AutoEncoder+LSTM+Attention.ipynb` notebooks and execute them to train or experiment with different models.

## Configuration Options

*   **`MODEL_PATH`:** Specifies the path to the trained emotion recognition model (`best_emotion_model.pt`). You can change this if you have a different model file.
*   **`SEQ_LEN`:** Defines the number of frames used as input to the LSTM model.  This value should match the sequence length used during training.
*   **`IMG_SIZE`:** Specifies the size to which input frames are resized.  Ensure this matches the size used during training (e.g., 224x224).
*   **`DEVICE`:** Determines the device (CPU or GPU) used for inference.  The code automatically detects if a GPU is available and uses it if so.
* `PRED_INTERVAL` (Streamlit Application): The time interval between predictions in seconds. Adjust this value to control the prediction frequency. The default is 2.0 seconds.
* Environment variables can be set for the above config options to ensure reproducibility.

## Contributing Guidelines

Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

1.  **Fork the repository.**
2.  **Create a new branch for your feature or bug fix.**
3.  **Make your changes and commit them with clear, concise commit messages.**
4.  **Submit a pull request to the main branch.**

## License Information

License not specified. Please contact the repository owner for license information.

## Acknowledgments

*   This project utilizes several open-source libraries, including PyTorch, Flask, and Streamlit. We gratefully acknowledge the contributions of the developers of these libraries.
*   CREMA-D dataset for video emotion data (if used).
