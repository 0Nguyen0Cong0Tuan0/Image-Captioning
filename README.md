# Image-Captioning

This project implements an image captioning system using deep learning techniques. The model generates descriptive captions for images by combining image feature extraction and sequence modeling.

## Project Structure

- **main.ipynb**: The main Jupyter Notebook containing the implementation of the image captioning pipeline.
- **best_model.h5**: The trained model saved after the training process.
- **features.pkl**: Pre-extracted image features stored for reuse.
- **model.png**: Visualization of the model architecture.
- **README.md**: Documentation for the project.

## Workflow

1. **Feature Extraction**:
   - Uses the VGG16 model to extract features from images.
   - Features are stored in `features.pkl` to avoid re-extraction.

2. **Caption Preprocessing**:
   - Captions are cleaned, tokenized, and mapped to their respective images.
   - Vocabulary size and maximum caption length are calculated.

3. **Model Creation**:
   - Combines image features and sequence modeling using LSTM layers.
   - The model is trained using a custom data generator.

4. **Evaluation**:
   - BLEU scores are calculated to evaluate the quality of generated captions.

5. **Caption Generation**:
   - Captions are generated for test images and compared with actual captions.

## Dependencies

The project uses the following libraries:
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- NLTK
- tqdm
- scikit-learn
- PIL (Pillow)

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open `main.ipynb` in Jupyter Notebook or VS Code.
4. Follow the cells sequentially to preprocess data, train the model, and generate captions.

## Results

The model generates captions for images and evaluates them using BLEU scores. Example results can be visualized in the notebook.

## Acknowledgments

- The dataset used is the Flickr30k dataset.
- The VGG16 model is used for feature extraction.
- BLEU scores are used for evaluation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.