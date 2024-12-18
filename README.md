# Dog Breed Image Classifier

## Overview
This project demonstrates the use of convolutional neural networks (CNNs) to classify images as "dogs" or "not dogs" and, for those identified as dogs, to further determine their specific breed. The application leverages pre-trained CNN models (AlexNet, VGG, and ResNet) from ImageNet to explore the trade-off between accuracy and runtime in image classification tasks.

## Project Objectives

1. **Determine the best image classification algorithm** for identifying images as "dogs" or "not dogs."
2. **Evaluate the accuracy** of the "best" algorithm in identifying specific dog breeds.
3. **Analyze the trade-off** between runtime and accuracy for each CNN architecture.

## Key Features
- Utilizes pre-trained CNN models (AlexNet, VGG, and ResNet).
- Compares algorithms based on accuracy and computational efficiency.
- Measures runtime for each model to analyze performance trade-offs.

## Algorithms
This project evaluates three CNN architectures:

1. **AlexNet**: A lightweight architecture suitable for basic classification tasks.
2. **VGG**: A deeper architecture offering improved accuracy but higher computational cost.
3. **ResNet**: An advanced architecture with residual connections, enabling high accuracy with reduced overfitting.

## Dataset
The models use pre-trained weights from the ImageNet dataset, which consists of 1.2 million images and over 1,000 classes. For this project:
- Input: Images to be classified as "dogs" or "not dogs."
- Output: Dog breed (if identified as a dog) or classification as "not a dog."

## File Structure
```
.
├── __pycache__/                   # Compiled Python files
├── pet_images/                    # Directory for input images
├── adjust_results4_isadog.py      # Adjusts classification results for 'is a dog'
├── alexnet_pet-images.txt         # AlexNet results file
├── calculates_results_stats.py    # Calculates statistics for results
├── check_images.py                # Script for image checks
├── check_images.txt               # Text file for image checks
├── classifier.py                  # Main classifier script
├── classify_images.py             # Image classification function
├── dognames.txt                   # List of dog breeds
├── get_input_args.py              # Parses input arguments
├── get_pet_labels.py              # Extracts pet labels from filenames
├── imagenet1000_clsid_to_human.txt
├── print_functions_for_lab_checks.py
├── print_results.py               # Prints classification results
├── run_models_batch.sh            # Bash script to run models in batch
├── run_models_batch_uploaded.sh
└── test_classifier.py             # Test script for classifier
```

## Implementation Details

1. **Image Preprocessing**:
   - Resize and normalize images to match the input requirements of the selected CNN model.
   - Augment images to enhance model robustness.

2. **Model Evaluation**:
   - Measure classification accuracy ("dogs" vs. "not dogs").
   - Measure breed classification accuracy for identified dogs.
   - Record runtime for each model.

3. **Performance Metrics**:
   - **Accuracy**: Percentage of correct predictions.
   - **Runtime**: Total time taken to classify a batch of images.

## Results
- Comparative analysis of AlexNet, VGG, and ResNet based on accuracy and runtime.
- Insights into the trade-off between computational cost and model performance.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/dog-breed-classifier.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the classifier:
   ```bash
   python classifier.py --input_dir path_to_images --model {alexnet, vgg, resnet}
   ```
4. Test the classifier:
   ```bash
   python test_classifier.py
   ```

## Dependencies
- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Matplotlib

## Future Improvements
- Integrate additional CNN architectures for comparison.
- Explore real-time classification using lightweight models.
- Develop a user-friendly GUI for non-technical users.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [ImageNet Dataset](http://www.image-net.org/)
- PyTorch library for deep learning models

#Dana Hmeed

---
