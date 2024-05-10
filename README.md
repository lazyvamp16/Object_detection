# Lost Object Detection with Machine Learning

Losing or misplacing items at school, home, or in other spaces can be frustrating. This project aims to leverage machine learning and CCTV camera snapshots to help locate missing objects efficiently.

## Overview

This project utilizes Convolutional Neural Networks (CNNs) implemented with TensorFlow's Sequential model to analyze images from CCTV footage. The goal is to automatically detect and recognize objects like keys, mobile phones, spectacles, sports equipment (e.g., cricket balls, water bottles), and more within these images.

## Key Features

- **Object Detection**: The CNN model is trained to identify specific objects within images.
- **Snapshot Analysis**: CCTV camera snapshots are used as input data for the machine learning model.
- **Database Integration**: Results from the object detection process are stored in SQL database tables for easy retrieval and tracing.

## Workflow

1. **Data Collection**: Gather snapshots from CCTV cameras containing scenes where objects might be lost or misplaced.
2. **Preprocessing**: Prepare the image data for training the CNN model.
3. **Model Training**: Use TensorFlow's Sequential API to train the CNN on a labeled dataset of object images.
4. **Object Detection**: Apply the trained model to new CCTV snapshots to detect objects.
5. **Database Storage**: Store the detection results in an SQL database for future reference.

## Technologies Used

- TensorFlow (Sequential API)
- Python (for data preprocessing, model training, and integration with SQL)
- SQL Database (for storing and querying detection results)

## Usage

To run the project:

1. Ensure you have Python installed along with TensorFlow and required dependencies.
2. Set up your SQL database and update database credentials in the Python scripts.
3. Train the CNN model using prepared datasets.
4. Run the object detection script on CCTV snapshots to populate your database with detection results.

## Future Improvements

- Implement real-time object detection using live CCTV feeds.
- Enhance model accuracy through additional training and data augmentation techniques.
