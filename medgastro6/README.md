# medgastro6

A Flutter project for medical image analysis and VQA.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Dataset Setup

The project uses a large dataset that is not included in the repository due to size constraints. To set up the dataset:

1. The dataset chunks are stored in `api-folder/dataset_chunks/` and are gitignored
2. To rebuild the dataset on a new machine:
   - Run the data processing scripts in the `api-folder` directory
   - The dataset will be automatically generated in the correct location
   - Make sure you have all required Python dependencies installed

## Project Structure

- `api-folder/`: Contains the backend API and data processing code
  - `dataset_chunks/`: Directory for storing processed dataset chunks (gitignored)
  - `kvasir_data/`: Original dataset images
  - Various Python scripts for data processing and API endpoints

## Development Setup

1. Clone the repository
2. Install Flutter dependencies:
   ```bash
   flutter pub get
   ```
3. Install Python dependencies:
   ```bash
   pip install -r api-folder/requirements.txt
   ```
4. Run the data processing scripts to generate the dataset
5. Start the Flutter application:
   ```bash
   flutter run
   ```
