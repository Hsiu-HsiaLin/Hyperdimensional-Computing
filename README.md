# Hyperdimensional computing for image similarity analysis

## Overview
This project aims to evaluate the similarity between images by analyzing image data. It involves loading images from a dataset, encoding images into a hyperdimensional vector, calculating similarities between images, clustering the similarity results using the K-means algorithm, and finally adjusting the similarity calculations based on predefined grouping information.

## Installation
This project requires `numpy`, `opencv-python`, and `sklearn`. Dependencies can be installed using the `requirements.txt` file included in the project. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Running the Project

1. Ensure all dependencies are installed.
2. Prepare the dataset in the expected format for `hd.py`.
3. Run `hd.py` to generate similarities.
4. Run `kmeans.py` to cluster the similarities.
5. Run `abs_sim.py` to adjust the similarities based on predefined groups.

## File Descriptions
- `hd.py`: Handles loading the image dataset, encoding images into a hyperdimensional vector, calculating, and saving the similarities between images.
- `kmeans.py`: Uses similarity data to perform K-means clustering analysis and saves the normalized clustering results.
- `abs_sim.py`: Adjusts the similarity values in the clusters based on predefined group information and saves the final results.
