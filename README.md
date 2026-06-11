# Recommendation system for museum exhibits

The diploma thesis titled «Development of an Intelligent Recommendation System for Museum Exhibits» focuses on the design and development of a recommendation system aimed at personalizing the user experience in digital museum environments. Special attention is given to the use of image embeddings and machine learning methods for analyzing visual characteristics of artworks. During the study, modern approaches in recommender systems and computer vision were analyzed. A key feature of the developed system is the use of the CLIP model for generating embeddings and computing similarity between artworks, which enables the formation of personalized and visually relevant recommendations.

## Repository Structure

- thesis/ – diploma thesis
- presentation/ – defense presentation
- code/ – source code
- data/ – scripts for dataset and embedding
- images/ - system architecture, ER diagram, workflow schemes, and results

## Dataset
The project uses the WikiArt dataset from Kaggle:

Dataset: https://www.kaggle.com/datasets/steubk/wikiart

For the experiments, a subset of artworks from the following artistic styles was selected:

- Art Nouveau (Modern)
- Baroque
- Cubism
- Early Renaissance
- Expressionism
- Fauvism
- High Renaissance
- Impressionism
- Mannerism (Late Renaissance)
- Pointillism
- Pop Art
- Post-Impressionism
- Realism
- Rococo
- Romanticism

The selected styles provide significant visual diversity and allow evaluation of the recommendation system across different artistic movements.

## Model Selection: CLIP vs ResNet50

An alternative approach would be to use CNNs such as ResNet50 for extracting image features. However, CNNs are primarily trained for classification tasks and tend to capture low- and mid-level visual attributes like texture, shape, and object identity — which limits their ability to reflect the semantic closeness between artworks. As a result, recommendations based on ResNet embeddings may feel less meaningful to the end user.

This project uses **CLIP** due to its high-level semantic understanding and strong performance in similarity-based recommendations.

### Table — Comparison of ResNet50 and CLIP

| Criterion | CLIP | ResNet50 |
|---|---|---|
| Model Type | Multimodal (image + text) | Convolutional Neural Network (CNN) |
| Primary Task | Image–text matching | Image classification |
| Training Type | Trained on image–text pairs | Trained on labeled images |
| Context Understanding |  Yes (captures meaning and style) |  Limited |
| Feature Level | High-level (semantic) | Low- and mid-level (shapes, textures) |
| Use in Recommendations |  Suitable | Limited suitability |
| Zero-shot Capability | Yes | No |
| Similarity Quality | High | Lower |

## Key Features

- Content-based recommendation system
- CLIP image embeddings
- Artwork similarity search
- FastAPI backend
- PostgreSQL database
- Streamlit web interface

## Architecture
![Architecture](images/Web%20Application%20Architecture.png)

## Database Schema
![ER Diagram](images/ER%20Diagram.png)

## How It Works

### Scenario 1 — Similar Artwork Search
![Scenario 1](images/Scheme%20of%20first%20scenario%20of%20recommendation%20system.%20Similar%20Artwork%20Search.png)

A user uploads an image → CLIP generates an embedding → 
the system finds the most visually similar artworks in the database.

### Scenario 2 — Personalized Recommendations
![Scenario 2](images/Scheme%20of%20second%20scenario%20of%20recommendation%20system.%20Personalized%20Recommendations.png)

Based on the user's favorites, the system builds a preference 
profile and recommends artworks from similar styles and periods.

## Results

### Onboarding Page
![Onboarding](images/Onboarding%20Page.png)

### Search by Image
![Search](images/Search%20by%20Image%20Page%20result.png)

### Personal Recommendations
![Recommendations](images/Personal%20Recommendation%20Page%20Result.png)

## Technologies

- Python
- CLIP 
- Streamlit
- PostgreSQL
- FastAPI

## Author

Inkara Suleimenova
