# Recommendation system for museum exhibits

The diploma thesis titled «Development of an Intelligent Recommendation System for Museum Exhibits» focuses on the design and development of a recommendation system aimed at personalizing the user experience in digital museum environments. Special attention is given to the use of image embeddings and machine learning methods for analyzing visual characteristics of artworks. During the study, modern approaches in recommender systems and computer vision were analyzed. A key feature of the developed system is the use of the CLIP model for generating embeddings and computing similarity between artworks, which enables the formation of personalized and visually relevant recommendations.

## Business Problem

Digital museums often contain thousands of artworks, making it difficult for visitors to discover exhibits aligned with their interests. Traditional browsing methods rely on manual filtering, search, or curator-defined categories.

This project addresses this challenge by automatically recommending artworks based on visual similarity and user preferences, helping visitors explore collections in a more personalized and engaging way.

## Project Objectives

- Develop a content-based recommendation system for museum exhibits.
- Apply computer vision techniques to artwork analysis.
- Explore the effectiveness of image embeddings for similarity search.
- Build a complete end-to-end recommendation platform.
- Improve artwork discovery and user engagement in digital museum environments.

## Key Features

- Content-based recommendation system
- CLIP image embeddings
- Artwork similarity search
- FastAPI backend
- PostgreSQL database
- Streamlit web interface

## Architecture
![Architecture](images/Web%20Application%20Architecture.png)

The system consists of the following components:

- Image processing and embedding generation using CLIP.
- PostgreSQL database for storing artworks, users, and embeddings.
- Recommendation engine for similarity search and personalization.
- FastAPI backend providing API endpoints.
- Streamlit frontend for user interaction.

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

## Database Schema
![ER Diagram](images/ER%20Diagram.png)

## Recommendation Scenarios

### Scenario 1 - Similar Artwork Search
![Scenario 1](images/Scheme%20of%20first%20scenario%20of%20recommendation%20system.%20Similar%20Artwork%20Search.png)

A user uploads an artwork image.

The system:

1. Generates a CLIP embedding.
2. Searches for nearest neighbors in the embedding space.
3. Returns visually similar artworks from the database.

This functionality enables intuitive artwork discovery based purely on visual characteristics.

### Scenario 2 - Personalized Recommendations
![Scenario 2](images/Scheme%20of%20second%20scenario%20of%20recommendation%20system.%20Personalized%20Recommendations.png)

The system analyzes artworks marked as favorites by the user.

Based on these preferences:

1. A user profile embedding is created.
2. Similar artworks are identified.
3. Personalized recommendations are generated.

This approach allows recommendations to adapt to individual artistic preferences.

## Results

### Onboarding Page
![Onboarding](images/Onboarding%20Page.png)

### Search by Image
![Search](images/Search%20by%20Image%20Page%20result.png)

### Personal Recommendations
![Recommendations](images/Personal%20Recommendation%20Page%20Result.png)

## Repository Structure

- thesis/ – diploma thesis
- presentation/ – defense presentation
- code/ – source code
- data/ – scripts for dataset and embedding
- images/ - system architecture, ER diagram, workflow schemes, and results


## Technologies

- Python
- CLIP 
- Streamlit
- PostgreSQL
- FastAPI

## Author

Inkara Suleimenova
