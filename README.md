# Recommendation system for museum exhibits

The diploma thesis titled «Development of an Intelligent Recommendation System for Museum Exhibits» focuses on the design and development of a recommendation system aimed at personalizing the user experience in digital museum environments. Special attention is given to the use of image embeddings and machine learning methods for analyzing visual characteristics of artworks. During the study, modern approaches in recommender systems and computer vision were analyzed. A key feature of the developed system is the use of the CLIP model for generating embeddings and computing similarity between artworks, which enables the formation of personalized and visually relevant recommendations.

## Technologies

- Python
- CLIP 
- Streamlit
- PostgreSQL
- FastAPI

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

## Author

Inkara Suleimenova
