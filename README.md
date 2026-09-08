## Amazon Product Intelligence — Cloud, Data & Machine Learning Project

Amazon Product Intelligence is a hands-on project developed to understand how **cloud storage, data processing, exploratory data analysis, machine learning, and application development** can work together in a real-world data pipeline.

The project uses an Amazon product dataset obtained from **Kaggle** as the source data. The dataset was collected, cleaned, and processed before being stored in **Microsoft Azure Blob Storage**, giving me practical experience with cloud-based data storage and accessing data from the cloud through a connection string.

After storing the data in Azure, I performed **data cleaning and Exploratory Data Analysis (EDA)** to understand the dataset, identify patterns, examine product categories, prices, ratings, discounts, and other important attributes.

I then developed and trained **machine learning models** using the processed dataset. The project uses techniques such as **TF-IDF, Cosine Similarity, and K-Means clustering** to analyze products and generate product recommendations based on similarity.

Since the project uses a **historical/static dataset**, the trained machine learning models do not need to be retrained every time the application runs. Instead, the trained models are saved and reused by the application, while the product data is retrieved from Azure Blob Storage.

The final result is an interactive **Streamlit dashboard** that connects the different components together. Users can search and filter products using categories, price, and ratings, while the application provides AI-based recommendations from the relevant products.

Through this project, I gained practical exposure to:

* Cloud data storage using **Azure Blob Storage**
* Working with **cloud connection strings**
* Retrieving data from cloud storage
* Working with datasets obtained through **Kaggle**
* Data cleaning and preprocessing
* **Exploratory Data Analysis (EDA)**
* Understanding the basic workflow of **Machine Learning**
* Training and saving ML models
* **TF-IDF and Cosine Similarity** for recommendation
* **K-Means clustering**
* Building interactive applications using **Streamlit**
* Creating a functional and user-friendly **UI**
* Connecting data, machine learning, cloud storage, and an application into one workflow

The main objective of the project was not simply to build a recommendation system, but to gain practical understanding of how a **cloud-based data and machine learning application is built from end to end**.
