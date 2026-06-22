# Live New York Times Analytics Portal and Predictive Trend Forecast

A production-grade web application that connects directly to the New York Times API to ingest, clean, and visualize live news headline trends in real-time. The portal features a custom statistical language suite alongside a machine learning pipeline designed to track historical daily occurrences and project public interest indicators 30 days into the future.

## Live Demo
Experience the application live: [Click here](https://nlp-nyt-headline-analytics-dnvbw26pkneajb7wq9lb6r.streamlit.app)

## Core Features and Engineering Pillars
* **Live API Streaming Architecture:** Bypasses static datasets to pull real-time historical data payloads directly via the official New York Times Archive API gateway.
* **Natural Language Text Processing Pipeline:** Converts raw text strings into structured, analytical data frames using automated casing normalization, tokenization, and custom-filtered alphanumeric stop-word removal.
* **Daily Keyword Frequency Tracking:** An indexed analytical timeline tracking frequency distribution across distinct calendar days, optimized to isolate specific geopolitical and macroeconomic narrative breakthroughs.
* **Machine Learning Trend Forecasting:** Integrates a Scikit-Learn Linear Regression pipeline that fits historical headline counts to extrapolate a structured 30-day trajectory forecast, complete with optimized data-label spacing for clean visualization.
* **Markovian Predictive Language Engine:** A statistical language forecasting module using token adjacency frequencies to dynamically calculate and chart next-word probability weights based on selection criteria.

## Tech Stack and Dependencies
This system is optimized for execution speed, low operational memory footprint, and quick rendering times.

* **Core Framework:** `streamlit==1.32.0`
* **Machine Learning and Analysis:** `scikit-learn==1.4.0`, `pandas==2.2.0`, `numpy==1.26.0`
* **Data Ingestion and Engineering:** `requests==2.31.0`
* **Natural Language Processing:** `nltk==3.8.1`, `wordcloud==1.9.3`
* **Data Visualization Matrix:** `matplotlib==3.8.2`, `seaborn==0.13.2`

## How to Access and Run the Application

### 1. Secure an Official NYT API Key
To execute the processing pipeline, a connection credential key is required:
1. Navigate to the official developer platform: https://developer.nytimes.com/
2. Create a free developer account profile.
3. Register an application to activate and copy your custom Archive API key identifier.

### 2. Launching the App Interface
Input your API key directly within the app's encrypted sidebar parameter inputs or configure it via Streamlit Secrets. This credential is handled securely within runtime memory and is never written to disk or logged externally.

## Production Scaling and Enterprise Architecture
To showcase how this architecture scales to an enterprise production environment, the design is ready for the following structural enhancements:

### Automated Cloud Ingestion Pipeline
Instead of querying the NYT API directly from the client frontend on every reload, the data pipeline should be isolated using a scheduled event-driven cloud function triggered daily by an automated orchestrator like Apache Airflow.

### Database Warehouse Layer
To optimize API usage limits and enable deep historical analytics across multiple years, fetched JSON payloads can be transformed and written to an enterprise analytical warehouse like Snowflake or PostgreSQL, utilizing dbt (Data Build Tool) for scheduled batch transformation layers.

### Advanced Modeling Scale
The native predictive engines can be serialized and cached externally using Redis to handle rapid client requests, or scaled out into containerized microservices managed via Docker and FastAPI endpoints.
