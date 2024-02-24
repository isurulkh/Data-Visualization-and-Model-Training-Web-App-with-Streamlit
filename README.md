## Data Visualizer & Model Trainer Web App

This Streamlit app empowers you to analyze and model your data directly within a web browser.

**Key Features**

* **Data Upload:** Easily import your data in CSV format.
* **Data Preview:** Get a quick glimpse of the first few rows of your dataset.
* **Data Preprocessing:** 
    * Automatic conversion of date columns into usable features.
    * One-hot encoding of categorical variables.
* **Data Analysis:**
    * Descriptive statistics.
    * Missing value overview.
    * Correlation heatmap.
* **Interactive Data Visualizations:**
    * Line plots
    * Bar plots
    * Scatter plots
    * Histograms
    * Interactive plots using Plotly
    * Box plots
    * Pair plots with Seaborn 
* **Regression Model Building and Comparison:**
    * Linear Regression
    * Ridge Regression
    * Lasso Regression
    * Random Forest 
    * Automated model selection based on Mean Squared Error (MSE)

**Getting Started**

1. **Prerequisites:**
   * Python 3.x (https://www.python.org/downloads/)
   * pip (https://pip.pypa.io/en/stable/installation/)
   * Streamlit, pandas, NumPy, matplotlib, seaborn, Plotly, scikit-learn (these will be installed in step 3)

2. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/[project-name].git

3.**Create a virtual environment:**

  ```bash
  python -m venv venv
  
  ```
4.**Create a virtual environment:**

  * **Windows**
    ```bash
    .\venv\Scripts\activate
    ```
  
 *  **Mac**
    ```bash
    source venv/bin/activate
    ```

5.**Install dependencies:**

  ```bash
  pip install -r requirements.txt
  
  ```
#### Streamlit Application

* Run the Streamlit App: In a new terminal window or tab, navigate to your project directory, activate the virtual environment, and execute the following command:
 ```bash
streamlit run app.py
 ```


