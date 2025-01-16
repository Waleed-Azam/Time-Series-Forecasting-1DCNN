# Acoustic Doppler Velocity Log (ADVL) Time Series Forecasting

This repository contains Python implementations for analyzing and forecasting acoustic Doppler velocity log (ADVL) data. The data comprises signals from 4 beams, used for guidance and path-following, with the additional challenge of detecting fish interference. The presence of fish is indicated by deviations in peak values, which typically range between 3.5–4 meters in normal conditions.

## Features

- **LSTM Model:** Leverages Long Short-Term Memory networks for sequential modeling and forecasting.
- **1D CNN Model:** Utilizes Convolutional Neural Networks for extracting spatial patterns in time series data.
- **Mini-Batch Training:** Supports efficient training on large datasets.
- **Forecasting:** Predicts future velocity log values with a focus on identifying fish interference.

---

## Installation

Clone the repository and install the required dependencies.

### Clone the Repository

```bash
git clone https://github.com/<your-username>/advl-forecasting.git
cd advl-forecasting


Install Dependencies
Ensure you have Python 3.7+ installed. Install the required libraries using pip.

bash
Copy
Edit
pip install -r requirements.txt
Requirements
The requirements.txt file includes:

tensorflow (Deep learning framework)
numpy (Numerical computations)
pandas (Data manipulation)
matplotlib (Data visualization)
scikit-learn (Data preprocessing and metrics)
Data Preparation
Ensure your dataset is structured as follows:

Input: Time series data from 4 beams with normalized values.
Output: A target variable representing the peak distance or deviation (e.g., to detect fish interference).
Example Data Structure
Time	Beam1	Beam2	Beam3	Beam4	Target
t1	3.9	3.8	4.0	3.7	1
t2	3.5	3.6	3.9	3.8	0
Usage
Running the Models
Preprocess Data: Ensure your data is normalized and split into training, validation, and test sets.

Train Models: Run the script to train both LSTM and CNN models.

bash
Copy
Edit
python advl_forecast.py
View Forecasts: The script generates plots comparing actual vs. predicted values, highlighting deviations caused by fish interference.
Configurable Parameters
Modify the following parameters in the script for custom setups:

BATCH_SIZE: Mini-batch size for training
EPOCHS: Number of epochs
SEQ_LEN: Sequence length for time series inputs
LEARNING_RATE: Learning rate for optimizers
Output
Plots: Visualizations comparing actual and predicted time series values.
Metrics: Evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).
External Resources
Here are some useful resources related to this project:

TensorFlow Documentation: Learn more about LSTM and CNN implementations.
ADCP Overview: Basics of Acoustic Doppler Current Profilers.
Deep Learning for Time Series Forecasting: Related open-source repository.
Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

License
This project is licensed under the MIT License.

Contact
For questions or suggestions, please feel free to reach out:

Author: Your Name
Email: waleed.ntnu@gmail.com
vbnet
Copy
Edit

Let me know if you'd like me to refine or expand on any section!
