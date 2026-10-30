# Spam Email Classifier

A simple Machine Learning web application that classifies text messages as **Spam** or **Not Spam**. The project uses Natural Language Processing (NLP) techniques to convert text into numerical features and a Machine Learning classification model to predict whether a given message is spam.

The trained model is integrated with a Flask web application, allowing users to enter a message and receive an instant spam classification.

## Features

* Classifies messages as **Spam** or **Not Spam**
* Simple and user-friendly web interface
* Text feature extraction using TF-IDF
* Machine Learning classification using Multinomial Naive Bayes
* Flask-based backend
* Pre-trained model saved using Pickle
* Real-time predictions through the web interface

## Technologies Used

* Python
* Flask
* Pandas
* Scikit-learn
* HTML
* CSS
* Machine Learning
* Natural Language Processing (NLP)

## Machine Learning Workflow

The project follows the following ML pipeline:

1. Load and preprocess the spam message dataset.
2. Convert the message labels into numerical values:

   * Ham / Not Spam → `0`
   * Spam → `1`
3. Split the dataset into training and testing sets.
4. Convert text messages into numerical features using **TF-IDF Vectorization**.
5. Train a **Multinomial Naive Bayes** classifier.
6. Evaluate the trained model on the test dataset.
7. Save the trained model and TF-IDF vectorizer using Pickle.
8. Load the trained model in the Flask application.
9. Accept user input and classify new messages as Spam or Not Spam.

## Project Structure

```text
Spam-Email-Classifier/
│
├── dataset/
│   └── spam.csv
│
├── model/
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The project uses the **SMS Spam Collection Dataset**, which contains messages labeled as either spam or ham (legitimate messages).

Dataset source: UCI Machine Learning Repository — SMS Spam Collection.

The dataset contains two main columns:

* `v1` — Message label (`ham` or `spam`)
* `v2` — Message text

Note: Although this project is presented as a Spam Email Classifier, the training dataset primarily contains SMS messages. The same text classification concepts can be extended to email spam datasets.

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd Spam-Email-Classifier
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script:

```bash
python train_model.py
```

This will train the Multinomial Naive Bayes model and generate:

```text
model/spam_model.pkl
model/vectorizer.pkl
```

## Run the Application

Start the Flask development server:

```bash
python app.py
```

Open the local URL displayed in the terminal in your web browser.

Enter a message in the text area and click **Classify Email** to get the prediction.

## Example
## Screenshots

### Home Page

![Home Page](screenshots/home.png)

### Spam Prediction

![Spam Prediction](screenshots/spam-result.png)

### Not Spam Prediction

![Not Spam Prediction](screenshots/not-spam-result.png)

### Input

```text
Congratulations! You have won a $1,000 cash prize. Click here now to claim your reward!
```

### Output

```text
Spam Detected
```

### Input

```text
Hi, are we still meeting for lunch tomorrow at 1 PM?
```

### Output

```text
Not Spam
```

## Future Improvements

* Compare multiple Machine Learning algorithms
* Display prediction confidence
* Add advanced text preprocessing
* Train the model using a larger email-specific spam dataset
* Add model evaluation visualizations such as a confusion matrix
* Deploy the application for public access

## Author

**Mohit Patil**


## License

This project is intended for educational and learning purposes.
