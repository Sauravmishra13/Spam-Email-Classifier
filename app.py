from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load TF-IDF vectorizer
with open("model/vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        message = request.form.get("message")

        if message:

            # Convert text into TF-IDF features
            message_vector = vectorizer.transform([message])

            # Predict
            result = model.predict(message_vector)[0]

            if result == 1:
                prediction = "Spam"
            else:
                prediction = "Not Spam"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)