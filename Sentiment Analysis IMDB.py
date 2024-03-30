#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from ttkthemes import ThemedStyle

# Loading the dataset
data = pd.read_csv("IMDBDataset.csv")

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['class'], test_size=0.2, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Training the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

def analyze_review():
    review = user_review.get()  # Get the review from the text box

    # Vectorize the review using the same vectorizer
    review_vectorized = vectorizer.transform([review])

    # Making prediction on the review
    prediction = classifier.predict(review_vectorized)

    # Create and style custom pop-up window
    popup = tk.Toplevel(root)
    popup.title("Review Analysis")
    popup.geometry("300x100")
    popup.configure(bg="#FFD700")  # Custom background color

    # Display the prediction result
    label = ttk.Label(popup, text=f"The review is {prediction[0]}.", font=("Helvetica", 12))
    label.pack(pady=10)

def analyze_data():
    # Making predictions on the test data
    predictions = classifier.predict(X_test_vectorized)

    # Calculating accuracy, precision, and recall
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label='positive')
    recall = recall_score(y_test, predictions, pos_label='positive')

    # Preparing data for tabulate
    data = [["Accuracy", accuracy], ["Precision", precision], ["Recall", recall]]

    # Create and style custom message box
    messagebox.showinfo("Analysis Results", tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))

# Creating the main window
root = tk.Tk()
root.title("IMDB Sentiment Analysis")
root.geometry("400x200")  # Set initial window size

# Use ThemedStyle to apply a custom theme
style = ThemedStyle(root)
style.set_theme("radiance")

# Title label
title_label = ttk.Label(root, text="IMDB Sentiment Analysis", font=("Helvetica", 20))
title_label.pack(pady=10)

# Text box for entering a review
user_review_label = ttk.Label(root, text="Enter your review:")
user_review_label.pack()
user_review = ttk.Entry(root, width=50)
user_review.pack()

# Button to analyze the user review
analyze_review_button = ttk.Button(root, text="Analyze Review", command=analyze_review)
analyze_review_button.pack(pady=5)

# Button to analyze the IMDb dataset
analyze_data_button = ttk.Button(root, text="Analyze IMDb Dataset", command=analyze_data)
analyze_data_button.pack(pady=5)

# Run the application
root.mainloop()

