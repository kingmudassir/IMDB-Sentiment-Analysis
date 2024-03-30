# IMDB-Sentiment-Analysis

## Language:
- Python

## IDE Used:
• The IDE I have used is Jupyter Notebook

## Brief Explanation:
- The dataset is loaded, split into training and testing sets, and then vectorized using
CountVectorizer. A Multinomial Naive Bayes classifier is trained on the training data.
Two functions are defined for analyzing user-entered reviews and the dataset itself,
which display results using message boxes. Finally, tkinter widgets like labels, entry
boxes, and buttons are created to interact with the user, and the application runs in the
main loop. It also utilizes the tkinter library to create a GUI application for sentiment
analysis of movie reviews from the IMDB dataset.

## Working of Code:
- It reads the IMDB dataset from a CSV file into a pandas DataFrame named data.
- The dataset is split into training and testing sets using train_test_split function from sklearn.
- The text data in both training and testing sets is vectorized using CountVectorizer to convert
text into numerical features.
- A Naive Bayes classifier is initialized and trained on the vectorized training data.
-- analyze_review(): This function is called when the user wants to analyze a review
entered into the GUI. It vectorizes the input review, makes a prediction using the
trained classifier, and displays the prediction using a message box.
-- analyze_data(): This function is called when the user wants to analyze the entire
IMDB dataset. It makes predictions on the testing data, calculates accuracy,
precision, and recall scores, formats the results, and displays them in a message
box.
- It creates the main window of the GUI application using tkinter.
- The mainloop() function is called to run the GUI application, which waits for user
interactions.
