# Amazon Product Quality Prediction Tool
### Collaborators: Saksham A., Alex J., Gokul S.

## Project Overview 
Can one predict that quality of unseen amazon products given access to a breadth of information about known amazon products? In particular, can we determine whether a product is great (> 4.5 star rating) or not great (<= 4.5 start rating). This was the question that inspired the creation of our tool, which predicts the quality of amazon products based on their reviews,  sales rank, and price. 
## Main Features
### Sentiment Classifier 
Our tool relies heavily on data about the *sentiment* contained in reviews and review summaries, as such we needed to create a sentiment classifier. This classifier has been trained using the reviews provided in the training data. 

### Binary Classifier
Our binary classifier utilizes review sentiment, review summary sentiment, sales rank, and price to predict the product quality. We used several different classifiers:  logistic regression, KNN, standard NN, SVM, and decision trees to name a few. 
## How to Use Tool
1. Clone the repo locally and simply execute the driver.py.
    * make sure that all of the dependencies and import statements are correct on your local system


*This project was created as part of Dartmouth's Machine Learning course under the instruction of V.S. Subrahmanian.*  
