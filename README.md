# Grant-Recipient-Classifier
This classifier is my implementation of a model that categorizes applicants for tuition grants at Alusive Africa as either qualified or not as per the provided application data.

## Introduction and Motivation

Machine Learning (ML) has revolutionized various industries by providing data-driven insights and automating decision-making processes. In the context of tuition support grants, ML can be employed to assess applications and categorize applicants. Additionally, it can be used to determining appropriate amounts of financial aid to award. This approach leverages existing technologies such as:

1. **Data Collection and Preprocessing**: Collecting comprehensive data from student applications, including financial background, academic performance, and extracurricular involvement.
2. **Predictive Modeling**: Utilizing algorithms like regression analysis, decision trees, or neural networks to predict the grant amount based on historical data.
3. **Natural Language Processing (NLP)**: Analyzing essay components or personal statements in applications to gauge the need and eligibility for grants.
4. **Clustering and Classification**: Grouping students with similar financial and academic profiles to ensure fair and equitable distribution of funds.

The integration of ML into the grant evaluation process at our organization, Alusive Africa, has profound significance with social impact:

1. **Fairness and Objectivity**: ML models can mitigate human biases in decision-making, ensuring a more equitable distribution of grants based on data-driven insights.
2. **Efficiency and Scalability**: Automating the evaluation process reduces administrative burden, allowing the organization to process more applications in less time. This scalability can support a larger number of students.
3. **Transparency**: Data-driven decision-making promotes transparency, as the criteria for grant allocation can be clearly defined and communicated to applicants.
4. **Personalized Support**: By analyzing individual needs and circumstances, ML can tailor grant amounts to provide optimal support for each student, addressing their unique financial challenges.

## Mission Statement

As a numbers enthusiast, my mission is to empower decision making using the goldmine of data. Being the C.E.O of Alusive Africa, I am committed to leveraging the power of data-driven decision making in the application review process. Classifying our large pool of applicants as considerable or not using an ML model would complement the decision-making process for the human reviewers enormously. This mission stands to enhance education, create jobs, and support good governance.

## Problem Statement

At Alusive Africa, reviewing round one grant applications turned out to be a labor-intensive process pretty fast. This situation can be time-consuming and error-prone. However, with the integration of ML technology, the review process stands to be simplified and effectively performed.

According to our round one grant report, many fee-paying students face significant financial barriers that hinder their ability to focus well in higher education. Despite having the potential and motivation, these students struggle to afford tuition and other associated costs, leading to increased delayed graduation time and unmet academic potential.

### Problem

1. **Equity and Access**: The inability to afford education perpetuates social and economic inequalities. Students from low-income backgrounds often have limited opportunities to improve their socioeconomic status through education.
2. **Dropout Rates**: Financial stress is a leading cause of student dropout, resulting in wasted potential and investment both for the students and educational institutions.
3. **Mental Health**: Financial strain can lead to significant stress and anxiety among students, affecting their academic performance and overall well-being.
4. **Future Opportunities**: Students who cannot complete their education are likely to face limited career prospects, perpetuating cycles of poverty and underemployment.

## Objectives

1. **Collect and Preprocess Application Data**: Gather comprehensive data from student applications, including financial background, academic performance, and extracurricular involvement, and preprocess it for analysis.
2. **Select Appropriate ML Algorithms**: Identify and select suitable machine learning algorithms for the task at hand.
3. **Train the Model**: Utilize the collected data, split it into training and testing sets, and iterate on model development to improve performance.
4. **Classify Applicants**: Develop a machine learning model to classify each applicant based on their application data as qualified or not for the grant.
5. **Evaluate the Model’s Performance**: Continuously assess the model’s performance using appropriate metrics to ensure it meets the desired accuracy and reliability.

# Technical Requirements

## Data Collection and Storage (Google Forms and Sheets)
**Purpose**: To gather, store, and manage application data, including financial information, academic records, and personal statements.
### Methods of Use:
- Implement secure online Google Forms for application submissions.
- Use Google Spreadsheets to store and organize collected data.
- Ensure data encryption and access controls to protect sensitive information.

## Data Preprocessing Tool (Google Sheets & Python)
**Purpose**: To clean, transform, integrate, reduce, split, and balance raw data to prepare it for analysis by the ML model.
### Methods of Use:
- Use Google Sheets for data preparation, including implementing data cleaning procedures to handle missing values, outliers, and inconsistencies.
- Use feature engineering techniques to create relevant input variables for ML models.
### Transformation:
- Included normalization and encoding categorical variables.
### Limitations:
- Data quality and completeness are critical; missing or inaccurate data can affect model performance.
- Ethical considerations in handling sensitive information are paramount.

## Machine Learning Frameworks (Tensor FLow)
**Purpose**: To build, compile, train, and evaluate a tuned predictive model for binary classification using neural networks.

# [Data Set](https://docs.google.com/spreadsheets/d/1_gkNhY8uMD0pnO9gVOyt6QHLIq7LGKgGeeRsjUEMk-c/edit?usp=sharing)

The data set used is our own, collected from the second-round grant applicants and prepared for building this model. The data set includes the following features:

- **Academic Standing**: The academic performance of the student.
- **Disciplinary Standing**: The disciplinary record of the student.
- **Financial Standing**: The financial situation of the student.
- **Fee Balance (USD)**: The remaining fee balance the student needs to pay.
- **ALU Grant Status**: The status of the student's ALU awarded grant.
- **Previous Alusive Grant Status**: The status of any previous grants received from Alusive Africa.
- **Total Monthly Income**: The total monthly income of the student's household.
- **Students in Household**: The number of students in the student's household.
- **Household Size**: The total number of members in the student's household.
- **Household Supporters**: The number of people in the household who are financially supporting the family.
- **Household Dependants**: The number of dependents in the student's household.
- **ALU Grant Amount**: The amount of grant received from ALU.
- **Grant Requested**: The amount of grant requested by the student.
- **Amount Affordable**: The amount the student can afford to pay.
- **Grant Classifier**: The classification of the grant award status based on the provided data.

# Methodology

Here are the three models implemented and compared:

1. Basic Neural Network (**Initial Model**)
2. Regularized Neural Network (**L2 Regularization + Dropout**)
3. Advanced Regularized Neural Network (**L2 + Dropout + BatchNorm + Scheduler**)

## 1. Basic Neural Network

This simple neural network had 3 layers and 64 neurons in the first, 32 in the second and 16 in the third hidden layer. A sequential model was constructed using TensorFlow and Keras

Here's an overview of the model architecture :-

```
Input Layer
    Type: Dense
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 480

First Hidden Layer
    Type: Dense
    Number of Neurons: 64
    Activation Function: ReLU
    Output Shape: (None, 64)
    Number of Parameters: 2112

Second Hidden Layer
    Type: Dense
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 2080

Third Hidden Layer
    Type: Dense
    Number of Neurons: 16
    Activation Function: ReLU
    Output Shape: (None, 16)
    Number of Parameters: 528

Output Layer
    Type: Dense
    Number of Neurons: 1
    Activation Function: Sigmoid
    Output Shape: (None, 1)
    Number of Parameters: 17
```

### Model Accuracy

**Loss:** 0.10990891605615616 **Accuracy:** 0.9318181872367859

The model's accuracy was 93.18%

## 2. Regularized Neural Network

This regularized neural network had 3 layers and 64 neurons in the first, 32 in the second and 16 in the third hidden layer. 

In order to improve the generalization and prevent overfitting, add L2 regularization to the neural network model. This regularization technique penalizes large weights in the network, reducing the complexity of the model. 

Use a regularization parameter of 0.01.Additionally, use a dropout layers coupled with some early stopping in the model training process.

Here's an overview of the model architecture :-

```
Input Layer
    Type: Dense
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 480

Dropout Layer 1
    Type: Dropout
    Dropout Rate: 0.5
    Output Shape: (None, 32)
    Number of Parameters: 0

First Hidden Layer
    Type: Dense
    Number of Neurons: 64
    Activation Function: ReLU
    Output Shape: (None, 64)
    Number of Parameters: 2112

Dropout Layer 2
    Type: Dropout
    Dropout Rate: 0.5
    Output Shape: (None, 64)
    Number of Parameters: 0

Second Hidden Layer
    Type: Dense
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 2080

Dropout Layer 3
    Type: Dropout
    Dropout Rate: 0.5
    Output Shape: (None, 32)
    Number of Parameters: 0

Third Hidden Layer
    Type: Dense
    Number of Neurons: 16
    Activation Function: ReLU
    Output Shape: (None, 16)
    Number of Parameters: 528

Output Layer
    Type: Dense
    Number of Neurons: 1
    Activation Function: Sigmoid
    Output Shape: (None, 1)
    Number of Parameters: 17
```

### Model Accuracy

**Loss:** 0.734389066696167 **Accuracy:** 0.8863636255264282

The model's accuracy was 88.63%

## 3. Advanced Regularized Neural Network

This regularized neural network had 3 layers and 64 neurons in the first, 32 in the second and 16 in the third hidden layer. 

By using the combination of L2 regularization, dropout, batch normalization, and learning rate scheduling, you can create a robust model that is less prone to overfitting and more capable of generalizing well to new data.  

These techniques ensures that the model is well-regularized and trained effectively.

Here's an overview of the model architecture :-

```
Input Layer
    Type: Dense
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 480

Batch Normalization Layer 1
    Type: BatchNormalization
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 128

Dropout Layer 1
    Type: Dropout
    Dropout Rate: 0.5
    Output Shape: (None, 32)
    Number of Parameters: 0

First Hidden Layer
    Type: Dense
    Number of Neurons: 64
    Activation Function: ReLU
    Output Shape: (None, 64)
    Number of Parameters: 2112

Batch Normalization Layer 2
    Type: BatchNormalization
    Number of Neurons: 64
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 256

Dropout Layer 2
    Type: Dropout
    Dropout Rate: 0.5
    Output Shape: (None, 64)
    Number of Parameters: 0

Second Hidden Layer
    Type: Dense
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 2080

Batch Normalization Layer 2
    Type: BatchNormalization
    Number of Neurons: 32
    Activation Function: ReLU
    Output Shape: (None, 32)
    Number of Parameters: 128

Dropout Layer 3
    Type: Dropout
    Dropout Rate: 0.5
    Output Shape: (None, 32)
    Number of Parameters: 0

Third Hidden Layer
    Type: Dense
    Number of Neurons: 16
    Activation Function: ReLU
    Output Shape: (None, 16)
    Number of Parameters: 528

Output Layer
    Type: Dense
    Number of Neurons: 1
    Activation Function: Sigmoid
    Output Shape: (None, 1)
    Number of Parameters: 17
```

### Model Accuracy

**Loss:** 1.1690747737884521 **Accuracy:** 0.9545454382896423

The model's accuracy was 95.45%


# Performance Comparison

## 1. Basic Neural Network

Trainning and validation loss:

![alt text](<Graphs/Model 1 Training and Validation Loss.png>)

Training and validation accuracy:

![alt text](<Graphs/Model 1 Training and Validation Accuracy.png>)

## 2. Regularized Neural Network

Trainning and validation loss:

![alt text](<Graphs/Model 2 Training and Validation Loss.png>)

Training and validation accuracy:

![alt text](<Graphs/Model 2 Training and Validation Accuracy.png>)

## 3. Advanced Regularized Neural Network

Trainning and validation loss:

![alt text](<Graphs/Model 3 Training and Validation Loss.png>)

Training and validation accuracy:

![alt text](<Graphs/Model 3 Training and Validation Accuracy.png>)

#### The “L2 + Dropout + BatchNorm + Scheduler” Performance

The model referred to as “L2 + Dropout + BatchNorm + Scheduler” includes several techniques to enhance performance and generalization. Here’s a brief explanation of each component and its value:

1.	*L2 Regularization (Weight Decay):*
	
    •	Value: Helps prevent overfitting by penalizing large weights. It adds a term to the loss function that is proportional to the sum of the squared weights.
	
    •	Benefit: Encourages the model to use smaller weights, which can lead to simpler models that generalize better.
2.	*Dropout:*
	
    •	Value: Prevents overfitting by randomly setting a fraction of input units to zero during training. This forces the network to learn redundant representations and reduces the reliance on any single neuron.
	
    •	Benefit: Helps in making the model more robust by preventing co-adaptation of neurons.
3.	*Batch Normalization:*
	
    •	Value: Normalizes the inputs of each layer to have zero mean and unit variance. This helps in stabilizing and accelerating training by reducing internal covariate shift.
	
    •	Benefit: Improves training speed, helps in regularization, and can allow for higher learning rates.
4.	*Learning Rate Scheduler:*
	
    •	Value: Adjusts the learning rate during training according to a predefined schedule. Common strategies include reducing the learning rate when a plateau in validation performance is detected.
	
    •	Benefit: Helps in achieving faster convergence and better final performance by adjusting the learning rate dynamically.


Overall, the "L2 + Dropout + BatchNorm + Scheduler" model comes out on top.

It edges out the "Initial Model" in all measured aspects, achieving slightly better precision, perfect recall, a higher F1 score, and a stronger ROC-AUC score.
While the "L2 Regularization + Dropout" model boasts good recall, it falls behind in other areas.

Here's why the "L2 + Dropout + BatchNorm + Scheduler" model shines:

1.  **High precision and recall:** It effectively captures all true positives while minimizing false positives.

2.  **Strong F1 score:** This indicates a good balance between precision and recall.

3.  **Superior ROC-AUC score:** This signifies its ability to differentiate between positive and negative cases.

In simpler terms, this model excels at correctly identifying both positive and negative instances, making it the most reliable choice based on the metrics as shown in the graph below.

![alt text](<graphs/Comparison of Precision, Recall, F1 Score, and ROC-AUC Score.png>)