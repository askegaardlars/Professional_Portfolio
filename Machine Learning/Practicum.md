# Practicum2021

## Logistic Regression
Below are the specifications for replicating a successful run of Logistic_Regression.py
### Dependency information
* System: MacOS, Windows 10, Windows 11, and Ubuntu 18.04
* Python version: 3.6 or above
* Packages version
  * pandas: 1.1.5
  * numpy: 1.19.5
  * scikit-learn: 0.24.2
### Functions usage

* get_five()
  * a function that retrieves the 5 highest-confidence device problem predictions in rank order for each instance
  * Contains 2 arguments:
    * clf: the classifier used for the test. In this case, it will be logistic regression using newton-cg as solver
    * X_test: contains 5% split of the training data to evaluate model precision during our testing phase
    * Return Values: Top 5 predicted label
* scoring()
  * Compares model device problem predictions to the actual device problem for each reported instance, and returns metrics that describe the performance of the model on the input data set
  * Contains 4 arguments:
    * clf: the classifier used for the test. In this case, it will be logistic regression using newton-cg as solver
    * X_test: contains an ‘Event Text’ list of 5% of the training data to evaluate model precision during testing phase
    * DP_test: contains a ‘Device Problem’ list of 5% of the training data whose ‘Device Problem’ observation indices correspond with the ‘Event Text’ observation indices of X_test
    * DP_pred: contains a list of predicted device problems from the OneVsRestClassifier for the ‘Event Texts’ of X_test in corresponding order
    * Return Values:
    * top5: holds a list of the top 5 predicted device problems in rank order for the current instance in the dataset.
    * scoring_result: holds a list of the proportion of correctly predicted ‘Device Problems’ to total number of ‘Device Problems’ for each reported instance
    * correct: holds a string list of the correct ‘Device Problems’ that appeared in the list of top 5 predicted labels for each reported instance
    * missing: holds a string list of the correct ‘Device Problems’ that were missing from the list of top 5 predicted labels for each reported instance
    * Actual: holds a string list of all the correct ‘Device Problems’ for each reported instance of the test set
* iter_test(path, n, clf)
  * Use for training and testing logistic regression model using a combined training and testing data set
  * Take 3 arguments
    * path: the absolute path of the data set. Example: r'C:\example\data.csv'
    * n: number of time we will fit the model and test it
    * clf: the classifier using for the test. In this case, it will be logistic regression using newton-cg as solver
  * Output
     * top5: the set of top5 labels that were predicted by the model. Usage: top5[index] will hold the set of top 5 predicted labels for all instances at test number (index + 1)
     * scoring_result: the list of matching percentage between actual and top5.  Usage: scoring_result[index] will hold the set of mathching percentage for all testing instances at test number (index + 1)
     * correct: the list of corrected predicted labels coming out of the top5.  Usage: correct[index] will hold the set of correctly predicted labels for all testing instances at test number (index + 1)
     * missing: the list of missing label inside top5. Usage: missing[index] will hold the set of missing labels in top5 for all testing instances at test number (index + 1)
     * actual: the set of labels that was retrieved from the testing data set. Usage: actual[index] will hold the set of label for all testing instances at test number (index + 1)
* scoring_dff_test()
  * Same usage as scoring but istead of only one path, the first argument will be the path to training data set and the second argument will be the path to the testing data set
* iter_test_dff_test()
  * Same usage as iter_test but istead of only one path, the first argument will be the path to training data set and the second argument will be the path to the testing data set

## Random Forest
Below are the specifications for replicating a successful run of Logistic_Regression.py
### Dependency information
* System: MacOS, Windows 10, Windows 11, and Ubuntu 18.04
* Python version: 3.6 or above
* Packages version
  * pandas: 1.1.5
  * numpy: 1.19.5
  * scikit-learn: 0.24.2
### Functions usage

* get_five()
  * a function that retrieves the 5 highest-confidence device problem predictions in rank order for each instance
  * Contains 2 arguments:
    * clf: the classifier used for the test. In this case, it will be logistic regression using newton-cg as solver
    * X_test: contains 5% split of the training data to evaluate model precision during our testing phase
    * Return Values: Top 5 predicted label
* scoring()
  * Compares model device problem predictions to the actual device problem for each reported instance, and returns metrics that describe the performance of the model on the input data set
  * Contains 4 arguments:
    * clf: the classifier used for the test. In this case, it will be logistic regression using newton-cg as solver
    * X_test: contains an ‘Event Text’ list of 5% of the training data to evaluate model precision during testing phase
    * DP_test: contains a ‘Device Problem’ list of 5% of the training data whose ‘Device Problem’ observation indices correspond with the ‘Event Text’ observation indices of X_test
    * DP_pred: contains a list of predicted device problems from the OneVsRestClassifier for the ‘Event Texts’ of X_test in corresponding order
    * Return Values:
    * top5: holds a list of the top 5 predicted device problems in rank order for the current instance in the dataset.
    * scoring_result: holds a list of the proportion of correctly predicted ‘Device Problems’ to total number of ‘Device Problems’ for each reported instance
    * correct: holds a string list of the correct ‘Device Problems’ that appeared in the list of top 5 predicted labels for each reported instance
    * missing: holds a string list of the correct ‘Device Problems’ that were missing from the list of top 5 predicted labels for each reported instance
    * Actual: holds a string list of all the correct ‘Device Problems’ for each reported instance of the test set
* iter_test(path, n, clf)
  * Use for training and testing logistic regression model using a combined training and testing data set
  * Take 3 arguments
    * path: the absolute path of the data set. Example: r'C:\example\data.csv'
    * n: number of time we will fit the model and test it
    * clf: the classifier using for the test. In this case, it will be logistic regression using newton-cg as solver
  * Output
     * top5: the set of top5 labels that were predicted by the model. Usage: top5[index] will hold the set of top 5 predicted labels for all instances at test number (index + 1)
     * scoring_result: the list of matching percentage between actual and top5.  Usage: scoring_result[index] will hold the set of mathching percentage for all testing instances at test number (index + 1)
     * correct: the list of corrected predicted labels coming out of the top5.  Usage: correct[index] will hold the set of correctly predicted labels for all testing instances at test number (index + 1)
     * missing: the list of missing label inside top5. Usage: missing[index] will hold the set of missing labels in top5 for all testing instances at test number (index + 1)
     * actual: the set of labels that was retrieved from the testing data set. Usage: actual[index] will hold the set of label for all testing instances at test number (index + 1)
* scoring_dff_test()
  * Same usage as scoring but istead of only one path, the first argument will be the path to training data set and the second argument will be the path to the testing data set
* iter_test_dff_test()
  * Same usage as iter_test but istead of only one path, the first argument will be the path to training data set and the second argument will be the path to the testing data set

## BERT
There are two different BERTSingleDataSet.py and BERTSeperateDataSet.py. The only difference is that one have seperate training and testing data set and the other one is using one dataset and split it into two data set.
### Dependency information
* System: Ubuntu 18.04
* GPU: GTX 980
* Python: 3.6
* Package version
  * pandas: 1.1.5
  * numpy: 1.19.5
  * scikit-learn: 0.24.2
  * pytorch-lightning: 1.5.8
  * torch: 1.10.1
  * transformer: 4.15.0
* Pre-trained model: bert-base-case
* Object:
  * Dataset: tokenize and encode the text data numerically in a structured format required for BERT
    * Sample Usage: Train_dataset = Dataset(train_df, tokenizer, max_token_len=MAX_TOKEN_COUNT) where train_df is the data frame training data
  * DataModule: set up the dataloader using pytorch-lightning
    * Sample Usage: data_module = DataModule(X_train, DP_train, X_val, DP_val, X_test, DP_test ,tokenizer,batch_size=BATCH_SIZE,max_token_len=MAX_TOKEN_COUNT) where X is for event text and DP stands for device problems
  * Tagger: additional configuration on the core BERT model
    * Sample Usage: trained_model = Tagger.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,n_classes=len(DP_test[0])). This will load the model from the checkpoint of the best version of the model in the training loop
 * Output:
   * Two data frame:
     * Actual: all the labels in the testing set for each instances
     * Top5: all the top 5 labels for each instances that were predicted by the model
## Model and result analysis
### Dependency information
* System: Windows 10
* R-studio: 4.0.2
### Data cleaning
* lemmatizing.Rmd
  * Lemmatizes the original data
  * ~/Medtronic/Files for Medtronic/Data Cleaning
* annex_data_cleaning.Rmd
  * Reformats the MAUDE annex data to list each level 3 code with its parent level 2 and 1 coes
### Modeling
We build our models in Python and run them in RStudio.
* logistic_regression.Rmd
* random Forest.Rmd
* Assessment
  * model_analysis_log.Rmd
    * Generates recall statistics for the logistic regression model
  * model_analysis_rf.Rmd
    * Generates recall statistics for the random forest model
  * model_analysis_bert.Rmd
    * Generates recall statistics for the bert model
    * Note that BERT is not ran in R, unlike the other models
### Visualization
  * graph_making.Rmd
    * Uses results from model_analysis_log.Rmd,model_analysis_rf.Rmd, and model_analysis_bert.Rmd to creates graphs found in presentation
