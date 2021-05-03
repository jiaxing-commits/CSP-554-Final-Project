''' Utility functions for spark_pkg/*/mllib.py '''
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.regression import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *


def encode_data(df, categorical_cols, numeric_cols, predict_col, encode_predict_col):
    """
    Args:
        categorical_cols (list): list of collumns to be one-hot encoded (does not include predict_col for classification)
        numeric_cols (list): numeric columns
        predict_col (string): attribute to predict
        encode_predict_col (boolean): should the predict_col be encoded (classification) or not (regression)
    Returns:
        DataFrame with columns
            'label': one hot encoded label column for classification. Not included for regression
            'features': numeric and one hot encoded variables. Included for both classificaiton and regression
    """
    cols = df.columns
    stages = []
    # one hot encoding stages for categorical predictor variables
    for categoricalCol in categorical_cols:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    # possibly one hot encode the predict_col if this is a classification problem
    predict_col_tf = predict_col
    if encode_predict_col:
        predict_col_tf = 'label'
        predict_col_stringIdx = StringIndexer(inputCol = predict_col, outputCol=predict_col_tf)
        stages += [predict_col_stringIdx]
    assemblerInputs = [c + "classVec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    # pipeline stages
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    # return appropriote subset of DataFrame
    selectedCols = [predict_col_tf,'features']
    df = df.select(selectedCols)
    return df


def eval_model(f, model_name, model, test, evaluator, metric_names):
    """
    Evaluate models
    Args:
        f (File): file to write to
        model_name (str): name of this model
        model: model to evaluate
        test (DataFrame): testing dataset
        evaluator (from pyspark.ml.evaluation): handle of an evaluator
        metric_names (list): list of metric names for evaluator to evaluate
    """
    metric_vals = [None]*len(metric_names)
    predictions = model.transform(test)
    for i in range(len(metric_names)):
        metric_vals[i] = evaluator(metricName=metric_names[i]).evaluate(predictions)
        print('\t %s: %.3f'%(metric_names[i], metric_vals[i]))
    f.write(model_name+','+','.join(str(val) for val in metric_vals)+'\n')


def run_regression_models(train, test, metric_file_path):
    """
    Modeling and metrics for regression models
    Args:
        train (DataFrame): training dataset
        test (DataFrame): testing dataset
        metric_file_path (str): path to file to output metrics
    Notes:
        - Did not train IsotonicRegression is it requires wieghts column
        which does not generalize well to our vanilla/black-box testing
        - Did not use cross validation for Gradient Boosted Trees
        as these models were taking a very long time to train 
    """
    metric_names = ['r2', 'rmse', 'mae']
    f = open(metric_file_path, 'w')
    f.write('model,'+','.join(metric_names)+'\n')
    name = 'Linear Regression'
    model = LinearRegression(maxIter=10)
    param_grid = ParamGridBuilder()\
        .addGrid(model.regParam,[0,.25,.5]) \
        .addGrid(model.elasticNetParam,[0,.25,.5])\
        .build()
    model_cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = param_grid,
        evaluator = RegressionEvaluator(),
        numFolds = 3,
        seed = 60616).fit(train)
    best_model = model_cv.bestModel
    print(name)
    print('\t Best regParam (lambda): %.2f'%best_model._java_obj.getRegParam())
    print('\t Best elasticNetparam (alpha): %.2f'%best_model._java_obj.getElasticNetParam())
    eval_model(f,name,model_cv,test,RegressionEvaluator,metric_names)

    name = 'Decision Tree'
    model = DecisionTreeRegressor(seed=60616)
    param_grid = ParamGridBuilder()\
        .addGrid(model.maxDepth,[5,10,15]) \
        .addGrid(model.maxBins,[8,16,32])\
        .build()
    model_cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = param_grid,
        evaluator = RegressionEvaluator(),
        numFolds = 3,
        seed = 60616).fit(train)
    best_model = model_cv.bestModel  
    print(name)
    print('\t Best maxDepth: %d'%best_model._java_obj.getMaxDepth())
    print('\t Best maxBins: %d'%best_model._java_obj.getMaxBins())
    eval_model(f,name,model_cv,test,RegressionEvaluator,metric_names)

    name = 'Random Forest'
    model = RandomForestRegressor(seed=60616)
    param_grid = ParamGridBuilder()\
        .addGrid(model.maxDepth,[5,10,15]) \
        .addGrid(model.numTrees,[10,15,20])\
        .build()
    model_cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = param_grid,
        evaluator = RegressionEvaluator(),
        numFolds = 3,
        seed = 60616).fit(train)
    best_model = model_cv.bestModel  
    print(name)
    print('\t Best maxDepth: %d'%best_model._java_obj.getMaxDepth())
    print('\t Best maxBins: %d'%best_model._java_obj.getMaxBins())
    print('\t Best numTrees: %d'%best_model._java_obj.getNumTrees())
    eval_model(f,name,model_cv,test,RegressionEvaluator,metric_names)

    name = 'Gradient Boosted Trees'
    model = GBTRegressor(seed=60616).fit(train)
    print(name)
    eval_model(f, name, model, test, RegressionEvaluator, metric_names)
    f.close()


def run_classification_models(train, test, metric_file_path, classes):
    """
    Modeling and metrics for classification models
    Args:
        train (DataFrame): training dataset
        test (DataFrame): testing dataset
        metric_file_path (str): path to file to output metrics
        classes (int): number of unique labels
    Notes:
        - Did not train MultilayerPerceptronClassifier is it requires feature size and output
        size and therefore does not generalize well to our vanilla/black-box testing
        - Should use BinaryClassificationEvaluator if classes==2 rather than MulticlassClassiicaitonEvaluator.
        However, using the later allows for uniform metrics across all models and does *not* 
        misrepresent binary classification metrics.
        - Did not use cross validation for One Vs Rest, Gradient Boosted Trees, or Linear Support Vector Machine
        as these models were taking a very long time to train 
    """
    metric_names = ['accuracy', 'weightedRecall', 'weightedPrecision']
    f = open(metric_file_path, 'w')
    f.write('model,'+','.join(metric_names)+'\n')
    name = 'Logistic Regression'
    model = LogisticRegression()
    param_grid = ParamGridBuilder()\
        .addGrid(model.regParam,[0,.25,.5]) \
        .addGrid(model.elasticNetParam,[0,.25,.5])\
        .build()
    model_cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = param_grid,
        evaluator = MulticlassClassificationEvaluator(),
        numFolds = 3,
        seed = 60616).fit(train)
    best_model = model_cv.bestModel
    print(name)
    print('\t Best regParam (lambda): %.2f'%best_model._java_obj.getRegParam())
    print('\t Best elasticNetparam (alpha): %.2f'%best_model._java_obj.getElasticNetParam())
    eval_model(f, name, model_cv, test, MulticlassClassificationEvaluator, metric_names)

    name = 'Decision Tree'
    model = DecisionTreeClassifier(seed=60616)
    param_grid = ParamGridBuilder()\
        .addGrid(model.maxDepth,[5,10,15]) \
        .addGrid(model.maxBins,[8,16,32])\
        .build()
    model_cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = param_grid,
        evaluator = MulticlassClassificationEvaluator(),
        numFolds = 3,
        seed = 60616).fit(train)
    best_model = model_cv.bestModel  
    print(name)
    print('\t Best maxDepth: %d'%best_model._java_obj.getMaxDepth())
    print('\t Best maxBins: %d'%best_model._java_obj.getMaxBins())
    eval_model(f, name, model_cv, test, MulticlassClassificationEvaluator, metric_names)

    name = 'Random Forest'
    model = RandomForestClassifier(seed=60616)
    param_grid = ParamGridBuilder()\
        .addGrid(model.maxDepth,[5,10,15]) \
        .addGrid(model.numTrees,[10,15,20])\
        .build()
    model_cv = CrossValidator(
        estimator = model,
        estimatorParamMaps = param_grid,
        evaluator = MulticlassClassificationEvaluator(),
        numFolds = 3,
        seed = 60616).fit(train)
    best_model = model_cv.bestModel  
    print(name)
    print('\t Best maxDepth: %d'%best_model._java_obj.getMaxDepth())
    print('\t Best numTrees: %d'%best_model._java_obj.getNumTrees())
    eval_model(f, name, model_cv, test, MulticlassClassificationEvaluator, metric_names)

    name = 'Naive Bayes'
    model = NaiveBayes()
    param_grid = ParamGridBuilder() \
        .addGrid(model.smoothing, [.5, 1, 2]) \
        .build()
    model_cv = CrossValidator(
        estimator=model,
        estimatorParamMaps=param_grid,
        evaluator=MulticlassClassificationEvaluator(),
        numFolds=3,
        seed=60616).fit(train)
    best_model = model_cv.bestModel
    print(name)
    print('\t Best smoothing: %.1f' % best_model._java_obj.getSmoothing())
    eval_model(f, name, model_cv, test, MulticlassClassificationEvaluator, metric_names)

    if classes == 2:
        name = 'Gradient Boosted Trees'
        model = GBTClassifier(seed=60616).fit(train)
        print(name)
        eval_model(f, name, model_cv, test, MulticlassClassificationEvaluator, metric_names)
    f.close()