# MLProject template based on Python language.
This template will serve as the basis for the development of any executable experiment in the **Rocket framework**.

*Rocket*, from its MLProject development IDE, allows you to run the MlFlow project easily. In this case we will execute the following example: [MlFlow Project about Wine-Quality](https://github.com/mlflow/mlflow/tree/master/examples/r_wine). This example demonstrates how to read data, log parameters, metrics, and models from Python in our *Rocket* framework.

## Running this Example

First, you need to download the wine dataset to upload it as part of the asset catalog:

```bash
$ sudo wget https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv
```

Next, once the csv has been loaded into HDFS, include it in the catalog by running the following statement for example. Don't forget to read the headers, infer the schema and set your own path.

```sql
CREATE TABLE default.wine_quality USING csv OPTIONS(path 'hdfs://s000001-hdfs-example.s000001/datasets/wine-quality.csv', header "true", inferSchema "true")
```

Now, from your MLProject IDE, you need to access the Data tab and include the file as input dataset, setting the features and target. After then you just need to run the experiment.
All the experiments run the command `mlflow run` passing the parameters indicated in the MLProject file. In this case:

```
parameters:
      training_data: string
      evaluation_data: string
      alpha: {type: float, default: 0.5}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {training_data} {evaluation_data} {alpha} {l1_ratio}"
```

These parameters will be read as arguments in the file *train.py* as follows:

```python
train = pd.read_csv(sys.argv[1], sep=',')
test = pd.read_csv(sys.argv[2], sep=',')

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = float(sys.argv[3]) if len(sys.argv) > 2 else 0.5
l1_ratio = float(sys.argv[4]) if len(sys.argv) > 3 else 0.5
```

At that time, you can already run the model and track the associated metrics and parameters, including the model itself:

```python
with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
```

This experiment will be run in a virtual environment that could be defined through the property `conda.env` in the MLProject file. Otherwise, it will be run with the installed default virtual environment.

*MLProject file:*
```
conda_env: conda.yaml
```

*conda.yaml:*
```
name: rocket-ml-flow

channels:
  - conda-forge
  - nodefaults

dependencies:
  - python=3.9.7
  - pip=21.2.4
  - pip:
      - mlflow==1.26.1
      - scikit-learn==1.0.2
      - numpy==1.22.3
      - pandas==1.4.2
```

The conda.yaml file includes all the mandatory dependencies to run your mlflow model correctly.

## Logging artifacts

Stratio Rocket provides the ability to store logged artifacts together with the MlProject execution and with the exported MlModel.
By default, all the artifacts created during the execution of the MlProject will be stored with the model.
If you want to store with the MlModel, only a subset of the artifacts created during execution of the MlProject, Stratio Rocket provides an optional section in the `MLproject` file where you can declare which artifacts will be exported with the model.
This new section is defined at root level in the `MLproject` file:

```
rocket:
  post_execution:
    model_artifacts:
      - <artifact file to store with the MlModel>
      - <artifacts directory to store with the MlModel>
```

In this section, you need to list all the artifact files or directories that will be logged with the exported MlModel.
You can log with the MlModel files and entire directories that are present in the MlFlow project directory or that are created during the training process.
If an artifact does not exist, it will simply be skipped.

More information about MLFlow projects, please check [MlFlow Documentation](https://www.mlflow.org/docs/latest/index.html). 

## Disclaimers

For an MLProject to be run properly, the following must be met:
* Optionally, a file named *.rocket.conf* can exist in the root directory of the project. If it exists, this file must contain the property `mlproyect.path` that will indicate the base path of the MLProject, otherwise the *MLProject* file must be located in the root directory.
* MLProject must be self-contained, that is, the *MLProject* file will define the root path of all its dependent files (model, virtual environment, etc).
* If an MLProject file has an environment descriptor defined through the `conda.env` property, this file must exist and have a proper YAML format.

