# mlops_pipeline.py
import kfp
from kfp import dsl

@dsl.pipeline(
    name='Iris MLOps Pipeline',
    description='An end-to-end pipeline for the Iris dataset including ingestion, training, evaluation, and deployment.'
)
def iris_pipeline(
    data_path: str = 'data_ingestion_preprocess/data/iris_dataset.csv',   # Adjust if using cloud storage
    evaluation_threshold: float = 0.90          # Deployment threshold for model accuracy
):
    # Data Ingestion & Preprocessing
    ingest_op = dsl.ContainerOp(
        name='Data Ingestion and Preprocessing',
        image='ghorbelmahdi/data_ingest_preprocess:latest',
        arguments=[]
    )
    
    # Model Training with Hyperparameter Tuning
    train_op = dsl.ContainerOp(
        name='Model Training',
        image='ghorbelmahdi/model_training:latest',
        arguments=[]
    )
    
    # Model Evaluation
    eval_op = dsl.ContainerOp(
        name='Model Evaluation',
        image='yghorbelmahdi/model_evaluation:latest',
        arguments=[],
        file_outputs={'accuracy': 'model_evalutaion/accuracy.txt'
    
    # Conditional Deployment: deploy only if evaluation metric exceeds threshold
    with dsl.Condition(eval_op.outputs['accuracy'] > evaluation_threshold, name='DeployModel'):
        deploy_op = dsl.ContainerOp(
            name='Model Deployment',
            image='ghorbelmahdi/model_deployment:latest',
            arguments=[]
        )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(iris_pipeline, 'iris_pipeline.yaml')
