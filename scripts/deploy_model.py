#!/usr/bin/env python3
"""
Deployment script for SolarGuardAI models to Google Cloud Vertex AI.
This script handles model deployment, endpoint creation, and monitoring setup.
"""

import argparse
import os
import yaml
import time
from google.cloud import aiplatform
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vertex_ai_deployment')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy model to Vertex AI')
    parser.add_argument(
        '--project-id',
        required=True,
        help='Google Cloud Project ID'
    )
    parser.add_argument(
        '--region',
        default='us-central1',
        help='Google Cloud Region'
    )
    parser.add_argument(
        '--model-path',
        required=True,
        help='GCS path to the model artifacts'
    )
    parser.add_argument(
        '--config-file',
        default='../config/vertex_ai_deployment.yaml',
        help='Path to deployment configuration YAML'
    )
    parser.add_argument(
        '--model-name',
        default='solar_flare_predictor',
        help='Name for the model in Vertex AI Model Registry'
    )
    parser.add_argument(
        '--version-id',
        default='v1',
        help='Version ID for the model'
    )
    parser.add_argument(
        '--replace-existing',
        action='store_true',
        help='Replace existing model if it exists'
    )
    parser.add_argument(
        '--deploy-endpoint',
        action='store_true',
        help='Deploy model to an endpoint after upload'
    )
    parser.add_argument(
        '--setup-monitoring',
        action='store_true',
        help='Set up model monitoring'
    )
    return parser.parse_args()

def load_config(config_file):
    """Load deployment configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def initialize_vertex_ai(project_id, region):
    """Initialize Vertex AI SDK."""
    aiplatform.init(project=project_id, location=region)
    logger.info(f"Initialized Vertex AI for project {project_id} in {region}")
    return aiplatform

def upload_model(model_path, model_name, version_id, replace_existing, config):
    """Upload model to Vertex AI Model Registry."""
    logger.info(f"Uploading model from {model_path} to Vertex AI Model Registry")
    
    # Check if model already exists
    try:
        existing_model = aiplatform.Model.list(
            filter=f"display_name={model_name}"
        )
        if existing_model and not replace_existing:
            logger.error(f"Model {model_name} already exists. Use --replace-existing to replace it.")
            return None
        elif existing_model and replace_existing:
            logger.info(f"Replacing existing model {model_name}")
            for model in existing_model:
                model.delete()
    except Exception as e:
        logger.warning(f"Error checking for existing model: {str(e)}")
    
    # Get model registry config
    registry_config = config.get('modelRegistry', {}).get('model', {})
    
    # Upload model
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
        description=registry_config.get('description', 'Solar flare prediction model'),
        version_id=version_id,
        labels=registry_config.get('labels', {})
    )
    
    logger.info(f"Model uploaded successfully. Model ID: {model.resource_name}")
    return model

def create_endpoint(endpoint_name, project_id, region, config):
    """Create a Vertex AI endpoint for model deployment."""
    logger.info(f"Creating endpoint: {endpoint_name}")
    
    # Get endpoint config
    endpoint_config = config.get('endpoint', {})
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_name,
        project=project_id,
        location=region
    )
    
    logger.info(f"Endpoint created successfully. Endpoint ID: {endpoint.resource_name}")
    return endpoint

def deploy_model_to_endpoint(model, endpoint, config):
    """Deploy model to the specified endpoint."""
    logger.info(f"Deploying model to endpoint")
    
    # Get deployment config
    endpoint_config = config.get('endpoint', {})
    machine_type = endpoint_config.get('deploymentResourcePool', {}).get('machineType', 'n1-standard-4')
    min_replicas = endpoint_config.get('autoscaling', {}).get('minNodeCount', 1)
    max_replicas = endpoint_config.get('autoscaling', {}).get('maxNodeCount', 5)
    
    # Deploy model
    deployment = endpoint.deploy(
        model=model,
        deployed_model_display_name=model.display_name,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100
    )
    
    logger.info(f"Model deployed successfully to endpoint")
    return deployment

def setup_model_monitoring(model, config, project_id, region):
    """Set up model monitoring for drift detection."""
    logger.info(f"Setting up model monitoring")
    
    # Get monitoring config
    monitoring_config = config.get('modelRegistry', {}).get('monitoring', {})
    
    if not monitoring_config.get('enable', False):
        logger.info("Model monitoring is disabled in config. Skipping.")
        return
    
    # Get baseline dataset
    baseline_dataset = monitoring_config.get('baselineDataset', {})
    baseline_uri = baseline_dataset.get('uri', '')
    
    if not baseline_uri:
        logger.warning("No baseline dataset specified. Skipping monitoring setup.")
        return
    
    # Set up monitoring job
    monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=f"{model.display_name}_monitoring",
        project=project_id,
        location=region,
        endpoint=model.resource_name,
        logging_sampling_strategy={
            "random_sample_config": {
                "sample_rate": monitoring_config.get('sampleRate', 0.1)
            }
        },
        schedule_config={
            "monitor_interval": {
                "seconds": monitoring_config.get('monitoringInterval', 3600)
            }
        },
        alert_config={
            "email_alert_config": {
                "user_emails": [monitoring_config.get('alertConfig', {}).get('email', '')]
            }
        }
    )
    
    logger.info(f"Model monitoring set up successfully")
    return monitoring_job

def main():
    """Main function to deploy model to Vertex AI."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Initialize Vertex AI
    vertex_ai = initialize_vertex_ai(args.project_id, args.region)
    
    # Upload model
    model = upload_model(
        args.model_path,
        args.model_name,
        args.version_id,
        args.replace_existing,
        config
    )
    
    if model is None:
        logger.error("Model upload failed. Exiting.")
        return 1
    
    # Deploy to endpoint if requested
    if args.deploy_endpoint:
        endpoint_name = config.get('endpoint', {}).get('name', 'solar-flare-prediction')
        endpoint = create_endpoint(endpoint_name, args.project_id, args.region, config)
        deployment = deploy_model_to_endpoint(model, endpoint, config)
        
        # Set up monitoring if requested
        if args.setup_monitoring:
            monitoring_job = setup_model_monitoring(model, config, args.project_id, args.region)
    
    logger.info("Deployment process completed successfully")
    return 0

if __name__ == "__main__":
    exit(main())