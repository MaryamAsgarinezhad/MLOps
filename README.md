# Distributed Training with Ray on Kubernetes

This project demonstrates how to set up a Ray cluster on a predefined Kubernetes environment and submit distributed training jobs using PyTorch's TorchTrainer. The training script has been adapted to support distributed execution, and the job is submitted directly to the Ray cluster via its HTTP API.

  📌 Note: The current training code is intentionally kept simple as a sample. However, the process shown here can be used to convert any PyTorch training script into a distributed trainable using Ray's TorchTrainer.


### Overview

This project focuses on:

   - Deploying a Ray Cluster on Kubernetes using KubeRay, a Kubernetes operator for Ray.

   - Running Distributed Training with PyTorch using TorchTrainer from Ray Train, allowing scalable model training across nodes.

   - Submitting Jobs to the Ray Cluster either by applying YAML job specs via Kubernetes or via Ray's HTTP job submission interface.

### Prerequisites

Make sure you have the following:

   - Access to a Kubernetes cluster (e.g., Minikube, EKS, GKE, etc.)

   - kubectl and helm installed

   - Docker installed and configured to push images to your container registry

   - Python 3.x with pip
 
   - Ray and PyTorch dependencies (as listed in trainRequirements.txt)


### Setting Up the Environment

   Install Python Requirements for Training Code:
```bash
pip install -r trainRequirements.txt
```
  Install KubeRay Operator:
```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator
```

### Deploying the Ray Cluster

Deploy a Ray cluster using the provided manifests (like ray-cluster.yaml):
```bash
kubectl apply -f KubeRayResources/ray-cluster.yaml
```

### Submitting the Training Job

Ray CLI via HTTP API
 - You can submit a job directly using Ray’s built-in CLI and REST API:

```bash
RAY_ADDRESS='http://ip:port' ray job submit --working-dir . -- python trainJob.py
```

 - RAY_ADDRESS='http://ip:port' is the URL of the Ray dashboard (Ray cluster head service)
 - You need to port-forwrd ray cluster head service or use its ClusterIP as RAY_ADDRESS.

```bash
kubectl port-forward svc/ray-head 8265:8265
```
 - Then the RAY_ADDRESS would be http://localhost:8265


### Monitoring and Managing Jobs

Access the Ray Dashboard:

```bash
kubectl port-forward svc/ray-head 8265:8265
```

  - Then go to: http://localhost:8265

Check Pod and Job Logs:

```bash
kubectl get pods
kubectl logs <job-pod-name>
```

  - Replace <job-pod-name> with the actual pod name running your training job.


### Monitoring and Metrics Collection

To gather and visualize resource usage metrics such as CPU, GPU, memory, RAM, and GPU RAM (GRAM) for each worker and head node, we use:


  - PodMonitor – to collect metrics at the pod level

  - ServiceMonitor – to collect metrics from Ray services


These resources are defined in:


   - KubeRayResources/ray-pod-monitor.yaml

   - KubeRayResources/ray-service-monitor.yaml


#### Make sure your Kubernetes cluster has Prometheus Operator deployed (e.g., via kube-prometheus-stack). Once deployed, Prometheus will automatically scrape Ray metrics exposed on /metrics.

   - These metrics can be visualized using Prometheus, Grafana, or any Prometheus-compatible dashboarding tool.


To apply the monitoring manifests:

```bash
kubectl apply -f KubeRayResources/ray-pod-monitor.yaml
kubectl apply -f KubeRayResources/ray-service-monitor.yaml
```
