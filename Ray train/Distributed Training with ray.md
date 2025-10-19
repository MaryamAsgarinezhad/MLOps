
Ray is a unified framework for scaling AI and Python applications. Ray consists of a core distributed runtime and a set of AI libraries for simplifying ML compute:

![[Pasted image 20250817141852.png]]

Ray AI Libraries:

- [Data](https://docs.ray.io/en/latest/data/dataset.html): Scalable Datasets for ML
- [Train](https://docs.ray.io/en/latest/train/train.html): Distributed Training
- [Tune](https://docs.ray.io/en/latest/tune/index.html): Scalable Hyperparameter Tuning
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Scalable Reinforcement Learning
- [Serve](https://docs.ray.io/en/latest/serve/index.html): Scalable and Programmable Serving

Ray Core:

- [Tasks](https://docs.ray.io/en/latest/ray-core/tasks.html): Stateless functions executed in the cluster.
- [Actors](https://docs.ray.io/en/latest/ray-core/actors.html): Stateful worker processes created in the cluster.
- [Objects](https://docs.ray.io/en/latest/ray-core/objects.html): Immutable values accessible across the cluster.

Monitoring and Debugging:

- Monitor Ray apps and clusters with the [Ray Dashboard](https://docs.ray.io/en/latest/ray-core/ray-dashboard.html).
- Debug Ray apps with the [Ray Distributed Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html).

----
## Define Ray on Kubernetes:

RayCluster is a custom resource definition (CRD). **KubeRay operator** will listen to the resource events about RayCluster and create related Kubernetes resources (e.g. Pod & Service).

```shell
helm repo add kuberay https://ray-project.github.io/kuberay-helm/

# Install both CRDs and KubeRay operator v1.1.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0

# Install a RayCluster custom resource
helm install raycluster kuberay/ray-cluster --version 1.1.0

# Verify the installation
kubectl get pods
# NAME                                          READY   STATUS    RESTARTS   AGE
# kuberay-operator-6fcbb94f64-gkpc9             1/1     Running   0          89s
# raycluster-kuberay-head-qp9f4                 1/1     Running   0          66s
# raycluster-kuberay-worker-workergroup-2jckt   1/1     Running   0          66s

# Forward the port of Dashboard for monitoring
kubectl port-forward svc/raycluster-kuberay-head-svc 8265:8265
# Check 127.0.0.1:8265 for the Dashboard

# Log in to Ray head Pod and execute a job.
kubectl exec -it ${RAYCLUSTER_HEAD_POD} -- bash
python -c "import ray; ray.init(); print(ray.cluster_resources())"
#Check 127.0.0.1:8265/#/job. The status of the job should be "SUCCEEDED".
```
---
# Ray Jobs

Once we have deployed a Ray cluster (on VMs or Kubernetes, we are ready to run a Ray application.

![[Pasted image 20250817154920.png]]

There are two ways to run a Ray job on a Ray cluster:

1. (Recommended) Submit the job using the Ray Jobs API.
2. Run the driver script directly on the Ray cluster, for **interactive development**.

## Ray Jobs API

The recommended way to run a job on a Ray cluster is to use the _Ray Jobs API_, which consists of a **CLI tool**, **Python SDK**, and a **REST API**.
The Ray Jobs API allows you to submit locally developed applications to a remote Ray Cluster for execution.

After a job is submitted, it runs once to completion or failure, **regardless of the original submitter’s connectivity.** Retries or different runs with different parameters should be handled by the submitter. Jobs are bound to the lifetime of a Ray cluster, so if the cluster goes down, all running jobs on that cluster will be terminated.

## Running Jobs Interactively

If you would like to run an application _interactively_ and see the output in real time (for example, during development or debugging), you can:

- (Recommended) Run your script directly on a cluster node (e.g. after SSHing into the node using [`ray attach`](https://docs.ray.io/en/latest/cluster/cli.html#ray-attach-doc)/ exec into ray head pod), or
    
- (For Experts only) Use [Ray Client](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html#ray-client-ref) to run a script from your local machine while maintaining a connection to the cluster.

Jobs started in these ways are **not managed by the Ray Jobs API**, so the Ray Jobs API will not be able to see them or interact with them (with the exception of `ray job list` and `JobSubmissionClient.list_jobs()`).


For now, we will walk through the **CLI tools** for submitting and interacting with a Ray Job. Ray Jobs API also provides APIs for [programmatic job submission](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html#ray-job-sdk) and [job submission using REST](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/rest.html#ray-job-rest-api).

# Using the Ray Jobs CLI

Ray Jobs is available in versions 1.9+ and requires a full installation of Ray. 

```shell
pip install "ray[default]"
```

#### Submitting a job

Start with a sample script that you can run locally. The following script uses Ray APIs to submit a task and print its return value:
```python
# script.py
import ray

@ray.remote  # for non-distributed training tasks we need ray.remote annotation
def hello_world():
    return "hello world"

# Automatically connect to the running Ray cluster.
ray.init()
print(ray.get(hello_world.remote()))
```

Create an **empty** working directory with the preceding Python script inside a file named `script.py`.

| your_working_directory
| ├── script.py

Next, find the HTTP address of the Ray Cluster to which you can submit a job request. Submit jobs to the same address that the **Ray Dashboard** uses. By default, this job uses port 8265. To tell the Ray Jobs CLI how to find your Ray Cluster, pass the Ray Dashboard address. Set the `RAY_ADDRESS` environment variable:

```shell
$ export RAY_ADDRESS="http://127.0.0.1:8265"
```

We can also pass the `--address=http://127.0.0.1:8265` flag explicitly to each Ray Jobs CLI command, or prepend each command with `RAY_ADDRESS=http://127.0.0.1:8265`.

To pass headers per HTTP request to the Cluster, use the `RAY_JOB_HEADERS` environment variable. This environment variable must be in JSON form.

```shell
$ export RAY_JOB_HEADERS='{"KEY": "VALUE"}'
```

To submit the job, use **`ray job submit`**.

```shell
$ ray job submit --working-dir your_working_directory -- python script.py
```

### Interacting with Long-running Jobs

For long-running applications, you probably don’t want to require the client to wait for the job to finish. To do this, pass the `--no-wait` flag to `ray job submit`. Try this modified script that submits a task every second in an infinite loop:
```python
import ray
import time

@ray.remote
def hello_world():
    return "hello world"

ray.init()
while True:
    print(ray.get(hello_world.remote()))
    time.sleep(1)
```

Now submit the job:
```shell
$ ray job submit --no-wait --working-dir your_working_directory -- python script.py
```
### Next steps:
 
 - Query the logs of the job:
```shell
ray job logs raysubmit_tUAuCKubPAEXh6CW
```

- Query the status of the job:
```shell
ray job status raysubmit_tUAuCKubPAEXh6CW
```

- Request the job to be stopped:
```shell
ray job stop raysubmit_tUAuCKubPAEXh6CW
```

### Dependency management

To run a distributed application, ensure that all workers run in the same environment. This configuration can be challenging if multiple applications in the same Ray Cluster have different and conflicting dependencies.

To avoid dependency conflicts, Ray provides a mechanism called runtime environmentsو allowing an application to override the default environment on the Ray Cluster and run in an isolated environment, similar to virtual environments in single-node Python. **Dependencies can include both files and Python packages.**

The Ray Jobs API provides an option to specify the runtime environment when submitting a job. On the Ray Cluster, Ray installs the runtime environment across the workers and ensures that tasks in that job run in the same environment. 

```shell
ray job submit --runtime-env-json='{"pip": ["requests==2.26.0"]}' -- python script.py
```

# Python SDK Overview

## Setup

Ray Jobs is available in versions 1.9+ and requires a full installation of Ray. You can do this by running:
```shell
pip install "ray[default]"
```
## Submitting a Ray Job

The following script uses Ray APIs to submit a task and print its return value:

```python
import ray

@ray.remote
def hello_world():
    return "hello world"

ray.init()
print(ray.get(hello_world.remote()))
```
SDK calls are made via a `JobSubmissionClient` object. To initialize the client, provide the Ray cluster head node address and the port used by the Ray Dashboard (`8265` by default).
```python
from ray.job_submission import JobSubmissionClient

client = JobSubmissionClient("http://127.0.0.1:8265")
job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint="python script.py",
    # Path to the local directory that contains the script.py file
    runtime_env={"working_dir": "./"}
)
print(job_id)
```
Because job submission is asynchronous, the above call will return immediately with output like the following:

```
raysubmit_g8tDzJ6GqrCy7pd6
```

Now we can write a simple polling loop that checks the job status until it reaches a terminal state (namely, `JobStatus.SUCCEEDED`, `JobStatus.STOPPED`, or `JobStatus.FAILED`). We can also get the output of the job by calling `client.get_job_logs`.

```python
from ray.job_submission import JobSubmissionClient, JobStatus
import time

client = JobSubmissionClient("http://127.0.0.1:8265")
job_id = client.submit_job(
    # Entrypoint shell command to execute
    entrypoint="python script.py",
    # Path to the local directory that contains the script.py file
    runtime_env={"working_dir": "./"}
    # you can also specify remote URIs for your job’s working directory, such as S3 buckets or Git repositories.
)
print(job_id)

def wait_until_status(job_id, status_to_wait_for, timeout_seconds=5):
    start = time.time()
    while time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"status: {status}")
        if status in status_to_wait_for:
            break
        time.sleep(1)


wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
logs = client.get_job_logs(job_id)
print(logs)
```
In addition to getting the current status and output of a job, a submitted job can also be stopped by the user before it finishes executing.

```python
wait_until_status(job_id, {JobStatus.RUNNING})
print(f'Stopping job {job_id}')
client.stop_job(job_id)
wait_until_status(job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED})
logs = client.get_job_logs(job_id)
print(logs)
```

To get information about all jobs, call `client.list_jobs()`. This returns a `Dict[str, JobInfo]` object mapping Job IDs to their information.

Job information (status and associated metadata) is stored on the cluster indefinitely. To delete this information, you may call `client.delete_job(job_id)` for any job that is already in a terminal state.

## Specifying CPU and GPU resources

By default, the job entrypoint script always runs on the head node. We recommend doing heavy computation within Ray tasks, actors, or Ray libraries, not directly in the top level of your entrypoint script. No extra configuration is needed to do this.

However, if you need to do computation directly in the entrypoint script and would like to reserve CPU and GPU resources for the entrypoint script, you may specify the `entrypoint_num_cpus`, `entrypoint_num_gpus`, `entrypoint_memory` and `entrypoint_resources` arguments to `submit_job` (**also available  to `ray job submit` in the Jobs [CLI**](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html#ray-job-submission-cli-ref).) . These arguments function identically to the `num_cpus`, `num_gpus`, `resources`, and `_memory` arguments to ==`@ray.remote()`== decorator for tasks and actors as described in [Specifying Task or Actor Resource Requirements](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#resource-requirements).

If any of these arguments are specified, the entrypoint script will be scheduled on a node with at least the specified resources, **instead of the head node, which is the default**. 
```python
job_id = client.submit_job(
    entrypoint="python script.py",
    runtime_env={
        "working_dir": "./",
    }
    # Reserve 1 GPU for the entrypoint script
    entrypoint_num_gpus=1
)
```

If `num_gpus` is not specified, GPUs will still be available to the entrypoint script, but Ray will not provide isolation in terms of visible devices. To be precise, the environment variable **`CUDA_VISIBLE_DEVICES`** will not be set in the entrypoint script; it will only be set inside tasks and actors that have `num_gpus` specified in their `@ray.remote()` decorator.

## Client Configuration

Additional client connection options, such as custom HTTP headers and cookies, can be passed to the `JobSubmissionClient` class. A full list of options can be found in the [API Reference](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/jobs-package-ref.html#ray-job-submission-sdk-ref).

---
# Distributed Training using PyTorch

Converting an existing PyTorch script to use Ray Train.

1. Configure a model to run distributed and on the correct CPU/GPU device.
2. Configure a dataloader to shard data across the [workers](https://docs.ray.io/en/latest/train/overview.html#train-overview-worker) and place data on the correct CPU or GPU device.
3. Configure a [training function](https://docs.ray.io/en/latest/train/overview.html#train-overview-training-function) to report metrics and save checkpoints.
4. Configure [scaling](https://docs.ray.io/en/latest/train/overview.html#train-overview-scaling-config) and CPU or GPU resource requirements for a training job.
5. Launch a distributed training job with a [`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html#ray.train.torch.TorchTrainer "ray.train.torch.TorchTrainer") class.

## Set up a training function

For reference, the final code will look something like the following:
```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func():
    # Your PyTorch training code here.
    ...

scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()
```

1. `train_func` is the Python code that executes on each distributed training worker (**is the main script in your code**).
2. [`ScalingConfig`](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html#ray.train.ScalingConfig "ray.train.ScalingConfig") defines the number of distributed training workers and whether to use GPUs.
3. [`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html#ray.train.torch.TorchTrainer "ray.train.torch.TorchTrainer") launches the distributed training job.

You can also specify the input argument for `train_func` as a dictionary via the Trainer’s `train_loop_config`. For example:

```python
def train_func(config):
    lr = config["lr"]
    num_epochs = config["num_epochs"]

config = {"lr": 1e-4, "num_epochs": 10}
trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=config, ...)
```
### Warning

Avoid passing large data objects through `train_loop_config` to reduce the serialization and deserialization overhead. Instead, it’s preferred to initialize large objects (e.g. datasets, models) directly in `train_func`.
```python
 def load_dataset():
     # Return a large in-memory dataset
     ...

 def load_model():
     # Return a large in-memory model instance
     ...

-config = {"data": load_dataset(), "model": load_model()}

 def train_func(config):
-    data = config["data"]
-    model = config["model"]

+    data = load_dataset()
+    model = load_model()
     ...

 trainer = ray.train.torch.TorchTrainer(train_func, train_loop_config=config, ...)
```

### Set up a model

This section streamlines distributed training setup by automatically handling two critical tasks:

1. **Moving the model to the correct device** (e.g., GPU or CPU).
    
2. **Wrapping the model in PyTorch’s `DistributedDataParallel`(DDP)** for synchronized gradient updates across workers.

Instead of manually managing device placement and DDP wrapping in each worker:
```python
device_id = ...  # your custom logic
model = model.to(device_id or "cpu")
model = DistributedDataParallel(model, device_ids=[device_id])
```
—you can just call:
```python
model = ray.train.torch.prepare_model(model)
```

Under the hood, the function carries out the following tasks:

- **Device detection & placement**: It determines the right device for the current worker using Ray’s device utilities (like `get_device()`), and moves the model there if `move_to_device` is enabled.
    
- **Wrapping for parallelism**: If the training run involves multiple workers and the `parallel_strategy` is set (default is `"ddp"`), the model is wrapped in either `DistributedDataParallel` (DDP) or `FullyShardedDataParallel` (FSDP) when appropriate.
    
- **Uniform API across environments**: This allows you to write the same training code regardless of whether you're running on CPU, single-GPU, or multi-GPU setups. Ray abstracts away the boilerplate.

### Set up a dataset

This section simplifies the setup of your `DataLoader` for distributed training by automating two key tasks:

1. **Adding a `DistributedSampler`**: If you're using multiple training workers, Ray automatically adds a `DistributedSampler` to your `DataLoader`. This ensures that **each worker processes a unique subset of the data**, preventing overlap and improving training efficiency. If your original `DataLoader` already includes a `DistributedSampler`, Ray will respect this configuration and not add another one.
    
2. **Moving Batches to the Correct Device**: Ray can automatically move the data batches to the appropriate device (CPU or GPU) for each worker. This is particularly useful when training on multiple GPUs, as it ensures that each worker's data is placed on the correct device without manual intervention.
	
3. **Handles Epoch Setting**: If the number of workers is greater than 1, it's necessary to call `set_epoch(epoch)` on the sampler at the beginning of each epoch to ensure that each process uses a different random seed for shuffling, resulting in distinct data orderings across epochs. This variability is crucial for effective training and model performance. Ray handles this internally when you use `prepare_data_loader`

Note that this step isn’t necessary if you’re passing in Ray Data to your Trainer. See [Data Loading and Preprocessing](https://docs.ray.io/en/latest/train/user-guides/data-loading-preprocessing.html#data-ingest-torch).

```python
 from torch.utils.data import DataLoader
+import ray.train.torch

 def train_func():

     ...

     dataset = ...

     data_loader = DataLoader(dataset, batch_size=worker_batch_size, shuffle=True)
+    data_loader = ray.train.torch.prepare_data_loader(data_loader)

     for epoch in range(10):
+        if ray.train.get_context().get_world_size() > 1:
+            data_loader.sampler.set_epoch(epoch)

         for X, y in data_loader:
-            X = X.to_device(device)
-            y = y.to_device(device)

     ...
```
#### Note:

- `DistributedSampler`does not work with a `DataLoader` that wraps [`IterableDataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset "(in PyTorch v2.7)"). If you want to work with an dataset iterator, consider using [Ray Data](https://docs.ray.io/en/latest/data/data.html#data) instead of PyTorch DataLoader since it provides performant streaming data ingestion for large scale datasets.

##### When to Use Ray Train vs. Traditional PyTorch Methods

|Use Case|Recommended Approach|
|---|---|
|**Standard datasets with random access**|Continue using PyTorch's `DataLoader` with `DistributedSampler` for efficient data loading and sharding.|
|**Large-scale or streaming data**|Use Ray Data for scalable and efficient data loading and preprocessing.|
|**Custom data pipelines**|Implement custom data pipelines using Ray Data's transformation and mapping capabilities.|
|**Distributed training with Ray**|Utilize Ray Train's integration with Ray Data for seamless data handling and model training.|

In summary, for standard datasets where random access is possible, PyTorch's `DataLoader` with `DistributedSampler` is suitable. However, for large-scale or streaming data, and when working with `IterableDataset`, Ray Data provides a more efficient and scalable solution. Leveraging Ray Train's integration with Ray Data ensures a seamless and optimized workflow for distributed training.

### Report checkpoints and metrics

To monitor progress, you can report intermediate metrics and checkpoints using the [`ray.train.report()`](https://docs.ray.io/en/latest/train/api/doc/ray.train.report.html#ray.train.report "ray.train.report") utility function.
```python
+import os
+import tempfile

+import ray.train

 def train_func():

     ...

     with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        torch.save(
            model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt")
        )

+       metrics = {"loss": loss.item()}  # Training/validation metrics.

        # Build a Ray Train checkpoint from a directory
+       checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        # Ray Train will automatically save the checkpoint to persistent storage,
        # so the local `temp_checkpoint_dir` can be safely cleaned up after.
+       ray.train.report(metrics=metrics, checkpoint=checkpoint)

     ...
```
For more details, see [Monitoring and Logging Metrics](https://docs.ray.io/en/latest/train/user-guides/monitoring-logging.html#train-monitoring-and-logging) and [Saving and Loading Checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html#train-checkpointing).

## Configure persistent storage

Create a [`RunConfig`](https://docs.ray.io/en/latest/train/api/doc/ray.train.RunConfig.html#ray.train.RunConfig "ray.train.RunConfig") object to specify the path where results (including checkpoints and artifacts) will be saved.
```python
from ray.train import RunConfig

run_config = RunConfig(storage_path="/some/local/path", name="unique_run_name")
run_config = RunConfig(storage_path="s3://bucket", name="unique_run_name")
run_config = RunConfig(storage_path="/mnt/nfs", name="unique_run_name")

```
1. Local Storage: 
	**Use Case:** Ideal for single-node setups or development environments where all training processes run on the same machine.
	
	**Considerations:**
	
	- **Single-node Only:** Not suitable for multi-node clusters.
	- **Fault Tolerance:** Limited; if the node fails, data may be lost.
	- **Performance:** Depends on the local disk speed.
	
	**Note:** For multi-node clusters, using local storage on a single node (e.g., the head node) is not supported. Ray Train requires a shared storage location accessible by all nodes.

2. Shared Cloud Storage (e.g., AWS S3, Google Cloud Storage)
	**Use Case:** Recommended for distributed training across multiple nodes, especially in cloud environments.
	
	**Note:** Ensure that all nodes in the Ray cluster have access to the cloud storage to allow outputs from workers to be uploaded to the shared cloud bucket. 

3. Shared Network File System (NFS)
	**Use Case:** Suitable for on-premises setups with multiple nodes that need to access the same file system.
	
	**Benefits:**
	
	- **Centralized Storage:** All nodes can read/write to the same location.
	- **Performance:** Can offer high throughput if the NFS is well-configured.
	
	**Considerations:**
	
	- **Setup Complexity:** Requires proper NFS configuration and maintenance.
	- **Network Dependency:** Performance depends on the network speed and NFS server load.


## Launch a training job

Tying this all together, you can now launch a distributed training job with a [`TorchTrainer`](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html#ray.train.torch.TorchTrainer "ray.train.torch.TorchTrainer").

```python
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_func, scaling_config=scaling_config, run_config=run_config
)

result = trainer.fit()
```

## Access training results

After training completes, a [`Result`](https://docs.ray.io/en/latest/train/api/doc/ray.train.Result.html#ray.train.Result "ray.train.Result") object is returned which contains information about the training run, including the metrics and checkpoints reported during training.

```python
result.metrics     # The metrics reported during training.
result.checkpoint  # The latest checkpoint reported during training.
result.path        # The path where logs are stored.
result.error       # The exception that was raised, if training failed.
```

For more usage examples, see [Inspecting Training Results](https://docs.ray.io/en/latest/train/user-guides/results.html#train-inspect-results).

---
## Scale data ingest separately from training with Ray Data

Modify this example to load data with Ray Data instead of the native PyTorch DataLoader. With a few modifications, you can **scale data preprocessing and training separately**. For example, you can do the former with a pool of CPU workers and the latter with a pool of GPU workers. See [How does Ray Data compare to other solutions for offline inference?](https://docs.ray.io/en/latest/data/comparisons.html#how-does-ray-data-compare-to-other-solutions-for-ml-training-ingest) for a comparison between Ray Data and PyTorch data loading.

First, create [Ray Data Datasets](https://docs.ray.io/en/latest/data/key-concepts.html#datasets-and-blocks) from S3 data and inspect their schemas.

```python
import ray.data

import numpy as np

STORAGE_PATH = "s3://ray-example-data/cifar10-parquet"
train_dataset = ray.data.read_parquet(f'{STORAGE_PATH}/train')
test_dataset = ray.data.read_parquet(f'{STORAGE_PATH}/test')
train_dataset.schema()
test_dataset.schema()
```

Next, use Ray Data to transform the data. Note that both loading and transformation happen lazily, which means that only the training workers materialize the data.

```python
def transform_cifar(row: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # Define the torchvision transform.
    transform = transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    row["image"] = transform(row["image"])
    return row

train_dataset = train_dataset.map(transform_cifar)
test_dataset = test_dataset.map(transform_cifar)
```
Next, modify the training function you wrote earlier. Every difference from the previous script is highlighted and explained with a numbered comment; for example, “[1].”

```python
def train_func_per_worker(config: Dict):
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size_per_worker"]


    # [1] Prepare `Dataloader` for distributed training.
    # The get_dataset_shard method gets the associated dataset shard to pass to the 
    # TorchTrainer constructor in the next code block.
    # The iter_torch_batches method lazily shards the dataset among workers.
    # =============================================================================
    train_data_shard = ray.train.get_dataset_shard("train")
    valid_data_shard = ray.train.get_dataset_shard("valid")
    train_dataloader = train_data_shard.iter_torch_batches(batch_size=batch_size)
    valid_dataloader = valid_data_shard.iter_torch_batches(batch_size=batch_size)

    model = VisionTransformer(
        image_size=32,   # CIFAR-10 image size is 32x32
        patch_size=4,    # Patch size is 4x4
        num_layers=12,   # Number of transformer layers
        num_heads=8,     # Number of attention heads
        hidden_dim=384,  # Hidden size (can be adjusted)
        mlp_dim=768,     # MLP dimension (can be adjusted)
        num_classes=10   # CIFAR-10 has 10 classes
    )

    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # Model training loop.
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch}"):
            X, y = batch['image'], batch['label']
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss, num_correct, num_total, num_batches = 0, 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Valid Epoch {epoch}"):
                # [2] Each Ray Data batch is a dict so you must access the
                # underlying data using the appropriate keys.
                # =======================================================
                X, y = batch['image'], batch['label']
                pred = model(X)
                loss = loss_fn(pred, y)

                valid_loss += loss.item()
                num_total += y.shape[0]
                num_batches += 1
                num_correct += (pred.argmax(1) == y).sum().item()

        valid_loss /= num_batches
        accuracy = num_correct / num_total

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pt")
            )
            ray.train.report(
                metrics={"loss": valid_loss, "accuracy": accuracy},
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
            if ray.train.get_context().get_world_rank() == 0:
                print({"epoch_num": epoch, "loss": valid_loss, "accuracy": accuracy})
```
Finally, run the training function with the Ray Data Dataset on the Ray Cluster with 8 GPU workers.
```python
def train_cifar_10(num_workers, use_gpu):
    global_batch_size = 512

    train_config = {
        "lr": 1e-3,
        "epochs": 1,
        "batch_size_per_worker": global_batch_size // num_workers,
    }

    # Configure computation resources.
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    run_config = RunConfig(
        storage_path="/mnt/cluster_storage", 
        name=f"train_data_run-{uuid.uuid4().hex}",
    )

    # Initialize a Ray TorchTrainer.
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        # [1] With Ray Data you pass the Dataset directly to the Trainer.
        # ==============================================================
        datasets={"train": train_dataset, "valid": test_dataset},
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    result = trainer.fit()
    print(f"Training result: {result}")

if __name__ == "__main__":
    train_cifar_10(num_workers=8, use_gpu=True)
```

### [1] Preparing the DataLoader for Distributed Training:**

- **`ray.train.get_dataset_shard("train")`**: This function retrieves the dataset shard assigned to the current worker for the "train" dataset. In a distributed setting, each worker gets a portion (shard) of the dataset to process.
    
- **`iter_torch_batches(batch_size=batch_size)`**: This method converts the dataset shard into an iterable of PyTorch batches. Each batch is a dictionary where keys are column names, and values are tensors corresponding to that column. 

### [2] Accessing Data Within the Training Loop

- **`batch['image']` and `batch['label']`**: Each `batch` is a dictionary containing the data for the current iteration. Here, `'image'` and `'label'` are the keys, and their corresponding values are tensors representing the input images and their labels, respectively.
    
- **Model Prediction**: `model(X)` feeds the input data (`X`) through the model to obtain predictions.
    
- **Loss Calculation**: `loss_fn(pred, y)` computes the loss by comparing the model's predictions (`pred`) with the true labels (`y`).

### Summary

In a distributed training setup using Ray Train, each worker processes a shard of the dataset. The `get_dataset_shard` function retrieves the shard assigned to the current worker, and `iter_torch_batches` converts it into an iterable of PyTorch batches. Inside the training loop, each batch is accessed, and the model processes the input data to compute predictions and loss.

---

Which CRD?
https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html#kuberay-quickstart

---

