
## What’s a RayService?

A RayService manages these components:

- **RayCluster**: Manages resources in a Kubernetes cluster.
- **Ray Serve Applications**: Manages users’ applications.

## What does the RayService provide?

- **Kubernetes-native support for Ray clusters and Ray Serve applications:** After using a Kubernetes configuration to define a Ray cluster and its Ray Serve applications, you can use `kubectl` to create the cluster and its applications.
- **In-place updating for Ray Serve applications**
- **Zero downtime upgrading for Ray clusters**
- **High-availabilable services:** See [RayService high availability](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/rayservice-high-availability.html#kuberay-rayservice-ha) for more details.


----
## Step 1: Create a Kubernetes cluster with Kind

kind create cluster --image=kindest/node:v1.26.0

## Step 2: Install the KubeRay operator

Install the latest stable KubeRay operator from the Helm repository. Note that the YAML file in this example uses `serveConfigV2` (**takes a yaml multi-line scalar, which should be a Ray Serve multi-application config. See https://docs.ray.io/en/latest/serve/multi-app.html.**) to specify a multi-application Serve configuration (at different endpoints of the Ray Service URL), available starting from **KubeRay v0.6.0.**

## Step 3: Install a RayService

kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/v1.4.2/ray-operator/config/samples/ray-service.sample.yaml

- This .yaml config contains both **RayService** and **RayCluster** Configurations, so we only need to install Kuberay before.

## Step 4: Verify the Kubernetes cluster status

```yaml
# Step 4.1: List all RayService custom resources in the `default` namespace.
kubectl get rayservice

# Step 4.2: List all RayCluster custom resources in the `default` namespace.
kubectl get raycluster

# Step 4.3: List all Ray Pods in the `default` namespace.
kubectl get pods -l=ray.io/is-ray-node=yes

# [Example output]
# NAME                                               READY   STATUS    RESTARTS   AGE
# rayservice-sample-cxm7t-head                       1/1     Running   0          3m5s
# rayservice-sample-cxm7t-small-group-worker-8hrgg   1/1     Running   0          3m5s

# Step 4.4: Check the `Ready` condition of the RayService.
# The RayService is ready to serve requests when the condition is `True`.
kubectl describe rayservices.ray.io rayservice-sample

# [Example output]
# Conditions:
#   Last Transition Time:  2025-06-26T13:23:06Z
#   Message:               Number of serve endpoints is greater than 0
#   Observed Generation:   1
#   Reason:                NonZeroServeEndpoints
#   Status:                True
#   Type:                  Ready

# Step 4.5: List services in the `default` namespace.
kubectl get services

# NAME                               TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                                         AGE
# ...
# rayservice-sample-cxm7t-head-svc   ClusterIP   None            <none>        10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP   71m
# rayservice-sample-head-svc         ClusterIP   None            <none>        10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP   70m
# rayservice-sample-serve-svc        ClusterIP   10.96.125.107   <none>        8000/TCP                                        70m
```

When the Ray Serve applications are healthy and ready, KubeRay creates a **head service** and a **Ray Serve service** for the RayService custom resource. For example, `rayservice-sample-head-svc` and `rayservice-sample-serve-svc` in Step 4.5.

> **What do these services do?**

- **`rayservice-sample-head-svc`**  
    This service points to the **head pod** of the active RayCluster and is typically used to view the **Ray Dashboard** (port `8265`).
    
- **`rayservice-sample-serve-svc`**  
    This service exposes the **HTTP interface** of Ray Serve, typically on port `8000`.  
    Use this service to send HTTP requests to your deployed Serve applications (e.g., REST API, ML inference, etc.).


## Step 5: Verify the status of the Serve applications

```shell
# (1) Forward the dashboard port to localhost.
# (2) Check the Serve page in the Ray dashboard at http://localhost:8265/#/serve.
kubectl port-forward svc/rayservice-sample-head-svc 8265:8265
```

- Refer to [rayservice-troubleshooting.md](https://docs.ray.io/en/latest/cluster/kubernetes/troubleshooting/rayservice-troubleshooting.html#kuberay-raysvc-troubleshoot) for more details on RayService observability.

## Step 6: Send requests to the Serve applications by the Kubernetes serve service

```shell
# Step 6.1: Run a curl Pod.
# This command spins up a lightweight Kubernetes Pod that includes the `curl` tool
kubectl run curl --image=radial/busyboxplus:curl -i --tty

# Step 6.2: Send a POST request to the `/fruit/` endpoint of the Serve application.
curl -X POST -H 'Content-Type: application/json' rayservice-sample-serve-svc:8000/fruit/ -d '["MANGO", 2]'
# [Expected output]: 6

# Step 6.3: Send a request to the calculator app.
curl -X POST -H 'Content-Type: application/json' rayservice-sample-serve-svc:8000/calc/ -d '["MUL", 3]'
# [Expected output]: "15 pizzas please!"
```

**Why to use a curl pod?**

- Kubernetes services like `rayservice-sample-serve-svc` are typically **ClusterIP** services, meaning they're accessible only from inside the cluster.
    
- Running this Pod gives you an in-cluster environment from which you **can send HTTP requests** to your deployed Serve applications.

## Step 7: Clean up the Kubernetes cluster

```shell
# Delete the RayService.
kubectl delete -f https://raw.githubusercontent.com/ray-project/kuberay/v1.4.2/ray-operator/config/samples/ray-service.sample.yaml

# Uninstall the KubeRay operator.
helm uninstall kuberay-operator

# Delete the curl Pod.
kubectl delete pod curl
```
 ---

# Ray Serve: Scalable and Programmable Serving

- The python library used to communicate with a RayService on kubernetes.

Ray Serve is a scalable model serving library for building online inference APIs. Serve is 
framework-agnostic, so you can use a single toolkit to serve everything from deep learning models built with frameworks like PyTorch, TensorFlow, and Keras, to Scikit-Learn models, to arbitrary Python business logic. It has several features and performance optimizations for serving Large Language Models such as response streaming, dynamic request batching, multi-node/multi-GPU serving, etc.

Ray Serve is built on top of Ray, so it easily scales to many machines and offers flexible scheduling support such as fractional GPUs so you can share resources and serve many machine learning models at low cost.

#### This tutorial will walk you through the process of writing and testing a Ray Serve application. It will show you how to

- convert a machine learning model to a Ray Serve deployment
- test a Ray Serve application locally over HTTP
- compose multiple-model machine learning models together into a single application

```shell
pip install "ray[serve]" transformers requests torch
```

## Text Translation Model (before Ray Serve)

```python
# File name: model.py
from transformers import pipeline


class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation


translator = Translator()

translation = translator.translate("Hello world!")
print(translation)
```
## Converting to a Ray Serve Application

In this section, we’ll deploy the text translation model using Ray Serve, so it can be scaled up and queried over HTTP. We’ll start by converting `Translator` into a Ray Serve deployment.

```python
# File name: serve_quickstart.py
from starlette.requests import Request

import ray
from ray import serve

from transformers import pipeline


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2, "num_gpus": 0})
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


translator_app = Translator.bind()
# we need to `bind` our `Translator` deployment to arguments that will be passed into its constructor. Since `Translator`’s constructor doesn’t take in any arguments, we can call the deployment’s `bind` method without passing anything in 
```

#### The `Translator` class has two modifications:

1. It has a decorator, **`@serve.deployment`**.

	**The decorator converts `Translator` from a Python class into a ==Ray Serve `Deployment` object==.**
	
	Each deployment stores a single Python function or class that you write and uses it to serve requests. You can scale and configure each of your deployments independently using parameters in the `@serve.deployment` decorator. The example configures a few common parameters:
	
	- **`num_replicas`**: an integer that determines how many copies of our deployment process run in Ray. Requests are load balanced across these replicas, allowing you to scale your deployments horizontally.
		
	- **`ray_actor_options`**: a dictionary containing configuration options for each replica.
		
		- `num_cpus`: a float representing the logical number of CPUs each replica should reserve. You can make this a fraction to pack multiple replicas together on a machine with fewer CPUs than replicas.
			
		- `num_gpus`: a float representing the logical number of GPUs each replica should reserve. You can make this a fraction to pack multiple replicas together **on a machine with fewer GPUs than replicas.**
			
		- `resources`: a dictionary containing other resource requirements for the replicate, such as non-GPU accelerators like HPUs or TPUs.

2. It has a new method, **`__call__`**.
	
	Deployments receive Starlette HTTP `request` objects. By default, the deployment class’s `__call__` method is called on this `request` object. The return value is sent back in the HTTP response body.
	
	This is why `Translator` needs a new `__call__` method. The method processes the incoming HTTP request by reading its JSON data and forwarding it to the `translate` method. The translated text is returned and sent back through the HTTP response. You can also use Ray Serve’s FastAPI integration to avoid working with raw HTTP requests. Check out [FastAPI HTTP Deployments](https://docs.ray.io/en/latest/serve/http-guide.html#serve-fastapi-http) for more info about FastAPI with Serve.


## Running a Ray Serve Application

To test **locally**, we run the script with the **`serve run`** CLI command. This command takes in an import path to our deployment formatted as `module:application`. Make sure to run the command from a directory containing a local copy of this script saved as `serve_quickstart.py`, so it can import the application:

```shell
serve run serve_quickstart:translator_app
```

This command will run the `translator_app` application and then block, streaming logs to the console. It can be killed with `Ctrl-C`, which will tear down the application.

We can now test our model over HTTP. It can be reached at the following URL by default:

```shell
http://127.0.0.1:8000/
```

We’ll send a POST request with JSON data containing our English text. `Translator`’s `__call__` method will unpack this text and forward it to the `translate` method. Here’s a client script that requests a translation for “Hello world!”:

```python
# File name: model_client.py
import requests

english_text = "Hello world!"

response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
```

---

## Composing Multiple Models

Ray Serve allows you to compose multiple deployments into a single Ray Serve application. This makes it easy to combine multiple machine learning models along with business logic to serve a single request.

1. Summarize English text
2. Translate the summary into French

```python
# File name: serve_quickstart_composed.py
from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import pipeline


@serve.deployment
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)

        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]

        return translation


@serve.deployment
class Summarizer:
    def __init__(self, translator: DeploymentHandle):
        self.translator = translator

        # Load model.
        self.model = pipeline("summarization", model="t5-small")

    def summarize(self, text: str) -> str:
        # Run inference
        model_output = self.model(text, min_length=5, max_length=15)

        # Post-process output to return only the summary text
        summary = model_output[0]["summary_text"]

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        translation = await self.translator.translate.remote(summary)
        return translation


app = Summarizer.bind(Translator.bind())
```

This script contains our `Summarizer` class converted to a deployment and our `Translator` class with some modifications. In this script, **the `Summarizer` class contains the `__call__` method since requests are sent to it first**. It also **takes in a handle to the `Translator`** as one of its constructor arguments, so it can forward summarized texts to the `Translator` deployment. The `__call__` method also contains some new code:

```python
translation = await self.translator.translate.remote(summary)
```

`self.translator.translate.remote(summary)` issues an **asynchronous call to the `Translator`**’s `translate` method and returns a `DeploymentResponse` object immediately. ==Calling `await` on the response waits for the remote method call to execute and returns its return value.== The response could also be passed directly to another **`DeploymentHandle`** call

---

# Production Guide

The recommended way to run Ray Serve in production is on Kubernetes using the [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html#kuberay-quickstart) [RayService](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayservice-quick-start.html#kuberay-rayservice-quickstart) custom resource. The RayService custom resource automatically handles important production requirements such as health checking, status reporting, failure recovery, and upgrades. If you’re not running on Kubernetes, you can also run Ray Serve on a Ray cluster directly using the Serve CLI.

## Working example: Text summarization and translation application

Throughout the production guide, we will use the before-mentioned  Serve application as a working example. The application takes in a string of text in English, then summarizes and translates it into French (default), German, or Romanian.

Save this code locally in `text_ml.py`. In development, we would likely use the `serve run` command to iteratively run, develop, and repeat (see the [Development Workflow](https://docs.ray.io/en/latest/serve/advanced-guides/dev-workflow.html#serve-dev-workflow) for more information). When we’re ready to go to production, we will generate a structured [config file](https://docs.ray.io/en/latest/serve/production-guide/config.html#serve-in-production-config-file) that acts as the single source of truth for the application.

This config file can be generated using **`serve build`**:

```shell
$ serve build text_ml:app -o serve_config.yaml
```

The generated version of this file contains an `import_path`, `runtime_env`, and configuration options for each deployment in the application. The application needs the `torch` and `transformers` packages, so modify the `runtime_env` field of the generated config to include these two pip packages. Save this config locally in `serve_config.yaml`.

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
      - torch
      - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

You can use `serve deploy` to deploy the application to a local Ray cluster and `serve status` to get the status at runtime:

```shell
# Start a local Ray cluster.
ray start --head

# Deploy the Text ML application to the local Ray cluster.
serve deploy serve_config.yaml
2022-08-16 12:51:22,043 SUCC scripts.py:180 --
Sent deploy request successfully!
 * Use `serve status` to check deployments' statuses.
 * Use `serve config` to see the running app's config.

serve status
proxies:
  cef533a072b0f03bf92a6b98cb4eb9153b7b7c7b7f15954feb2f38ec: HEALTHY
applications:
  default:
    status: RUNNING
    message: ''
    last_deployed_time_s: 1694041157.2211847
    deployments:
      Translator:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
      Summarizer:
        status: HEALTHY
        replica_states:
          RUNNING: 1
        message: ''
```

Test the application using Python `requests`:

```python
import requests

english_text = (
    "It was the best of times, it was the worst of times, it was the age "
    "of wisdom, it was the age of foolishness, it was the epoch of belief"
)
response = requests.post("http://127.0.0.1:8000/", json=english_text)
french_text = response.text

print(french_text)
# 'c'était le meilleur des temps, c'était le pire des temps .'
```
## Next Steps

- Learn how to [monitor running Serve applications](https://docs.ray.io/en/latest/serve/monitoring.html#serve-monitoring).

---

# Serve Config Files

This section should help you:

- Understand the Serve config file format.
- Learn how to deploy and update your applications in production using the Serve config.
- Learn how to generate a config file for a list of Serve applications.


The Serve config is the recommended way to deploy and update your applications in production. It allows you to fully configure everything related to Serve, including system-level components like the proxy and application-level options like individual deployment parameters (recall how to [configure Serve deployments](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html#serve-configure-deployment)). One major benefit is you can dynamically update individual deployment parameters by modifying the Serve config, without needing to redeploy or restart your application.

- If you are deploying Serve on a VM, you can use the Serve config with the [serve deploy](https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#serve-in-production-deploying) CLI command. If you are deploying Serve on Kubernetes, you can embed the Serve config in a [RayService](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html#serve-in-production-kubernetes) custom resource in Kubernetes to


#### The Serve config is a YAML file with the following format:

```yaml
proxy_location: ...

http_options: 
  host: ...
  port: ...
  request_timeout_s: ...
  keep_alive_timeout_s: ...

grpc_options:
  port: ...
  grpc_servicer_functions: ...
  request_timeout_s: ...

logging_config:
  log_level: ...
  logs_dir: ...
  encoding: ...
  enable_access_log: ...

applications:
- name: ...
  route_prefix: ...
  import_path: ...
  runtime_env: ... 
  deployments:
  - name: ...
    num_replicas: ...
    ...
  - name:
    ...
```

The file contains **`proxy_location`, `http_options`, `grpc_options`, `logging_config`** and **`applications`**.

The **`proxy_location`** field configures where to run proxies (**Ray Serve web servers**) to handle traffic to the cluster. You can set `proxy_location` to the following values:

- EveryNode (default): Run a proxy on every node in the cluster that has at least one replica actor.
    
- HeadOnly: Only run a single proxy on the head node.
    
- Disabled: Don’t run proxies at all. Set this value if you are only making calls to your applications using deployment handles.


The **`http_options`** are as follows. Note that the HTTP config is global to your Ray cluster, and you can’t update it during runtime.

- **`host`**: The host IP address for Serve’s HTTP proxies. This is optional and can be omitted. By default, the `host` is set to `0.0.0.0` to expose your deployments publicly. If you’re using Kubernetes, you must set `host` to `0.0.0.0` to expose your deployments outside the cluster.
    
- **`port`**: The port for Serve’s HTTP proxies. This parameter is optional and can be omitted. By default, the port is set to `8000`.
    
- **`request_timeout_s`**: Allows you to set the end-to-end timeout for a request before terminating and retrying at another replica. By default, there is no request timeout.
    
- **`keep_alive_timeout_s`**: Allows you to set the keep alive timeout for the HTTP proxy. For more details, see [here](https://docs.ray.io/en/latest/serve/http-guide.html#serve-http-guide-keep-alive-timeout)


The **`grpc_options`** are as follows. Note that the gRPC config is global to your Ray cluster, and you can’t update it during runtime.

- **`port`**: The port that the gRPC proxies listen on. These are optional settings and can be omitted. By default, the port is set to `9000`.
    
- **`grpc_servicer_functions`**: gRPC defines services in `.proto` files and generates Python code. That code includes a function like `add_MyServiceServicer_to_server`. This option tells Ray Serve which gRPC services to add to its proxy.List of import paths for gRPC `add_servicer_to_server` functions to add to Serve’s gRPC proxy. The servicer functions need to be importable from the context of where Serve is running. This defaults to an empty list, which means the gRPC server isn’t started.
    
- **`request_timeout_s`**: Allows you to set the end-to-end timeout for a request before terminating and retrying at another replica. By default, there is no request timeout.

##### Ray Serve supports **two entry points**:

1. HTTP proxies (via REST API calls).
2. gRPC proxies (via gRPC function calls).


The **`logging_config`** is global config, you can configure controller & proxy & replica logs. Note that you can also set application and deployment level logging config, which will take precedence over the global config. See logging config API [here](https://docs.ray.io/en/latest/serve/api/doc/ray.serve.schema.LoggingConfig.html) for more details.

These are the fields per application:

- **`name`**: The names for each application that are auto-generated by `serve build`. The name of each application must be unique.
    
- **`route_prefix`**: An application can be called via HTTP at the specified route prefix. It defaults to `/`. The route prefix for each application must be unique.
    
- **`import_path`**: The path to your top-level Serve deployment (or the same path passed to `serve run`). The most minimal config file consists of only an `import_path`.
    
- **`runtime_env`**: Defines the environment that the application runs in. Use this parameter to package application dependencies such as `pip` packages (see [Runtime Environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments) for supported fields). The `import_path` must be available _within_ the `runtime_env` if it’s specified. The Serve config’s `runtime_env` can only use [remote URIs](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#remote-uris) in its `working_dir` and `py_modules`; it can’t use local zip files or directories. [More details on runtime env](https://docs.ray.io/en/latest/serve/production-guide/handling-dependencies.html#serve-runtime-env).
    
- **`deployments (optional)`**: A list of deployment options that allows you to override the `@serve.deployment` settings specified in the deployment graph code. Each entry in this list must include the deployment `name`, which must match one in the code. If this section is omitted, Serve launches all deployments in the graph with the parameters specified in the code. See how to [configure serve deployment options](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html#serve-configure-deployment).
    
- **`args`**: Arguments that are passed to the [application builder](https://docs.ray.io/en/latest/serve/advanced-guides/app-builder-guide.html#serve-app-builder-guide).

```yaml
proxy_location: EveryNode

http_options:
  host: 0.0.0.0
  port: 8000

applications:
- name: default
  route_prefix: /
  import_path: text_ml:app
  runtime_env:
    pip:
      - torch
      - transformers
  deployments:
  - name: Translator
    num_replicas: 1
    user_config:
      language: french
  - name: Summarizer
    num_replicas: 1
```

The file uses the same `text_ml:app` import path that was used with `serve run`(==serve run is for local running of Ray Serve applications==), and **has two entries in the `deployments` list for the translation and summarization deployments**. Both entries contain a `name` setting and some other configuration options such as `num_replicas`. This configuration can also be defined in `@serve.deployment` decorator from the application’s code, and is not necessarily defined here. 

## Dynamically change parameters without restarting replicas (`user_config`)

You can use the `user_config` field to supply a structured configuration for your deployment. You can pass arbitrary JSON serializable objects to the YAML configuration. Serve then applies it to all running and future deployment replicas. **The application of user configuration _doesn’t_ restart the replica**. This deployment continuity means that you can use this field to dynamically:

- adjust model weights and versions without restarting the cluster.
- adjust traffic splitting percentage for your model composition graph.
- configure any feature flag, A/B tests, and hyper-parameters for your deployments.

To enable the `user_config` feature, implement a **`reconfigure`** method that takes a JSON-serializable object (e.g., a Dictionary, List, or String) as its only argument:

```python
@serve.deployment
class Model:
    def reconfigure(self, config: Dict[str, Any]):
        self.threshold = config["threshold"]
```

If you set the `user_config` when you create the deployment below, Ray Serve calls this `reconfigure` method **right after the deployment’s `__init__` method**, and passes the `user_config` in as an argument. You can also **trigger the `reconfigure` method by updating your Serve config file with a new `user_config` and reapplying it to the Ray cluster.** See [In-place Updates](https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html#serve-inplace-updates) for more information.

The corresponding YAML snippet is:
```yaml
deployments:
    - name: Model
      user_config:
        threshold: 1.5
```

---

# Deploy on Kubernetes

This section should help you:

- understand how to install and use the [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html#kuberay-quickstart) operator.
- understand how to deploy a Ray Serve application using a [RayService](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayservice-quick-start.html#kuberay-rayservice-quickstart).
- understand how to monitor and update your application.

A [RayService](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/rayservice-quick-start.html#kuberay-rayservice-quickstart) CR encapsulates a **multi-node Ray Cluster** and a **Serve application that runs on top of it** into a single Kubernetes manifest. Deploying, upgrading, and getting the status of the application can be done using standard `kubectl` commands. This section walks through how to deploy, monitor, and upgrade the above example on Kubernetes.

## Installing the KubeRay operator
- Discussed before

## Setting up a RayService custom resource (CR)

Once the KubeRay controller is running, manage your Ray Serve application by creating and updating a `RayService` CR (Discussed before)

Under the `spec` section in the `RayService` CR, set the following fields:

**`serveConfigV2`**: Represents the configuration that Ray Serve uses to deploy the application. Using `serve build` to print the Serve configuration and copy-paste it directly into your [Kubernetes config](https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html#serve-in-production-kubernetes) and `RayService` CR.

**`rayClusterConfig`**: Populate this field with the contents of the `spec` field from the `RayCluster` CR YAML file. Refer to [KubeRay configuration](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html#kuberay-config) for more details.

**Tip**: 
	To enhance the reliability of your application, particularly when dealing with large dependencies that may require a significant amount of time to download, consider **including the dependencies in your image’s Dockerfile**, so the dependencies are available as soon as the pods start.
	If you have dependencies that must be installed during deployment, you can add them to the `runtime_env` in the Deployment code. Learn more [here](https://docs.ray.io/en/latest/serve/production-guide/handling-dependencies.html#serve-handling-dependencies)

## Deploying a Serve application

When the `RayService` is created, the `KubeRay` controller first creates a Ray cluster using the provided configuration. Then, once the cluster is running, it deploys the Serve application to the cluster using the [REST API](https://docs.ray.io/en/latest/serve/advanced-guides/deploy-vm.html#serve-in-production-deploying). The controller also creates a Kubernetes Service that can be used to route traffic to the Serve application.

To see an example, save this CR locally to a file named `ray-service.text-ml.yaml`:

```yaml
# Make sure to increase resource requests and limits before using this example in production.
# For examples with more realistic resource configuration, see
# ray-cluster.complete.large.yaml and
# ray-cluster.autoscaler.large.yaml.
apiVersion: ray.io/v1alpha1
kind: RayService
metadata:
  name: rayservice-sample
spec:
  serviceUnhealthySecondThreshold: 900 # Config for the health check threshold for Ray Serve applications. Default value is 900.
  deploymentUnhealthySecondThreshold: 300 # Config for the health check threshold for Ray dashboard agent. Default value is 300.
  # serveConfigV2 takes a yaml multi-line scalar, which should be a Ray Serve multi-application config. See https://docs.ray.io/en/latest/serve/multi-app.html.
  # Only one of serveConfig and serveConfigV2 should be used.
  serveConfigV2: |
    applications:
      - name: text_ml_app
        import_path: text_ml.app
        route_prefix: /summarize_translate
        runtime_env:
          working_dir: "https://github.com/ray-project/serve_config_examples/archive/36862c251615e258a58285934c7c41cffd1ee3b7.zip"
          pip:
            - torch
            - transformers
        deployments:
          - name: Translator
            num_replicas: 1
            ray_actor_options:
              num_cpus: 0.2
            user_config:
              language: french
          - name: Summarizer
            num_replicas: 1
            ray_actor_options:
              num_cpus: 0.2
  rayClusterConfig:
    rayVersion: '2.6.3' # should match the Ray version in the image of the containers
    ######################headGroupSpecs#################################
    # Ray head pod template.
    headGroupSpec:
      # The `rayStartParams` are used to configure the `ray start` command.
      # See https://github.com/ray-project/kuberay/blob/master/docs/guidance/rayStartParams.md for the default settings of `rayStartParams` in KubeRay.
      # See https://docs.ray.io/en/latest/cluster/cli.html#ray-start for all available options in `rayStartParams`.
      rayStartParams:
        dashboard-host: '0.0.0.0'
      #pod template
      template:
        spec:
          containers:
            - name: ray-head
              image: rayproject/ray:2.6.3
              resources:
                limits:
                  cpu: 2
                  memory: 2Gi
                requests:
                  cpu: 2
                  memory: 2Gi
              ports:
                - containerPort: 6379
                  name: gcs-server
                - containerPort: 8265 # Ray dashboard
                  name: dashboard
                - containerPort: 10001
                  name: client
                - containerPort: 8000
                  name: serve
    workerGroupSpecs:
      # the pod replicas in this group typed worker
      - replicas: 1
        minReplicas: 1
        maxReplicas: 5
        # logical group name, for this called small-group, also can be functional
        groupName: small-group
        # The `rayStartParams` are used to configure the `ray start` command.
        # See https://github.com/ray-project/kuberay/blob/master/docs/guidance/rayStartParams.md for the default settings of `rayStartParams` in KubeRay.
        # See https://docs.ray.io/en/latest/cluster/cli.html#ray-start for all available options in `rayStartParams`.
        rayStartParams: {}
        #pod template
        template:
          spec:
            containers:
              - name: ray-worker # must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc'
                image: rayproject/ray:2.6.3
                lifecycle:
                  preStop:
                    exec:
                      command: ["/bin/sh","-c","ray stop"]
                resources:
                  limits:
                    cpu: "1"
                    memory: "2Gi"
                  requests:
                    cpu: "500m"
                    memory: "2Gi"
```

Content inside runtime_env that contain web server scripts:
![[Pasted image 20250907123014.png]]
	In the above config since we got `route_prefix: /summarize_translate`, requests must be sent to `localhost:8000/summarize_translate`.


To deploy the example, we simply `kubectl apply` the CR. This creates the underlying Ray cluster, consisting of a head and worker node pod, as well as the service that can be used to query our application:

```shell
kubectl apply -f ray-service.text-ml.yaml

kubectl get rayservices
NAME                SERVICE STATUS   NUM SERVE ENDPOINTS
rayservice-sample   Running          1

kubectl get pods
NAME                                                          READY   STATUS    RESTARTS   AGE
rayservice-sample-raycluster-7wlx2-head-hr8mg                 1/1     Running   0          XXs
rayservice-sample-raycluster-7wlx2-small-group-worker-tb8nn   1/1     Running   0          XXs

kubectl get services
NAME                                          TYPE        CLUSTER-IP        EXTERNAL-IP   PORT(S)                                         AGE
rayservice-sample-head-svc                    ClusterIP   None              <none>        10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP   XXs
rayservice-sample-raycluster-7wlx2-head-svc   ClusterIP   None              <none>        10001/TCP,8265/TCP,6379/TCP,8080/TCP,8000/TCP   XXs
rayservice-sample-serve-svc                   ClusterIP   192.168.145.219   <none>        8000/TCP                                        XXs
```

## Querying the application

Once the `RayService` is running, we can query it over HTTP using the service created by the KubeRay controller. This service can be queried directly from inside the cluster, but to access it from your laptop you’ll need to configure a [Kubernetes ingress](https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html#kuberay-networking) or use port forwarding as below:

```shell
kubectl port-forward service/rayservice-sample-serve-svc 8000
curl -X POST -H "Content-Type: application/json" localhost:8000/summarize_translate -d '"It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief"'
c'était le meilleur des temps, c'était le pire des temps .
```

## Getting the status of the application

As the `RayService` is running, the `KubeRay` controller continually monitors it and writes relevant status updates to the CR. You can view the status of the application using `kubectl describe`. This includes the status of the cluster, events such as health check failures or restarts, and the application-level statuses reported by [`serve status`](https://docs.ray.io/en/latest/serve/monitoring.html#serve-in-production-inspecting).

```yaml
kubectl get rayservices
kubectl describe rayservice rayservice-sample
```

## Updating the application

To update the `RayService`, modify the manifest and apply it use `kubectl apply`. There are two types of updates that can occur:

- _Application-level updates_: when only the Serve config options are changed, the update is applied _in-place_ on the same Ray cluster. This enables [lightweight updates](https://docs.ray.io/en/latest/serve/advanced-guides/inplace-updates.html#serve-in-production-lightweight-update) such as scaling a deployment up or down or modifying autoscaling parameters.
    
- _Cluster-level updates_: when the `RayCluster` config options are changed, such as updating the container image for the cluster, it may result in a cluster-level update. In this case, a new cluster is started, and the application is deployed to it. Once the new cluster is ready, the Kubernetes service is updated to point to the new cluster and the previous cluster is terminated. There should not be any downtime for the application, but note that this requires the Kubernetes cluster to be large enough to schedule both Ray clusters.

![[Pasted image 20250907120930.png]]
![[Pasted image 20250907120948.png]]

## Autoscaling

You can configure autoscaling for your Serve application by setting the autoscaling field in the Serve config. Learn more about the configuration options in the [Serve Autoscaling Guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling).

To enable autoscaling in a KubeRay Cluster, you need to set `enableInTreeAutoscaling` to True. Additionally, there are other options available to configure the autoscaling behavior. For further details, please refer to the documentation [here](https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling).

---


