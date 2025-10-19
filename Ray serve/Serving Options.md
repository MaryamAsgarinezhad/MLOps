
# **vLLM**: 

Highest _out-of-the-box_ token throughput for decoder models on GPUs thanks to PagedAttention + continuous batching; simple **OpenAI-compatible** server; integrates with **Kubernetes via KServe**.

[arXiv](https://arxiv.org/pdf/2309.06180?utm_source=chatgpt.com)[ACM Digital Library](https://dl.acm.org/doi/10.1145/3600006.3613165?utm_source=chatgpt.com)[GitHub](https://github.com/vllm-project/vllm?utm_source=chatgpt.com)

### What Is KV Caching?

In transformer-based models, particularly during inference, the attention mechanism computes three components for each token: **Query (Q)**, **Key (K)**, and **Value (V)**. These components are used to determine the relevance of tokens to each other.

In autoregressive models like GPT, each new token generation depends on all previously generated tokens. Without caching, the model would need to recompute the K and V matrices for all prior tokens at each step, leading to redundant computations and increased latency.

**KV caching** addresses this by storing the K and V matrices from previous steps and reusing them in subsequent steps. This approach significantly reduces the computational overhead, leading to faster inference times.
During the first token generation, the K and V matrices are computed and stored.
For each new token, the model retrieves the cached K and V matrices and computes the attention for the new token only, rather than recalculating for all tokens.

### Store and Retrieve KV?

1. **Efficiency**: By caching K and V matrices, the model avoids redundant calculations, leading to faster token generation.
    
2. **Memory Management**: Storing these matrices allows for efficient memory usage, especially when dealing with long sequences.
    
3. **Scalability**: Efficient KV caching enables the deployment of large models in production environments, **handling multiple requests simultaneously**.

### **PagedAttention**

 Optimizes the storage and retrieval of attention key-value (KV) pairs in large language models (LLMs) on memory-constrained hardware (like GPUs). Inspired by operating system memory paging, it divides the KV cache into fixed-size blocks, allowing for efficient memory allocation and reducing fragmentation.

- **Memory Efficiency**: By allocating memory in fixed-size pages, PagedAttention *minimizes internal and external memory fragmentation,* leading to more efficient memory usage.
    
- **Flexible Memory Sharing**: It allows multiple sequences to share the same memory pages, optimizing GPU memory utilization and enabling larger batch sizes.
    
- **Dynamic Allocation**: Memory is allocated on-demand, ensuring that only the necessary resources are used, which is crucial for handling large models.

### **Continuous Batching**

A dynamic batching strategy that allows new inference requests to be processed immediately, without waiting for the current batch to complete.

- **Dynamic Scheduling**: As soon as a sequence completes its token generation, it is replaced with a new request. While one user’s response is being generated, other user's request starts being processed

---

# **Triton + TensorRT-LLM**:

Best for _squeezing every ounce_ of NVIDIA GPU performance (FP8/INT4, inflight batching, fused kernels), strict latency SLOs, and mixed multi-framework fleet; more setup (engine build, model repo).

[NVIDIA Docs+1](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/trtllm_user_guide.html?utm_source=chatgpt.com)[GitHub+1](https://github.com/NVIDIA/TensorRT-LLM?utm_source=chatgpt.com)


### **Technical Features**

- **FP8/INT4**: Low-precision formats (8-bit floating point and 4-bit integer) used to reduce memory usage and accelerate computations. [Baseten](https://www.baseten.co/blog/fp8-efficient-model-inference-with-8-bit-floating-point-numbers/?utm_source=chatgpt.com)
- **In-flight batching**: A technique where the system processes new requests while previous ones are still being handled, improving throughput. [NVIDIA GitHub](https://nvidia.github.io/TensorRT-LLM/overview.html?utm_source=chatgpt.com)
- **Fused kernels**: Combining multiple individual operations into a single kernel to reduce overhead and improve execution efficiency. [ApX Machine Learning](https://apxml.com/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations/low-precision-kernel-generation?utm_source=chatgpt.com)

**"Kernel"** refers to a **function or computation** that is executed on the GPU (or other accelerators like TPUs). Specifically, in the context of machine learning and deep learning, a kernel is a small, highly-optimized program that performs a specific task, like a matrix multiplication, addition, or activation function.
### **Performance & Scalability**

- **Strict latency SLOs**: Ensuring that response times meet predefined Service Level Objectives, crucial for real-time applications.
- **Mixed multi-framework fleets**: Deploying and managing models from different frameworks (e.g., TensorFlow, PyTorch) in a unified system.

### **Setup Considerations**

- **Engine build**: The process of compiling models into optimized formats for efficient inference.
- **Model repository**: A storage system for managing and versioning machine learning models.


**NVIDIA Triton Inference Server**

- **Purpose**: Triton is an open-source inference server designed to manage and deploy machine learning models at scale. It supports serving models from multiple frameworks, such as TensorFlow, PyTorch, ONNX, and custom frameworks.
    
- **Features**:
    
    - Supports multiple backends like TensorRT, TensorFlow, and PyTorch for serving models.
        
    - Offers features like **multi-model support**, **batching**, **model versioning**, and **model ensembles**.
        
    - Provides **scalability** and **load balancing** to manage inference workloads across multiple GPUs and nodes.
        
    - **Optimized for deployment at scale** in production environments and supports cloud-native environments, making it suitable for serving large-scale models.
        
    - **Supports both CPU and GPU** execution with optimizations for NVIDIA GPUs.
    
- **Use Case**: Triton is ideal when you need to serve multiple types of models from different frameworks with management and scaling features. It excels in large-scale, multi-framework, and multi-GPU inference workloads.


**NVIDIA TensorRT**

- **Purpose**: TensorRT is a **high-performance deep learning inference library** for optimizing and running models on NVIDIA GPUs. It's designed specifically for **inference optimization** rather than model deployment, providing optimizations like kernel fusion, precision reduction (e.g., FP16, INT8), and more efficient memory management.
    
- **Features**:
    
    - Optimizes models by **reducing their memory footprint** and **increasing execution speed** on GPUs.
        
    - Works best with models built using TensorFlow, PyTorch, and ONNX, converting them into efficient **TensorRT engines** for deployment.
        
    - Provides tools to **quantize models**, which significantly reduces the model size and improves performance without much loss in accuracy.
        
- **Use Case**: TensorRT is optimal for users who have **GPU-focused workloads** and want to **optimize model inference** for performance. It is more suitable when you need to **optimize a model for real-time inference**, especially when memory and latency are critical.

##### **Model Backend**

A **model backend** refers to the system or library responsible for **executing the machine learning model** during inference. It performs the actual computations (forward pass) using the model's architecture and parameters. In the context of Triton Inference Server, a backend is a plugin or runtime that executes models served by Triton.

- **Triton and Model Backends**: Triton can serve models from different **backends** like:
    
    - **TensorRT**: Used for high-performance optimization of models on NVIDIA GPUs.
        
    - **TensorFlow, PyTorch, ONNX**: Used for serving models trained in TensorFlow, PyTorch, or ONNX.
        
    - **Custom backends**: You can also implement custom backends for specialized model serving needs.

**The model backend handles the actual computation part of the model**, whereas Triton serves as the **inference management platform** that provides scalability, load balancing, and multi-framework support.

---

# **Ray Serve**

Great for _full applications_ (retrieval, routing, post-processing) with Python DAGs, autoscaling, and multi-model graphs; now includes “Serve LLM” helpers for OpenAI-compatible endpoints. Strong choice when serving isn’t just one model call.

- **Retrieval**: Fetching relevant data or information, such as retrieving documents from a database or retrieving embeddings from a vector store.
    
- **Routing**: Directing data to appropriate models or services based on certain criteria, like determining which model to use for a specific task.
    
- **Post-processing**: Transforming the model's output into a desired format, such as formatting a response or aggregating results.

In Ray Serve, DAGs (a collection of tasks and their dependencies, ensuring that tasks are executed in a specific order) are used to model the sequence of operations in a machine learning workflow, ensuring that each component (e.g., retrieval, routing, processing) is executed in the correct order.

[Ray+2Ray+2](https://docs.ray.io/en/latest/serve/index.html?utm_source=chatgpt.com)

---

# **KServe (Knative on K8s)**

Cloud-native control plane for model servers (including vLLM, Hugging Face runtime, Triton); serverless autoscaling, scale-to-zero, and new model-caching features. Choose when you already run on Kubernetes and want standardized multi-team ops.

[kserve.github.io](https://kserve.github.io/website/docs/model-serving/generative-inference/overview?utm_source=chatgpt.com)[CNCF](https://www.cncf.io/blog/2025/06/18/announcing-kserve-v0-15-advancing-generative-ai-model-serving/?utm_source=chatgpt.com)

---

