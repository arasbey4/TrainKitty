TrainKitty 🐈

Describe it. Train it. Run it.

TrainKitty is an open-source, local-first AI agent designed to help you build, fine-tune, evaluate, and export custom language models using locally running Ollama models and the Hugging Face ecosystem.

Instead of manually assembling training scripts, datasets, configurations, and evaluation pipelines, you describe the model you want to create. TrainKitty’s local AI agent helps plan and orchestrate the workflow, while dedicated training tools perform the actual machine learning operations.

✨ Vision

Make custom language model development more accessible without hiding the complexity or pretending that work has been completed when it has not.

TrainKitty aims to provide a unified workflow for:

* Defining a custom model’s purpose, personality, instructions, and behavior.
* Generating synthetic training datasets using a local Ollama model.
* Importing and preparing real datasets from Hugging Face.
* Selecting compatible base models and training methods.
* Checking available hardware and system requirements before training.
* Running real fine-tuning jobs and tracking actual metrics.
* Evaluating trained models against their original base models.
* Exporting model artifacts and integrating compatible models with Ollama.

🧠 Local AI Agent

TrainKitty uses a locally running Ollama model as its orchestration agent.

The agent interprets the user’s goal, develops a training plan, selects appropriate supported tools, and coordinates the workflow.

Ollama is the agent’s reasoning and orchestration layer, not the training engine. Actual training is performed by compatible machine learning frameworks and tools.

The project is designed around local inference, without requiring external AI inference APIs.

🏗️ How It Works

1. Describe the model you want to build.
2. Prepare a dataset by generating examples locally or importing an existing dataset.
3. Validate the dataset, model compatibility, dependencies, and hardware requirements.
4. Configure an appropriate training method and its parameters.
5. Train the target model using a supported training backend.
6. Evaluate the resulting model using real evaluation runs and recorded metrics.
7. Export the artifacts and, when supported, prepare the model for use with Ollama.

Every stage should report its actual status, outputs, errors, and limitations.

🧰 Technology

TrainKitty is intended to integrate with the following ecosystem:

* Ollama: Local AI inference and agent orchestration.
* Python: Backend and machine learning workflow management.
* PyTorch: Model training and tensor computation.
* Hugging Face Transformers: Model loading and training integration.
* Hugging Face Datasets: Dataset loading and processing.
* PEFT: Parameter-efficient fine-tuning, including LoRA where supported.
* TRL: Training workflows supported by compatible models and configurations.
* FastAPI: Backend API.
* React, TypeScript, and Vite: Web interface.
* SQLite and SQLAlchemy: Local persistence and job metadata.

Actual support depends on implementation, installed dependencies, hardware, and model compatibility.

🔒 Local-First by Design

TrainKitty prioritizes privacy and local control.

* No mandatory cloud AI inference.
* No required OpenAI, Anthropic, or Google AI API keys.
* Training datasets and generated artifacts remain local by default.
* No telemetry or automatic uploads by default.
* Network access may be required for explicitly requested model, dataset, or dependency downloads.

Local-first does not mean every model can run on every computer. Hardware and compatibility checks are essential.

🎯 Project Principles

Real operations, not simulated success

TrainKitty must never fabricate training progress, evaluation scores, model availability, or successful exports.

Transparent limitations

If a model, training method, conversion, or hardware configuration is unsupported, the application must explain why.

Reproducibility

Training configurations, dataset provenance, logs, metrics, and artifact metadata should be preserved wherever practical.

User control

Users should be able to inspect plans, review configurations, monitor jobs, and cancel supported operations.

Open source

TrainKitty is intended to be developed transparently, with documented workflows, automated tests, and clear contribution guidelines.

🚧 Development Status

TrainKitty is under active development.

Features should be considered available only when they have been implemented and verified. Planned functionality is not a claim of existing functionality.

🚀 Getting Started

Setup instructions will be documented as the required components are implemented and verified.

The intended environment includes:

* A supported Python installation.
* Ollama with a locally available model.
* The project’s required Python dependencies.
* Node.js and npm for the web interface.
* Sufficient RAM, storage, and compatible compute resources for the selected training workload.

Training requirements vary significantly by model size, sequence length, quantization, and training method.

🗺️ Roadmap

* [ ]	Repository structure and development environment
* [ ]	Local Ollama connection and model discovery
* [ ]	Natural-language model specification
* [ ]	Synthetic dataset generation with validation
* [ ]	Hugging Face dataset import and preparation
* [ ]	Hardware and dependency preflight checks
* [ ]	Real training job orchestration
* [ ]	Live logs, metrics, cancellation, and job persistence
* [ ]	Model evaluation and base-model comparison
* [ ]	Artifact export and compatibility validation
* [ ]	Supported Ollama integration
* [ ]	Automated tests and continuous integration
* [ ]	Installation, troubleshooting, and contributor documentation

🤝 Contributing

Contributions are welcome as the project develops.

Please include relevant tests, document configuration changes, and clearly distinguish implemented behavior from planned functionality. Machine learning workflows should be tested against real supported configurations, with lightweight fixtures used only for isolated tests.

📜 License

TrainKitty is intended to be released under the Apache License 2.0.

The repository should include the complete license text before the project is represented as officially licensed under Apache-2.0.

⸻

TrainKitty
Your local AI agent for building models that are actually trained.
