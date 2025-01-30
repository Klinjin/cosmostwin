<div align="center">
  <h1>🪐 LLM Cosmos twin</h1>
  <p></p> An AI tutor for graduate-level Cosmology class.</p>
</div>

- 📝 Data collection & generation
- 🔄 LLM training pipeline
- 📊 Simple RAG system
- 🚀 Production-ready AWS deployment
- 🔍 Comprehensive monitoring
- 🧪 Testing and evaluation framework

You can download and use the final trained model on [Hugging Face](https://huggingface.co/klin0604/CosmosTwinLlama-3.1-8B).

## 🔗 Dependencies

### Local dependencies


| Tool | Version | Purpose | Installation Link |
|------|---------|---------|------------------|
| pyenv | ≥2.3.36 | Multiple Python versions (optional) | [Install Guide](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) |
| Python | 3.11 | Runtime environment | [Download](https://www.python.org/downloads/) |
| Poetry | >= 1.8.3 and < 2.0 | Package management | [Install Guide](https://python-poetry.org/docs/#installation) |
| Docker | ≥27.1.1 | Containerization | [Install Guide](https://docs.docker.com/engine/install/) |
| AWS CLI | ≥2.15.42 | Cloud management | [Install Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Git | ≥2.44.0 | Version control | [Download](https://git-scm.com/downloads) |

### Cloud services

| Service | Purpose |
|---------|---------|
| [HuggingFace](https://huggingface.com/) | Model registry |
| [Comet ML](https://www.comet.com/site/products/opik/?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik) | Experiment tracker |
| [Opik](https://www.comet.com/site/products/opik/?utm_source=llm_handbook&utm_medium=github&utm_campaign=opik) | Prompt monitoring |
| [ZenML](https://www.zenml.io/) | Orchestrator and artifacts layer |
| [AWS](https://aws.amazon.com/) | Compute and storage |
| [MongoDB](https://www.mongodb.com/) | NoSQL database |
| [Qdrant](https://qdrant.tech/) | Vector database |
| [GitHub Actions](https://github.com/features/actions) | CI/CD pipeline |

## 🏃 Run project

Based on the setup and usage steps described above, assuming the local and cloud infrastructure works and the `.env` is filled as expected, follow the next steps to run the LLM system end-to-end:

### Data

1. Collect data: `poetry poe run-digital-data-etl`

2. Compute features: `poetry poe run-feature-engineering-pipeline`

3. Compute instruct dataset: `poetry poe run-generate-instruct-datasets-pipeline`

4. Compute preference alignment dataset: `poetry poe run-generate-preference-datasets-pipeline`

### Training

5. SFT fine-tuning Llamma 3.1: `poetry poe run-training-pipeline`

6. For DPO, go to `configs/training.yaml`, change `finetuning_type` to `dpo`, and run `poetry poe run-training-pipeline` again

7. Evaluate fine-tuned models: `poetry poe run-evaluation-pipeline`

### Inference

8. Call only the RAG retrieval module: `poetry poe call-rag-retrieval-module`

9. Deploy the LLM Twin microservice to SageMaker: `poetry poe deploy-inference-endpoint`

10. Test the LLM Twin microservice: `poetry poe test-sagemaker-endpoint`

11. Start end-to-end RAG server: `poetry poe run-inference-ml-service`

12. Test RAG server: `poetry poe call-inference-ml-service`

## 📄 License

This course is an open-source project released under the MIT license. Thus, as long you distribute our LICENSE and acknowledge our work, you can safely clone or fork this project and use it as a source of inspiration for whatever you want (e.g., university projects, college degree projects, personal projects, etc.).
