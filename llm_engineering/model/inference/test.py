from loguru import logger

from llm_engineering.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.settings import settings

if __name__ == "__main__":
    text = "Explain a dark matter halo's circular velocity."
    logger.info(f"Running inference for text: '{text}'")
    llm = LLMInferenceSagemakerEndpoint(
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE, inference_component_name=None
    )
    answer = InferenceExecutor(llm, text).execute()

    logger.info(f"Answer: '{answer}'")
