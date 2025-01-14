from abc import ABC, abstractmethod

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from llm_engineering import domain
from llm_engineering.application import utils
from llm_engineering.domain.cleaned_documents import CleanedDocument
from llm_engineering.domain.dataset import DatasetType, TrainTestSplit
from llm_engineering.domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from llm_engineering.domain.types import DataCategory
from llm_engineering.settings import settings

from . import constants
from . import utils as generation_utils
from .output_parsers import ListPydanticOutputParser


class DatasetGenerator(ABC):
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
    dataset_type: DatasetType | None = None

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"

        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        input_variables = {
            "dataset_format": dataset_format,
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )

    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        documents = generation_utils.extract_substrings(documents)

        grouped_prompts = {}
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts

        return grouped_prompts

    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        data_category = document.get_category()

        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )
        input_variables = {
            "extract": document.content,
        }
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt)
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)

        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:
        assert cls.dataset_type is not None, "Dataset type must be set before calling generate()"

        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                SystemMessage(content=cls.get_system_prompt().content),
                HumanMessage(content=prompt.content),
            ]

            return messages

        if mock:
            llm = FakeListLLM(responses=[constants.get_mocked_response(cls.dataset_type)])
        else:
            assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=2000 if cls.dataset_type == DatasetType.PREFERENCE else 1200,
                temperature=0.7,
            )
        parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())

        chain = llm | parser

        datasets = {}
        for category, category_prompts in prompts.items():
            langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
            batches = utils.misc.batch(langchain_category_prompts, size=24)

            flattened_instruct_dataset_samples = []
            for batch in batches:
                try:
                    batched_dataset_samples = chain.batch(batch, stop=None)

                    for instruct_dataset_sample_batch in batched_dataset_samples:
                        flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
                except OutputParserException:
                    logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

            dataset = domain.dataset.build_dataset(
                dataset_type=cls.dataset_type, category=category, samples=flattened_instruct_dataset_samples
            )
            datasets[category] = dataset
            logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

        processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

        return processed_datasets

    @classmethod
    def _get_dataset_sample_type(
        cls,
    ) -> type[domain.dataset.InstructDatasetSample] | type[domain.dataset.PreferenceDatasetSample]:
        return (
            domain.dataset.InstructDatasetSample
            if cls.dataset_type == DatasetType.INSTRUCTION
            else domain.dataset.PreferenceDatasetSample
        )

    @classmethod
    @abstractmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        pass


class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must be self-contained and cover a wide range of cosmology topics  (e.g., dark matter, thermal history, Big Bang Nucleosynthesis, recombination, Electron-positron annihilation, Neutrino decoupling, QCD phase transition, Dark matter freeze-out, \
    Electroweak phase transition, Baryogenesis and Leptogenesis , CMB, linear and non-linear cosmic structure formation, inflation, \
        cosmological fields, gravitational effects, or particle physics) present in the extract. \
Instructions must never explicitly mention a context or a course. \
Instructions must be clear and concise, avoiding any ambiguity. \
Instructions include asking about definitions, explanations, or equations. \
Answers must imitate the writing style of the context. \
Answer retains equations, symbols, and scientific notation exactly as presented in the extract.

Example instruction: Explain Hubble Expansion. \
Example answer: The first key observation in modern cosmology was the discovery of the expanding universe by Edwin Hubble. \
    By 1929, he had obtained Cepheid distances for 24 galaxies with red-shifts and claimed that these displayed a linear relationship: v = Hd, where H is the Hubble constant. \
    He cited theoretical predictions from relativistic cosmology that redshift or velocity should increase linearly with distance in an expanding Universe as a possible explanation. \
    The Hubble constant is hence a measure of the rate of expansion of the universe. \
    At the time, Hubble estimated\nH \u2243500 km s\u22121 Mpc\u22121, because his calibration of Cepheid luminosities was in error. \
    The best modern value of the Hubble constant is close to 70 km s\u22121 Mpc\u22121.

Example instruction: What's dark matter's growing mode? \
Example answer: At early times, the dominant energy of radiation drives the universe to expand so fast that the collisionless matter, whose gravity is sub dominant, has no time to respond, and \u03b4 grows at most only logarithmically. \
    The growing solution is \u03b4c t \u03b4c ti 1 Aln t ti , 75 12 thus only a solution with an initial velocity , \u03b4 ti 0, can actually grow so that A 0 , and only logarithmically at that. \
    At late times, the radiation eventually becomes negligible. \
    Since dark matter dominates gravity and the expansion rate, and we are in an Einstein-de Sitter (cosmological constant) and matter-dominated universe, dark matter yields the growing solution \u03b4\u221da(t). 

Example instruction: What's The Robertson-Walker Metric? \
Example answer: The Robertson-Walker metric or the Friedmann Robertson Walker metric is the metric of a homogeneous, isotropic, and expanding universe. \
    It was shown in 1935 by Robertson Walker that the assumption of homogeneity and isotropy requires the metric to take the form, in spherical polar coordinates: \ 
    ds^2 = -c^2dt^2 + a^2(t) [dr^2 + S_k^2(r)(dθ^2 + sin^2θ dφ^2)] = -c^2dt^2 + a^2(t) [dr^2 + S_k^2(r)dΩ^2] \
    where a(t) is the scale factor, and k is the curvature constant of the universe that can be +1, 0, or -1. 
    
Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
For mathematical expressions, use plain text (e.g., KE = 0.5 * m * v^2).
Everything must be valid json output for langchain_core.output_parsers.
Provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{extract}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
    ) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


class PreferenceDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.PREFERENCE

    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking a specific astrophysics question that relates to concepts, phenomena, or equations mentioned in the context.
2. A generated answer attempting to provide a detailed explanation for the instruction, named as 'rejected'. The rejected answer should demonstrate a plausible response but may include minor inaccuracies or lack of specificity.
3. An extracted answer that is a verbatim copy of a relevant portion directly from the context, named as 'chosen'. Ensure the chosen answer retains equations, symbols, and scientific notation exactly as presented in the extract.

Instructions must be:
- Self-contained and general, avoiding any direct mention of the context, system, course, or extract.
- Focused on a wide range of astrophysical topics (e.g., dark matter, thermal history, Big Bang Nucleosynthesis, recombination, Electron-positron annihilation, Neutrino decoupling, QCD phase transition, Dark matter freeze-out, \
    Electroweak phase transition, Baryogenesis and Leptogenesis , CMB, linear and non-linear cosmic structure formation, inflation, \
        cosmological fields, gravitational effects, or particle physics) present in the extract. 
        
Important:
- Ensure the extracted answer, the chosen one, is a word-for-word copy from the context, including equations, punctuation, and formatting.
- If the relevant text is not continuous, include two separate sentences from the context without skipping text or adding ellipses (...).
- Do not generate instructions or answers that require external information beyond the extract.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...",
        "rejected": "...",
        "chosen": "..."
    },
    ...
]

Extract:
{extract}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, domain.dataset.PreferenceDataset], test_size: float
    ) -> TrainTestSplit:
        datasets = generation_utils.filter_short_answers(datasets)
        datasets = generation_utils.filter_answer_format(datasets)

        remaining_samples = sum([dataset.num_samples for dataset in datasets.values()])
        logger.info(
            f"Filtered out short answers and answers with incorrect format. Remaining samples: {remaining_samples}"
        )

        train_test_split = generation_utils.create_preference_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


def get_dataset_generator(dataset_type: DatasetType) -> type[DatasetGenerator]:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructionDatasetGenerator
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDatasetGenerator
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
