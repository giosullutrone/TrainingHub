import random
from typing import Dict, Union, Callable
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from cookbooks import DATASET_COOKBOOK
from .DatasetRecipe import DatasetRecipe

import random
from sentence_transformers import SentenceTransformer
import numpy as np


class TopicSimilarityCalculator:
    """
    A class to calculate similarity scores between topics.

    Attributes:
        topics (list): A list of article topics (strings).
        model (SentenceTransformer): Pre-trained SentenceTransformer model.
        similarity_matrix (numpy.ndarray): Matrix containing pairwise cosine similarity scores between topics.
    """

    def __init__(self, topics):
        """
        Initializes the TopicSimilarityCalculator with the given list of topics.

        Args:
            topics (list): A list of article topics.
        """
        self.topics = topics
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self._calculate_similarity_matrix()

    def _calculate_similarity_matrix(self):
        """
        Calculates the pairwise cosine similarity matrix between topics.
        """
        # Embed all topics
        topic_embeddings = self.model.encode(self.topics)

        # Calculate pairwise cosine similarity between all topic embeddings
        self.similarity_matrix = np.inner(topic_embeddings, topic_embeddings)

        # Normalize the similarity scores to be between 0 and 1
        self.similarity_matrix = (self.similarity_matrix + 1) / 2

    def _order_similarity(self, topic_index):
        """
        Orders similarity scores for the given topic.

        Args:
            topic_index (int): Index of the topic in the topics list.

        Returns:
            dict: A dictionary containing combination scores for the given topic.
        """
        num_topics = len(self.topics)
        similarity_scores = {}

        # Sort the similarity scores for the given topic
        sorted_indices = np.argsort(self.similarity_matrix[topic_index])[::-1]

        # Assign a value from 0 to 5 for each combination based on their order
        for j, index in enumerate(sorted_indices):
            similarity_scores[self.topics[index]] = (num_topics - j - 1) / (num_topics - 1) * 5

        return similarity_scores

    def get_random_topic_similarity(self, input_topic):
        """
        Returns a random topic from the list different from the input topic,
        along with the combination score between the input topic and the random topic.

        Args:
            input_topic (str): The input topic for which similarity is calculated.

        Returns:
            tuple: A tuple containing the random topic and the combination score.
        """
        # Find the index of the input topic
        input_topic_index = self.topics.index(input_topic)

        # Select a random topic different from the input topic
        random_topic_index = random.choice([i for i in range(len(self.topics)) if i != input_topic_index])
        random_topic = self.topics[random_topic_index]

        # Get the similarity score for the random topic
        combination_score = self._order_similarity(input_topic_index)[random_topic]

        return random_topic, combination_score
    

class GenerationDatasetRecipe(DatasetRecipe):
    def postprocess_generation_function(self, sample: Dict) -> Dict:
        return sample


@DATASET_COOKBOOK.register()
class NewsGenerationDatasetRecipe(GenerationDatasetRecipe):
    def preprocess_function(self, sample: Dict, examples: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset, None]) -> Dict:
        if examples:
            prompt = "Generate NEW news documents following the examples.\n\n"
            prompt += "\n\n".join(["News:\n" + ex["text"][:512] + "\nTopic:" + ex["label_text"] for ex in examples]) + "\n\n"
        else: prompt = "Generate new news documents.\n\n"

        prompt += f"News:"
        return {"prompts": prompt, "labels": ""}


@DATASET_COOKBOOK.register()
class NewsCoTGenerationDatasetRecipe(GenerationDatasetRecipe):
    def __init__(self, preprocess_dataset = None, preprocess_function = None, dataset_load = None, dataset_response_template = None) -> None:
        super().__init__(preprocess_dataset, preprocess_function, dataset_load, dataset_response_template)
        self.topic_sim = TopicSimilarityCalculator([
            'alt.atheism',
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'comp.windows.x',
            'misc.forsale',
            'rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'soc.religion.christian',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.politics.misc',
            'talk.religion.misc'
        ])

    def preprocess_function(self, sample, examples) -> Dict:
        topic, score = self.topic_sim.get_random_topic_similarity(sample['label_text'])
        prompt = f"You are an AI specialized in evaluating the results from a topic modeling algorithm. \
Generate a single step-by-step explanation on why the combination of news and topic have achieved a given score from 0 to 5. \
Where 5 means that the algorithm has given the correct topic and 0 that the topic is the worst among the possible one. \
\n\nNews:\n{sample['text']}\
\n\nTopic:{topic}\
\n\nScore:{score}\
\n\nExplanation:"
        return {"prompts": prompt, "labels": "", "news": sample['text'], "topic": topic, "score": score}
