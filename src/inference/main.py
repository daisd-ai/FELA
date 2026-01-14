import json
import logging
import time

import faiss
from FlagEmbedding import FlagLLMReranker
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# DATASET_NAME = "Tweeki_gold"
DATASET_NAME = "RSS_500"
# DATASET_NAME = "reuters-128_wikidata"
DATA_DIR = "data/datasets"
WIKIDATA_FILE = "data/wikidata_id_to_profile.json"
FAISS_INDEX = "data/faiss_ubinary_wikidata_v2.index"
DATA_FILE = f"{DATA_DIR}/{DATASET_NAME}.jsonl"
TOP_K = 25
GPU_MEM_UTIL = 0.2

logging.info("Loading NER model")
ner_sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
)
ner_model = "daisd-ai/UniNER-W4A16"
ner_llm = LLM(
    model=ner_model,
    gpu_memory_utilization=GPU_MEM_UTIL,
)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model)

logging.info("Loading EL model")
el_sampling_params = SamplingParams(
    temperature=0,
    max_tokens=128,
)
el_model = "daisd-ai/anydef-v2-linear-W4A16"
el_llm = LLM(
    model=el_model,
    gpu_memory_utilization=GPU_MEM_UTIL,
)
el_tokenizer = AutoTokenizer.from_pretrained(el_model)

logging.info("Loading embedding model")
embedding_model = "mixedbread-ai/mxbai-embed-large-v1"
embedding_model = SentenceTransformer(
    embedding_model,
    prompts={
        "retrieval": "Represent this sentence for searching relevant passages: ",
    },
    default_prompt_name="retrieval",
    device="cuda",
)
logging.info("Loading reranker")
reranker = FlagLLMReranker(
    "BAAI/bge-reranker-v2-gemma",
    query_max_length=256,
    passage_max_length=256,
    use_fp16=True,
    devices=["cuda:0"],
)

logging.info("Loading index")
faiss_index = faiss.read_index_binary(FAISS_INDEX)

logging.info("Transfering to GPU")
gpu_index = faiss.index_cpu_to_all_gpus(faiss_index.index)
faiss_index.own_fields = False
faiss_index.index = gpu_index
faiss_index.own_fields = True

logging.info("Loading data")
with open(DATA_FILE, "r") as f:
    raw_data = f.readlines()

data = []
for line in raw_data:
    data.append(json.loads(line))

logging.info("Loading wikidata")
with open(WIKIDATA_FILE, "r") as f:
    wikidata = json.load(f)

ner_types = [
    "person",
    "organization",
    "location",
    "sports team",
    "geo-political entity",
    "facility",
    "event",
    "product",
    "creative work",
    "law",
    "nationality",
    "company",
    "finacial institution",
]

t = time.time()
prompts = []
for line in data:
    text = line["sentence"]
    for entity_type in ner_types:
        messages = [
            {
                "role": "user",
                "content": f"Text: {text}",
            },
            {"role": "assistant", "content": "I've read this text."},
            {
                "role": "user",
                "content": f"What describes {entity_type} in the text?",
            },
        ]
        prompt = ner_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

outputs = ner_llm.generate(prompts, ner_sampling_params)
outputs = [output.outputs[0].text for output in outputs]

results = []
for entities in outputs:
    try:
        entities = list(set(json.loads(entities)))
    except Exception:
        entities = []

    results.append(entities)

formatted_results = []
for i in range(0, len(results), len(ner_types)):
    formatted_results.append(
        {ner_types[j]: results[i + j] for j in range(len(ner_types))}
    )

ner_results = []
excluded_keys = ["organization", "location"]
for i, line in enumerate(data):
    organizations = list(
        set(formatted_results[i]["organization"])
        - set(formatted_results[i]["sports team"])
    )
    organizations = list(set(organizations) - set(formatted_results[i]["company"]))
    organizations = list(
        set(organizations) - set(formatted_results[i]["finacial institution"])
    )
    locations = list(
        set(formatted_results[i]["location"]) - set(formatted_results[i]["facility"])
    )
    locations = list(set(locations) - set(formatted_results[i]["geo-political entity"]))

    entities = []
    for key, value in formatted_results[i].items():
        if key in excluded_keys:
            continue

        if not value:
            continue

        entities.extend([f"{entity} ({key})" for entity in value])

    entities.extend([f"{entity} (organization)" for entity in organizations])
    entities.extend([f"{entity} (location)" for entity in locations])

    entities.extend(organizations)
    entities.extend(locations)

    ner_results.append(
        {
            "sentence": line["sentence"],
            "expected_ids": [i.split("|")[1] for i in line["link"]],
            "entities": entities,
        }
    )

logging.info(f"NER took {time.time() - t:.2f}s")

prompts = []
for context in ner_results:
    for entity in context["entities"]:
        messages = [
            {
                "role": "user",
                "content": context["sentence"],
            },
            {"role": "assistant", "content": "I read this text"},
            {"role": "user", "content": f"What is a profile for entity {entity}?"},
        ]
        prompt = el_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

outputs = el_llm.generate(prompts, el_sampling_params)

profiles = [output.outputs[0].text for output in outputs]

embeddings = embedding_model.encode(profiles)
embeddings = quantize_embeddings(embeddings, precision="ubinary")

scores, identifiers = faiss_index.search(embeddings, TOP_K)

final_results = {}
counter = 0
for context in ner_results:
    final_results[context["sentence"]] = {
        "expected_ids": context["expected_ids"],
        "entities": context["entities"],
        "profile_to_ids": [],
    }

    for entity in context["entities"]:
        tmp_results = []

        for i in identifiers[counter]:
            tmp_results.append((f"Q{i}", wikidata[f"Q{i}"]))

        final_results[context["sentence"]]["profile_to_ids"].append(
            {profiles[counter]: tmp_results}
        )

        counter += 1

logging.info("Evaluating")
counter_correct = 0
counter_retrieval = 0
counter_total = 0

# this portion of the code could be optimized - batch scoring instead of one by one
for key, value in final_results.items():
    expected_ids = value["expected_ids"]
    profile_to_ids = value["profile_to_ids"]

    counter_total += len(expected_ids)

    for d in profile_to_ids:
        for generated_profile, found_wikidata_profiles in d.items():
            profiles = [i[1] for i in found_wikidata_profiles]

            scores = reranker.compute_score(
                [[generated_profile, i] for i in profiles], normalize=True
            )

            # extract highest score with position
            highest_score = max(scores)
            highest_score_position = scores.index(highest_score)
            gold_entity = found_wikidata_profiles[highest_score_position][0]

            if gold_entity in expected_ids:
                counter_correct += 1
                ########
                expected_ids.remove(gold_entity)
                counter_retrieval += 1
                continue
                ########

            wikidata_ids = set([i[0] for i in found_wikidata_profiles])
            for wikidata_id in wikidata_ids:
                if wikidata_id in expected_ids:
                    counter_retrieval += 1

logging.info(f"EL took {time.time() - t:.2f}s")

logging.info(
    f"E2E entity linking on {DATASET_NAME} accuracy: {counter_correct}/{counter_total} ({(counter_correct / counter_total) * 100:.2f}%)"
)
logging.info(
    f"E2E entity linking on {DATASET_NAME} retrieval: {counter_retrieval}/{counter_total} ({(counter_retrieval / counter_total) * 100:.2f}%)"
)
