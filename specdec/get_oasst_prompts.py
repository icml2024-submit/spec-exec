import json
import random

import datasets
from langdetect import detect
from tqdm.auto import tqdm

dataset_name = "OpenAssistant/oasst1"
dataset_splits = ["validation"]
accepted_languages = {"en"}
random_seed = 1337
FILE_PATH = "data/oasst_prompts.json"
system_prompt = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()

# HF version
# system_prompt = """
# Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable.
# The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats
# when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful.
# """.strip()


formatting = dict(
    prompter_first="{}[INST] <<SYS>>\n" + system_prompt.strip() + "\n<</SYS>\n\n\n{} [\\INST]\n",
    prompter="{}[INST] {} [\\INST]\n",
    assistant="{}{}\n",
)

assistant_prefix = "[\\INST]\n"  # the last non-llm text before llm response; usually present in formatting
assistant_stop_sequence = "[INST]"  # stop generation if assistant generates this sequence OR </s>

data = datasets.load_dataset(dataset_name)
data = datasets.concatenate_datasets([data[split] for split in dataset_splits])

data_by_message_id = {row["message_id"]: row for row in tqdm(data, desc="data by message id")}
assert len(data_by_message_id) == len(data)

data_by_parent_id = {}
for row in tqdm(data, desc="data by parent id"):
    data_by_parent_id.setdefault(row["parent_id"], [])
    data_by_parent_id[row["parent_id"]].append(row)

found = attempted = 0
with tqdm() as progress:

    def _bfs(node_id=None, conversation_prefix="", raw_conversation_prefix=""):
        global found, attempted
        if node_id not in data_by_parent_id:
            attempted += 1
            if accepted_languages is not None:
                try:
                    if detect(raw_conversation_prefix) not in accepted_languages:
                        return  # wrong language; skip
                except:
                    return  # failed to detect language; skip unless accepted_languages is None
            found += 1
            progress.update()
            progress.desc = f"accepted {found} out of {attempted} conversations"
            yield conversation_prefix
        else:
            for child in data_by_parent_id[node_id]:
                format_string = formatting[child["role"] + "_first" * (child["parent_id"] is None)]
                yield from _bfs(
                    node_id=child["message_id"],
                    conversation_prefix=format_string.format(conversation_prefix, child["text"].strip()),
                    raw_conversation_prefix="\n".join((raw_conversation_prefix, child["text"].strip())),
                )

    conversations = list(_bfs())

random.seed(random_seed)

# cleaning conversation from non-english lines
stopwords = ["wichtigsten", "canciones"]  # found manually. no guarantee of completeness.

clean_conversations = []
for c in conversations:
    flag_ok = True
    for stopword in stopwords:
        if stopword in c:
            flag_ok = False
            break
    if flag_ok:
        clean_conversations.append(c)

oasst_prompts = []
for conversation in conversations:
    conversation_parts = conversation.split(assistant_prefix)
    prefix_length = random.randint(1, len(conversation_parts) - 1)  # random.randint includes the upper limit
    oasst_prompts.append(assistant_prefix.join(conversation_parts[:prefix_length]) + assistant_prefix)

oasst_prompts = list(set(oasst_prompts))  # there were ~50% of duplicates

random.seed(random_seed)
random.shuffle(oasst_prompts)
oasst_prompts = [(i, v) for i, v in enumerate(oasst_prompts)]  # to tuples with indices for better json readability

# Saving the permuted lines to a json file
with open(FILE_PATH, "w") as f:
    json.dump(oasst_prompts, f, indent=4)

print(f"saved {len(oasst_prompts)} unique prompts to {FILE_PATH}")
