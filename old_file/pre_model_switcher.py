from selectors import EpollSelector
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_metric
from evaluate import evaluator
from datasets import load_dataset

if 'training' not in st.session_state:
    st.session_state['training'] = False
if 'run' not in st.session_state:
    st.session_state['run'] = False

st.title("Model Switcher : A short demo")
st.subheader("Choose method of choosing file")
task_type = st.selectbox("Select Task Type", ["Sentiment Analysis"])
model_upload_method = st.selectbox("Select Model Weight Upload Method", ["Huggingface", "Upload File"])
dataset_upload_method = st.selectbox("Select your data upload method", ["Huggingface", "Upload File"])
# Model Initialization - Hugging Face Library - User Input file
# Model Weight Initialization
model = None
dataset = None
model_name = None
small_eval_dataset = None

def model_from_hf(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name)
def enter_dataset():
    dataset_name = st.text_input("🤗Huggingface Dataset", placeholder="Dataset")
    return dataset_name
col1, col2 = st.columns(2)

if model_upload_method == "Huggingface" :
    model_name = col1.text_input("🤗Huggingface Model", value="Model Name", placeholder="Model")
else:
    # Upload Local File
    # model_code = col1.file_uploader("Upload the Model Architecture",
    # type ='py',
    # accept_multiple_files= False)
    # if model_code:
    #     with open("Model.py",'wb') as f:
    #         f.write(model_code.getbuffer())
    model_name = col1.text_input("🤗Huggingface Model", value="Model Name", placeholder="Model")
    model_weight = col2.file_uploader("Upload Model Weight")

c = col2 if model_upload_method == "Huggingface" else col1
if dataset_upload_method == "Huggingface":
    dataset_name = c.text_input("🤗Huggingface Dataset", value="Dataset Name", placeholder="Data")
elif dataset_upload_method == "Upload Dataset File":
    dataset_name = "Upload File"


run = st.button("Run")
if run:
    st.session_state["run"] = True
if st.session_state["run"]:
    if model_name == "Model Name":
        model_name = "Model Name"
    else:
        try:
            model = model_from_hf(model_name)
            print("Success!")
        except OSError:
            st.error("".join(["Model name ", model_name, " is not found in HuggingFace"]))
    if dataset_name == "Dataset Name":
        dataset_name = "Dataset Name"
    else:
        try:
            dataset = load_dataset(dataset_name)
            print("Success!")
            example_data_n = 10
            with st.expander(f"See {example_data_n} example data"):
                st.write(dataset['train'][:example_data_n])
            with st.expander(f"See the shape of data"):
                st.write(dataset.shape)

        except OSError:
            st.error("".join(["Dataset name ", dataset_name, " is not found in HuggingFace"]))

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = dataset.map(lambda x: tokenize_function(tokenizer,x), batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    ) 
    return trainer

if model is not None and dataset is not None:
    run_eval = st.button("Start Evaluating")

    if run_eval:
        st.session_state['training'] = True
    if st.session_state['training']:
        st.write("Train!")
        trainer = train()
        with st.expander(f"See Training Result"):
            st.write(trainer.train())
        with st.expander(f"See Eval Result"):
            res = trainer.evaluate(small_eval_dataset)
            st.write(res)

    


# Load Training Data

# Load Test Data


# Benchmark
# Show accuracy, loss, Compare with other