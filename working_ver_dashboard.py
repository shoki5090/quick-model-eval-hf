from selectors import EpollSelector
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_metric, DatasetDict
from evaluate import evaluator
from datasets import load_dataset
import zipfile

if 'training' not in st.session_state:
    st.session_state['training'] = False
if 'run' not in st.session_state:
    st.session_state['run'] = False
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = False
if 'dataset_name' not in st.session_state:
    st.session_state['dataset_name'] = False
if 'run_eval' not in st.session_state:
    st.session_state['run_eval'] = False
if 'op_type' not in st.session_state:
    st.session_state['op_type'] = False
if 'dataset_type' not in st.session_state:
    st.session_state['dataset_type'] = False
pre_train_dir = "./pt_save_pretrained/"

model = None
dataset = None

def model_from_hf(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name)

def enter_dataset():
    dataset_name = st.text_input("🤗Huggingface Dataset", placeholder="Dataset")
    return dataset_name

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenized_datasets(model_name, dataset):
    if st.session_state['op_type'] == "huggingFace":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif st.session_state['op_type'] == "upload": 
        tokenizer = AutoTokenizer.from_pretrained(pre_train_dir)
    tokenized_datasets = dataset.map(lambda x: tokenize_function(tokenizer,x), batched=True)  
    return  tokenized_datasets

def train(model, train_dataset, eval_dataset):


    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    ) 
    return trainer

def smaller_dataset(model_name, dataset):
    tokenized_datasets_data = tokenized_datasets(model_name, dataset)

    small_train_dataset = tokenized_datasets_data["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets_data["test"].shuffle(seed=42).select(range(1000))

    return small_train_dataset, small_eval_dataset
if __name__ == '__main__':

    st.title("NLP model/weight benchmark")
    task_type = st.selectbox("Select Task Type", ["Sentiment Analysis"])
    op_type = st.selectbox("Operation type",["HuggingFace",'Self Upload'])
    
    pt_save_directory = pre_train_dir

    if op_type == 'HuggingFace':
        #list of inputs: 
        # hf model name
        hf_model_name = st.text_input("HF model name")
        st.session_state["model_name"] = hf_model_name
        st.session_state["op_type"] = "huggingFace"
    else:
        #hf model name
        # hf model weights

        pt_file = st.file_uploader("Please upload the PT Zip File", accept_multiple_files= False)

        if pt_file:
            with open("pt_save_pretrained.zip",'wb') as f:
                f.write(pt_file.getbuffer())
            with zipfile.ZipFile("pt_save_pretrained.zip","r") as zip_ref:
                zip_ref.extractall("./")
        st.session_state["op_type"] = "upload"

    dataset_upload_method = st.selectbox("Data Upload", ["HF", "Upload File"])
    dataset_name = None
    if dataset_upload_method == "HF":
        dataset_name = enter_dataset()
        if dataset_name:
            st.session_state["dataset_name"] = dataset_name
        st.session_state["dataset_type"] = "huggingface"
    else:
        uploaded_dataset = st.file_uploader("Please upload the Data")
        with open("dataset.csv",'wb') as f:
            f.write(uploaded_dataset.getbuffer())
        st.session_state["dataset_type"] = "upload"

    
    run = st.button("Run")
    if run:
        st.session_state["run"] = True
    if st.session_state["run"]:
        if st.session_state["op_type"] == "huggingFace":
            if st.session_state["model_name"]:
                model_name = st.session_state["model_name"]
                try:
                    model = model_from_hf(model_name)
                    print("Success!")
                except OSError:
                    st.error("".join(["Model name ", model_name, " is not found in HuggingFace"]))
            else:
                st.write("No model is loaded yet")
        elif st.session_state["op_type"] == "upload":
            model = AutoModelForSequenceClassification.from_pretrained(pre_train_dir)
        
        if st.session_state["dataset_type"] == "huggingface":
            if st.session_state["dataset_name"]:
                dataset_name = st.session_state["dataset_name"]
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
        elif st.session_state["dataset_type"] == "upload":
            try:
                dataset = load_dataset("csv", data_files="dataset.csv")


                
                dataset = dataset["train"]
                st.write(dataset)
                train_devtest = dataset.train_test_split(shuffle = True, seed = 200, test_size=0.3)
                posts_dev_test = train_devtest['test'].train_test_split(shuffle = True, seed = 200, test_size=0.50)
                posts_train_dev_test_dataset = DatasetDict({
                    'train': train_devtest['train'],
                    'test': posts_dev_test['test'],
                    'dev': posts_dev_test['train']})
                print("Success!")
                example_data_n = 10
                with st.expander(f"See {example_data_n} example data"):
                    st.write(posts_train_dev_test_dataset['train'][:example_data_n])
                with st.expander(f"See the shape of data"):
                    st.write(posts_train_dev_test_dataset.shape)
                dataset = posts_train_dev_test_dataset

            except OSError:
                st.error("".join(["Dataset name ", "./dataset.csv", " is not found in local drive"]))

        
        run_eval = None
        if model is not None and dataset is not None:
            run_eval = st.button("Start Evaluating")

        if run_eval:
            st.session_state['training'] = True
        if st.session_state['training']:
            train_dataset, eval_dataset = smaller_dataset(st.session_state["model_name"], dataset)
            with st.spinner('Running...'):
                trainer = train(model,train_dataset, eval_dataset)
            with st.expander("See Training Result"):
                st.write(trainer.train())
            with st.expander("See Eval Result"):
                res = trainer.evaluate(eval_dataset)
                st.write(res)



    #model initializatin

    ##Dataset selection 
    