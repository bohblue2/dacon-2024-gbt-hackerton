from transformers import AutoTokenizer, BigBirdForSequenceClassification
from torch import nn 

def get_model_and_tokenizer(
    model_name: str, 
    num_labels: int, 
    attention_type: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BigBirdForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        attention_type=attention_type
    )
    if model_name == "monologg/kobigbird-bert-base":
        model.classifier.dense = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        model.classifier.out_proj = nn.Linear(model.config.hidden_size, num_labels)
    return model, tokenizer