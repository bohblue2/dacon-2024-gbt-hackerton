from transformers import AutoConfig, AutoTokenizer, BigBirdForSequenceClassification, AutoModelForSequenceClassification
from torch import nn 

def get_model_and_tokenizer(
    model_name: str, 
    num_labels: int, 
    attention_type: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "monologg/kobigbird-bert-base":
        model = BigBirdForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            attention_type=attention_type
        )
        model.classifier.dense = nn.Linear(model.config.hidden_size, model.config.hidden_size)
        model.classifier.out_proj = nn.Linear(model.config.hidden_size, num_labels)
    elif model_name == "klue/roberta-large":
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 54
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
    return model, tokenizer