from transformers import BertTokenizer, BertModel # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import gdown # type: ignore

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def predict_sentiment(text, model, tokenizer, device='cpu', max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)  # Assuming logits is the output of your model
    label = preds.item()

    if label == 2:
        return "Positive Tweet"
    elif label == 1:
        return "Neutral Tweet"
    elif label == 0:
        return "Negative Tweet"
    else:
        return "Unknown Label"

def predict(text):
    return predict_sentiment(text, model, tokenizer)

def download_model_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def load_model_and_tokenizer(model_path):
    model = BERTClassifier('bert-base-uncased', num_classes=3)  # Adjust num_classes according to your task
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

file_id = '1w4Gu3IETSsFYFS_L8aGFmV7Jupl2STB0'  # Replace with your file ID
output_path = '../bert_classifier_label.pth'
download_model_from_drive(file_id, output_path)

model, tokenizer = load_model_and_tokenizer(output_path)

print(predict("Modi is a very good PM."))  # Change the text as needed
