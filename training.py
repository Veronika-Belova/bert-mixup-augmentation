import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast,BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
import evaluate

# Загрузка данных и токенизация
dataset = load_dataset("rotten_tomatoes")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# MixUp функции
def mixup_data(x, y, alpha=1.0):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

   # Кастомный тренер с MixUp
class MixupTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        input_ids = inputs['input_ids'].to(self.args.device)
        attention_mask = inputs['attention_mask'].to(self.args.device)
        labels = inputs['labels'].to(self.args.device)

        model_embeddings = model.bert.embeddings(input_ids).detach()
        mixed_embeddings, labels_a, labels_b, lam = mixup_data(model_embeddings, labels)

        outputs = model(inputs_embeds=mixed_embeddings, attention_mask=attention_mask)
        loss = mixup_criterion(F.cross_entropy, outputs.logits, labels_a, labels_b, lam)

        loss.backward()
        return loss.detach()

# Создание модели и запуск тренировки
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    weight_decay=0.01,
   )

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = metric.compute(predictions=predictions, references=labels)["accuracy"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

trainer = MixupTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()