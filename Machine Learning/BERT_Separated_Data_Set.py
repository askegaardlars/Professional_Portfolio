import pandas as pd

#import dill as pickle

import numpy as np

from tqdm.auto import tqdm

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl

from torchmetrics.functional import accuracy, f1, auroc

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns

##from pylab import rcParams

import matplotlib.pyplot as plt

from matplotlib import rc

from sklearn.preprocessing import MultiLabelBinarizer

import csv
#%matplotlib inline

#%config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

##sns.set(style='whitegrid', palette='muted', font_scale=1.2)
##
##HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
##
##sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
##
##rcParams['figure.figsize'] = 12, 8
##
##pl.seed_everything(RANDOM_SEED)


mlb = MultiLabelBinarizer()

df = pd.read_csv(r"~/Practicum2021/lematizedData.csv",encoding='ISO-8859-1')
test_df = pd.read_csv(r"~/Practicum2021/real_test_lem.csv",encoding='ISO-8859-1')
X = df["event_text"]
DP_train_val = mlb.fit_transform(df["Device Problem"].str.split("; "))
DP_test_actual = test_df["Device Problem"]
DP_test = mlb.transform(test_df["Device Problem"].str.split("; "))
X_test = test_df["event_text"]

X_train, X_val, DP_train, DP_val = train_test_split(X, DP_train_val, test_size=0.1)

#print(X_train[0])
LABEL_COLUMNS = df.columns.tolist()[1:]

BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

#token_counts = []
#for _, row in train_df.iterrows():
#	token_count = len(tokenizer.encode(row["event_text"],max_length=512,truncation=True))
#	token_counts.append(token_count)


MAX_TOKEN_COUNT = 512

class Dataset(Dataset):
	def __init__(self,quest, tags,tokenizer: BertTokenizer, max_token_len: int = 128):
		self.tokenizer = tokenizer
		self.text = quest
		self.labels = tags
		self.max_token_len = max_token_len
	def __len__(self):
		return len(self.text)
	def __getitem__(self, item_idx):
		text = self.text.iloc[item_idx]
		encoding = self.tokenizer.encode_plus(
			text,
			None,
			add_special_tokens=True,
			max_length=self.max_token_len,
			return_token_type_ids=False,
			padding="max_length",
			truncation=True,
			return_attention_mask=True,
			return_tensors='pt',
		)
		return dict(
			EventText = text,
			input_ids=encoding["input_ids"].flatten(),
			attention_mask=encoding["attention_mask"].flatten(),
			labels=torch.tensor(self.labels[item_idx], dtype = torch.float)
		)
    
#train_dataset = Dataset(
#	train_df,
#	tokenizer,
#	max_token_len=MAX_TOKEN_COUNT
#)

#sample_item = train_dataset[0]

#bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

#sample_batch = next(iter(DataLoader(train_dataset, batch_size=2, num_workers=2)))

#sample_batch["input_ids"].shape, sample_batch["attention_mask"].shape

class DataModule(pl.LightningDataModule):
	def __init__(self,x_tr,y_tr,x_val,y_val,x_test,y_test,tokenizer, batch_size=18,max_token_len=200):
		super().__init__()
		self.tr_text = x_tr
		self.tr_label = y_tr
		self.val_text = x_val
		self.val_label = y_val
		self.test_text = x_test
		self.test_label = y_test
		self.tokenizer = tokenizer
		self.batch_size = batch_size
		self.max_token_len = max_token_len
	def setup(self, stage=None):
		self.train_dataset = Dataset(self.tr_text,self.tr_label,tokenizer=self.tokenizer,max_token_len= self.max_token_len)
		self.val_dataset= Dataset(self.val_text,self.val_label,tokenizer=self.tokenizer,max_token_len = self.max_token_len)
		self.test_dataset = Dataset(self.test_text,self.test_label,tokenizer=self.tokenizer,max_token_len = self.max_token_len)
	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=4
			)
	def val_dataloader(self):
  		return DataLoader(
  			self.val_dataset,
  			batch_size=self.batch_size,
  			)
	def test_dataloader(self):
    		return DataLoader(
      			self.test_dataset,
      			batch_size=self.batch_size,
      			)

N_EPOCHS = 25

BATCH_SIZE = 1

data_module = DataModule(X_train, DP_train, X_val, DP_val, X_test, DP_test ,tokenizer,batch_size=BATCH_SIZE,max_token_len=MAX_TOKEN_COUNT)
data_module.setup()
#TrainModule = data_module.train_dataloader()
#ValModule = data_module.val_dataloader()

class Tagger(pl.LightningModule):
	def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
		super().__init__()
		self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
		self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
		self.n_training_steps = n_training_steps
		self.n_warmup_steps = n_warmup_steps
		self.criterion = nn.BCELoss()
	def forward(self, input_ids, attention_mask, labels=None):
    		output = self.bert(input_ids, attention_mask=attention_mask)
    		output = self.classifier(output.pooler_output)
    		output = torch.sigmoid(output)
    		loss = 0
    		if labels is not None:
    			loss = self.criterion(output, labels)
    		return loss, output
	def training_step(self, batch, batch_idx):
    		input_ids = batch["input_ids"]
    		attention_mask = batch["attention_mask"]
    		labels = batch["labels"]
    		loss, outputs = self(input_ids, attention_mask, labels)
    		self.log("train_loss", loss, prog_bar=True, logger=True)
    		return {"loss": loss, "predictions": outputs, "labels": labels}
	def validation_step(self, batch, batch_idx):
    		input_ids = batch["input_ids"]
    		attention_mask = batch["attention_mask"]
    		labels = batch["labels"]
    		loss, outputs = self(input_ids, attention_mask, labels)
    		self.log("val_loss", loss, prog_bar=True, logger=True)
    		return loss
	def test_step(self, batch, batch_idx):
    		input_ids = batch["input_ids"]
    		attention_mask = batch["attention_mask"]
    		labels = batch["labels"]
    		loss, outputs = self(input_ids, attention_mask, labels)
    		self.log("test_loss", loss, prog_bar=True, logger=True)
    		return loss
	def training_epoch_end(self, outputs):
    		labels = []
    		predictions = []
    		for output in outputs:
      			for out_labels in output["labels"].detach().cpu():
        			labels.append(out_labels)
      			for out_predictions in output["predictions"].detach().cpu():
        			predictions.append(out_predictions)
    		labels = torch.stack(labels).int()
    		predictions = torch.stack(predictions)
    		for i, name in enumerate(LABEL_COLUMNS):
      			class_roc_auc = auroc(predictions[:, i], labels[:, i])
      			self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
	def configure_optimizers(self):
    		optimizer = AdamW(self.parameters(), lr=2e-5)
    		scheduler = get_linear_schedule_with_warmup(
      			optimizer,
      			num_warmup_steps=self.n_warmup_steps,
      			num_training_steps=self.n_training_steps
    			)
    		return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=scheduler,interval='step'))
    		
steps_per_epoch= len(X_train) // BATCH_SIZE

total_training_steps = steps_per_epoch * N_EPOCHS

warmup_steps = total_training_steps // 5

warmup_steps, total_training_steps

model = Tagger(n_classes=len(DP_train[0]),n_warmup_steps=warmup_steps,n_training_steps=total_training_steps)

#_, predictions = model(sample_batch["input_ids"], sample_batch["attention_mask"])

#predictions

#criterion = nn.BCELoss()

#criterion(predictions, sample_batch["labels"])

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints",filename="best-checkpoint",save_top_k=1,verbose=True,monitor="val_loss",mode="min")

logger = TensorBoardLogger("lightning_logs", name="complaint")

#early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
#trainer = pl.Trainer(logger=logger,checkpoint_callback=checkpoint_callback,callbacks=[early_stopping_callback],max_epochs=N_EPOCHS,gpus=[5],progress_bar_refresh_rate=30)

trainer = pl.Trainer(logger=logger,checkpoint_callback=checkpoint_callback,max_epochs=N_EPOCHS,gpus=[5],progress_bar_refresh_rate=30)

trainer.fit(model, data_module)


trained_model = Tagger.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path,
  n_classes=len(DP_test[0])
)


test_dataset = Dataset(
	X_test,
	DP_test,
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT
)

text_batch = next(iter(DataLoader(test_dataset, batch_size=2, num_workers=2)))

actual = []
index = 0


for i in range(len(DP_test_actual)):
	actual.append(DP_test_actual[i].split("; "))
#print(actual)
#trainer.test(model, DataLoader(test_dataset, batch_size=1, num_workers=2))

#text = iter(DataLoader(test_dataset, batch_size = 1, num_workers=2))

#instance_text = []
#for i in range(len(test_df['event_text'])):
#	row = test_df.iloc[i]
#	instance_text.append(row.event_text)

#for i in text:
#	_, test_prediction = trained_model(i["input_ids"],i["attention_mask"])
#	for prediction in test_prediction:
#		print(prediction)
instance_score = []

index = 0
for i in iter(DataLoader(test_dataset, batch_size = 1, num_workers=2)):
	instance_score.append([])
	_, test_prediction = trained_model(i["input_ids"],i["attention_mask"])
	test_prediction = test_prediction.flatten().detach().numpy()
	for prediction in test_prediction:
		instance_score[index].append(prediction)
	index = index + 1

instance_score_reverse = [len(instance_score)] * 0
for i in range(len(instance_score)):
	instance_score_reverse.append([])
for i in range(len(instance_score)):
	instance_score_reverse[i] = [-x for x in instance_score[i]]
best5 = [len(instance_score)] * 0
for i in range(len(instance_score)):
	best5.append([])
for i in range(len(instance_score)):
	for j in list(np.asarray(instance_score_reverse[i]).argsort()[:5]):
		best5[i].append(mlb.classes_[j])
#print(best5)
d = {'Actual': actual}
data = pd.DataFrame(d)
data.to_csv('bert_actual_dff_25_epochs.csv',index=False)

d = {'Top 5': best5}
data = pd.DataFrame(d)
data.to_csv('bert_top5_dff_25_epochs.csv',index=False)
