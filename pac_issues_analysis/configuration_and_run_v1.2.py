# -*- coding: utf-8 -*-
import os

os.chdir(os.path.dirname(os.path.realpath("__file__")))  # sets the directory
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from multiprocessing import Process, Queue
import multiprocessing as mp

# tqdm(pd, ncols=30)

# ==================
# Basic processing imports
# ==================

import re
import string as s
import nltk
import nltk.data
from collections import defaultdict, Counter
from resources.text_cleaning.cleaning_functions import Preprocessing

# ==================
# Snorkel Packages
# ==================
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LabelingFunction
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis

# ==================
# Transformer Text Summary
# ==================
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------


class TopicModelingPreprocessing:
    def __init__(self, topic_path):
        self.feat_dict = {
            "keywords": "any_of_these_words",
            "not_match": "none_of_these_words",
            "required": "exact_phrases",
        }
        xls = pd.ExcelFile(topic_path)
        for item in self.feat_dict.keys():
            self.feat_dict[item] = pd.read_excel(xls, self.feat_dict[item])

    # Using Regex for pattern matching LF functions
    # Assign labels based on the keywords matching
    def regex_topic_match(self, x, keywords, notmatch, required, label):
        ABSTAIN = -1
        NO_TOPIC = -2
        s = str(x)

        # check if there are any "NOne of these words" in the string
        for word in notmatch:
            # use exact match for none of these words
            if re.search(r"\b" + word + r"\b", s) in notmatch:
                return ABSTAIN

        # check if there are any "Any of these words" in the string
        for word in keywords:
            # this also is more flexible -- allowing for extra characters at the end to count
            # aka pizza  and pizzas are both True
            if re.search(rf"\b{word}e?s?\b|\b{word}ing?\b", s, flags=re.I):
                return label

        # check if there are any "Exact match" in the string
        for word in required:
            # exact match needed to be True
            if re.search(r"\b" + word + r"\b", s):
                return label

        return NO_TOPIC

    # Create loop to auto generate class for each topic LF function
    def topic_match_lf(self, keywords, topics, i):
        return LabelingFunction(
            name=f"keyword_{keywords[0]}",  # test if it words if this is erased or changed to topics as names
            f=self.regex_topic_match,
            resources=dict(
                keywords=self.feat_dict["keywords"][topics[i]],
                notmatch=self.feat_dict["not_match"][topics[i]],
                required=self.feat_dict["required"][topics[i]],
                label=self.feat_dict["labels"][topics[i]],
            ),
        )

    def topic_labeling_preprocessing(self):
        topics = self.feat_dict["keywords"]["topic"].values.tolist()
        label_dict = {topics[k]: k for k in range(len(topics))}

        for item in self.feat_dict.keys():
            if item == "required":
                flag = True
            self.feat_dict[item] = clean_keywords(
                self.feat_dict[item], topics, required_word=True
            )
        self.feat_dict["labels"] = label_dict

        lfs = []
        for i in range(len(topics)):
            print(f"topic {i+1}:     ", topics[i])
            print(f"keyword {i+1}:   ", self.feat_dict["keywords"][topics[i]])
            print(f"not_match {i+1}: ", self.feat_dict["not_match"][topics[i]])
            print(f"required {i+1}:  ", self.feat_dict["required"][topics[i]])
            print(f"label {i+1}:     ", self.feat_dict["labels"][topics[i]])
            lfs.append(self.topic_match_lf([topics[i]], topics, i))
        return topics, lfs


# ---------------------------------------------------
# Topic Training and Paraphrasing
# ---------------------------------------------------


class TopicTrainingAndParaphrasing:
    def __init__(self, input_path, topics, lfs):
        self.label_model = LabelModel(cardinality=len(topics), verbose=True)
        self.df = pd.read_csv(input_path, encoding="utf-8")
        self.lfs = lfs

    def topic_train(self, how_many_rows_to_train=None, output_name="tss.csv"):
        df_train = self.df.copy()

        # create lfs for each topic class
        applier = PandasLFApplier(lfs=self.lfs)
        L_train = applier.apply(df=df_train)

        print("Start topic training")
        self.label_model.fit(
            L_train, n_epochs=500, log_freq=50, seed=123
        )  # original n_epochs: 500 but it was changed because of fast execution
        df_train["class_simple"] = self.label_model.predict(
            L=L_train, tie_break_policy="abstain"
        )
        df_train = df_train[df_train["class_simple"] != "-2"]
        df_train["topics"] = (
            df_train["class_simple"].apply(str).apply(assign_topics, args=(topics,))
        )
        df_train = df_train[df_train["topics"] != "NOT_RELEVANT"]
        print(LFAnalysis(L=L_train, lfs=self.lfs).lf_summary())
        return df_train

    def para_train(self):
        print("Queue made")
        input_queue = Queue(maxsize=10000)
        output_queue = Queue(maxsize=10000)
        print("Making processes")
        for i in range(6):
            p = Process(target=worker, args=(input_queue, output_queue), daemon=True)
            p.start()

        print("Adding sentence to queue and dequeing ")

        count = 0
        self.df["paraphrase"] = ""
        for i, row in self.df.iterrows():
            input_queue.put((count, row["text"]))
            count += 1
            if count == 50:
                break

        paraphrases = dict()
        remaining = count
        while remaining > 0:
            item, paraphrase = output_queue.get()
            paraphrases[item] = paraphrase
            print(remaining, "/", count)
            remaining -= 1

        for i, row in self.df.iterrows():
            if i in paraphrases:
                print(i)
                self.df.at[i, "paraphrase"] = paraphrases[i]

        self.df.reset_index(drop=True, inplace=True)  # RESET EXTRA_ID FIRST
        self.df["extra_id"] = self.df.index
        dfe = self.df[["extra_id", "paraphrase"]]  # create df to explode
        dfe = dfe.explode("paraphrase")
        dfe = dfe[~pd.isnull(dfe["paraphrase"])]
        del self.df["paraphrase"]  # before merging - erase text columns on df
        df = pd.merge(
            self.df, dfe, how="right", on="extra_id"
        )  # join exploded df to original data
        df = df[pd.notnull(df["paraphrase"])]
        return df


def worker(input_queue, output_queue):
    model = T5TextParaphraseModel()
    model.paraphrase_text(input_queue, output_queue)


class T5TextParaphraseModel:
    def __init__(self):
        self.set_seed(42)
        self.tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "ramsrigouthamg/t5_paraphraser"
        ).to(self.device)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def paraphrase_text(self, input_queue, output_queue):
        while True:
            count, sentence = input_queue.get()

            encoding = self.tokenizer.encode_plus(
                sentence, padding=True, return_tensors="pt"
            )
            input_ids, attention_masks = (
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            )
            beam_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                do_sample=True,
                max_length=256,
                top_k=120,
                top_p=0.98,
                early_stopping=True,
                num_return_sequences=10,
            )

            final_outputs = []
            for beam_output in beam_outputs:
                sent = self.tokenizer.decode(
                    beam_output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                final_outputs.append(sent)
            output_queue.put((count, final_outputs))


# ---------------------------------------------------
# Helper Functions
# 1. clean topic keywords
# 2. create regex function for labeling functions
# 3. translate between topics and their labels
# 4. output.txt final result into csv file
# ---------------------------------------------------


def clean_keywords(args, topics, required_word=False):
    if not required_word:
        for i in range(1, len(args.columns)):
            args.iloc[:, i] = (
                args.iloc[:, i].apply(str).apply(Preprocessing().text_clean)
            )
    args = args.drop(["topic"], axis=1).values.tolist()
    args = [[str(x).rstrip() for x in y if str(x) != "nan"] for y in args]
    if not required_word:
        return [[x for x in y if len(x) > 1] for y in args]
    return dict(zip(topics, args))


def assign_topics(x, topics):
    if x == "-1":
        return "NOT_RELEVANT"
    elif x.isdigit() and int(x) != -1:
        return topics[int(x)]
    else:
        return ""


def merge(topic_df, sent_df, topics, analyze_date_time=False):
    final_df = pd.merge(topic_df, sent_df, how="inner", on="extra_id")
    final_df = final_df.drop_duplicates("text")
    final_df.reset_index(drop=True, inplace=True)
    final_df["extra_id"] = final_df.index
    print(final_df.shape)
    output = final_df.copy()

    output["class_simple_pred_label_acc_combined"] = (
        output["class_simple_pred_label"] + output["class_simple_pred_label2"]
    )
    output["topic"] = output["class_simple_pred_label_acc_combined"].apply(
        self.assign_topics, args=(topics,)
    )  # Convert numeric label to strings

    output = output[output["topic"] != "NOT_RELEVANT"]
    output = output[pd.notnull(output["topic"])]

    return output


if __name__ == "__main__":
    # File with all reviews that we want to analyze:
    root = "/Users/shiyishen/PycharmProjects/pac_issues"

    # input path
    input_path = os.path.join(
        root, "inputs/pac_fund_issues.csv"
    )  # reviews column needs to be named "text"

    # TOPIC MODELING TRAINING FILE(S)
    topic_path = os.path.join(root, "inputs/pac_fund_topics_shiyi.xlsx")

    # TOPIC TRAINING FILE(S)
    snorkel_training_df = os.path.join(
        root, "inputs/snorkel_pac_fund_topic_traning_df.csv"
    )


    # Folder for exporting results and train/valid.txt
    # make folder to save files into
    ext_path = os.path.join(root, "outputs/")
    if not os.path.exists(ext_path):
        os.makedirs(ext_path)

    # FINAL OUTPUT FILE
    final_output = os.path.join(ext_path, "2021_pac_fund_issues_classified.csv")
    bl = TopicModelingPreprocessing(topic_path)
    topics, lfs = bl.topic_labeling_preprocessing()
    topic_para = TopicTrainingAndParaphrasing(input_path, topics, lfs)
    # df = topic_para.para_train()
    df = topic_para.topic_train()
    df.to_csv("snorkel_topics_demo.csv")

