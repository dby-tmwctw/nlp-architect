import json

import pandas as pd

from nlp_architect.utils.generic import normalize, balance

import csv

good_columns = ["overall", "reviewText", "summary"]


def review_to_sentiment(review):
    # Review is coming in as overall (the rating, reviewText, and summary)
    # this then cleans the summary and review and gives it a positive or negative value
    norm_text = normalize(review[1])
    review_sent = ["neutral", norm_text]
    score = float(review[0])
    if score > 0:
        review_sent = ["positive", norm_text]
    elif score < 0:
        review_sent = ["negative", norm_text]

    return review_sent


class Customized_Reviews(object):
    """
    Takes CSV file from the NLP input team and process it into usable object by LTSM model
    """

    def __init__(self, review_file, run_balance=True):
        self.run_balance = run_balance

        print("Parsing and processing json file")
        data = []

        # with open(review_file, "r") as f:
        #     for line in f:
        #         data_line = json.loads(line)
        #         selected_row = []
        #         for item in good_columns:
        #             selected_row.append(data_line[item])
        #         # as we read in, clean
        #         data.append(review_to_sentiment(selected_row))
        
        with open(review_file, newline='\n') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='\"')
            for row in reader:
                data.append(review_to_sentiment(row))

        # Not sure how to easily balance outside of pandas...but should replace eventually
        self.amazon = pd.DataFrame(data, columns=["Sentiment", "clean_text"])
        self.all_text = self.amazon["clean_text"]
        self.labels_0 = pd.get_dummies(self.amazon["Sentiment"])
        self.labels = self.labels_0.values
        self.text = self.amazon["clean_text"].values

    def process(self):
        self.amazon = self.amazon[self.amazon["Sentiment"].isin(["positive", "negative"])]

        if self.run_balance:
            # balance it out
            self.amazon = balance(self.amazon)

        print("Sample Data")
        print(self.amazon[["Sentiment", "clean_text"]].head())

        # mapping of the labels with dummies (has headers)
        self.labels_0 = pd.get_dummies(self.amazon["Sentiment"])
        self.labels = self.labels_0.values
        self.text = self.amazon["clean_text"].values