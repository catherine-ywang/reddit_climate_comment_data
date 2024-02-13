import os
import json
import csv
from datasets import GeneratorBasedBuilder, DatasetInfo, Features, Value, Sequence, Split, SplitGenerator

class SuperGlue(GeneratorBasedBuilder):
    def __init__(self, **kwargs):
        # Initialize any necessary variables
        super().__init__(**kwargs)

    def _info(self):
        # Define features based on your dataset structure
        features = Features({
            "Subreddit": Value("string"),
            "Posts": Sequence({
                "PostID": Value("int32"),
                "PostTitle": Value("string"),
                "Comments": Sequence({
                    "CommentID": Value("string"),
                    "Author": Value("string"),
                    "CommentBody": Value("string"),
                    "Timestamp": Value("string"),
                    "Upvotes": Value("int32"),
                    "NumberofReplies": Value("int32"),
                }),
            }),
        })

        # Return DatasetInfo
        return DatasetInfo(
            description="Your dataset description here",
            features=features,
            homepage="URL to dataset homepage",
            citation="Your dataset citation here",
        )
    def _split_generators(self, dl_manager):
        # Set the path to your local CSV file
        json_path = "https://github.com/catherine-ywang/reddit_climate_comment_data/blob/main/climate_data.json"

        # Return a SplitGenerator with the local CSV file path
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={
                    "json_path": json_path,
                },
            ),
        ]
    def _generate_examples(self, json_path):
        # Generate examples from your dataset
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            for idx, row in enumerate(data):
                subreddit = row["Subreddit"]
                posts = []
                for post in row["Posts"]:
                    post_id = post["PostID"]
                    post_title = post["PostTitle"]
                    comments = []
                    for comment in post["Comments"]:
                        comment_id = comment["CommentID"]
                        author = comment["Author"]
                        comment_body = comment["CommentBody"]
                        timestamp = comment["Timestamp"]
                        upvotes = comment["Upvotes"]
                        number_of_replies = comment["NumberofReplies"]
                        comments.append({
                            "CommentID": comment_id,
                            "Author": author,
                            "CommentBody": comment_body,
                            "Timestamp": timestamp,
                            "Upvotes": upvotes,
                            "NumberofReplies": number_of_replies
                        })
                    posts.append({
                        "PostID": post_id,
                        "PostTitle": post_title,
                        "Comments": comments
                    })
                yield idx, {
                    "Subreddit": subreddit,
                    "Posts": posts
                }

    




