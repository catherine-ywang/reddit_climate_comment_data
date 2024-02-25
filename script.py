import csv
import json
import os
import logging
import datasets
import zipfile
from csvtojsontransformer import CSVtoJSONTransformer

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2024}
}
"""

# TODO: Add description of the dataset
_DESCRIPTION = """\
    This dataset was generated using a custom script (`script.py`) available in the following GitHub repository: [GitHub Repository](https://github.com/your-username/your-repository). 
To generate the dataset, clone the repository and follow the instructions provided in the repository's README file.
"""

# TODO: Add a link to an official homepage for the dataset
_HOMEPAGE = ""

# TODO: Add the licence for the dataset if available
_LICENSE = ""

class NewDataset(datasets.GeneratorBasedBuilder):
    # TODO: Add link to the official dataset URLs
    _URLS = {
        "train": "https://raw.githubusercontent.com/catherine-ywang/reddit_climate_comment_data/main/climate_comments.csv.zip"
    }

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features({
            "Posts": datasets.Sequence({
                "PostID": datasets.Value("string"),
                "PostTitle": datasets.Value("string"),
                "PostAuthor": datasets.Value("string"),
                "PostBody": datasets.Value("string"),
                "PostUrl": datasets.Value("string"),
                "PostPic": datasets.Value("string"),
                "Subreddit": datasets.Value("string"),
                "PostTimestamp": datasets.Value("string"),
                "PostUpvotes": datasets.Value("int32"),
                "PostPermalink": datasets.Value("string"),
                "Comments": datasets.Sequence({
                    "CommentID": datasets.Value("string"),
                    "CommentAuthor": datasets.Value("string"),
                    "CommentBody": datasets.Value("string"),
                    "CommentTimestamp": datasets.Value("string"),
                    "CommentUpvotes": datasets.Value("int32"),
                    "CommentPermalink": datasets.Value("string"),
                    "Replies": datasets.Sequence({
                        "ReplyID": datasets.Value("string"),
                        "ReplyAuthor": datasets.Value("string"),
                        "ReplyBody": datasets.Value("string"),
                        "ReplyTimestamp": datasets.Value("string"),
                        "ReplyUpvotes": datasets.Value("int32"),
                        "ReplyPermalink": datasets.Value("string"),
                    })
                })
            })
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = self._URLS  # Get the URLs directly from _URLS
        data_dir = dl_manager.download_and_extract(urls)
        # Since the downloaded file is a zip, we need to extract it
        zip_file = os.path.join(data_dir, "climate_comments.csv.zip")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        # Return the path to the extracted CSV file
        csv_file = os.path.join(data_dir, "climate_comments.csv")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": csv_file,
                    "split": "train",
                },
            ),
        ]



    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            csv_reader = csv.DictReader(f)
            for idx, row in enumerate(csv_reader):
                post_info = {
                    "PostID": row["PostID"],
                    "PostTitle": row["PostTitle"],
                    "PostAuthor": row["PostAuthor"],
                    "PostBody": row["PostBody"],
                    "PostUrl": row["PostUrl"],
                    "PostPic": row["PostPic"],
                    "Subreddit": row["Subreddit"],
                    "PostTimestamp": row["PostTimestamp"],
                    "PostUpvotes": int(row["PostUpvotes"]),
                    "PostPermalink": row["PostPermalink"],
                    "Comments": []
                }

                # Iterate over comments
                for comment_key in row:
                    if comment_key.startswith("Comment") and row[comment_key] != "":
                        comment = {
                            "CommentID": row[f"{comment_key}ID"],
                            "CommentAuthor": row[f"{comment_key}Author"],
                            "CommentBody": row[f"{comment_key}Body"],
                            "CommentTimestamp": row[f"{comment_key}Timestamp"],
                            "CommentUpvotes": int(row[f"{comment_key}Upvotes"]),
                            "CommentPermalink": row[f"{comment_key}Permalink"],
                            "Replies": []
                        }

                        # Iterate over replies
                        for reply_key in row:
                            if reply_key.startswith(f"{comment_key}Reply") and row[reply_key] != "":
                                reply = {
                                    "ReplyID": row[f"{reply_key}ID"],
                                    "ReplyAuthor": row[f"{reply_key}Author"],
                                    "ReplyBody": row[f"{reply_key}Body"],
                                    "ReplyTimestamp": row[f"{reply_key}Timestamp"],
                                    "ReplyUpvotes": int(row[f"{reply_key}Upvotes"]),
                                    "ReplyPermalink": row[f"{reply_key}Permalink"]
                                }
                                comment["Replies"].append(reply)

                        post_info["Comments"].append(comment)

                yield idx, post_info

