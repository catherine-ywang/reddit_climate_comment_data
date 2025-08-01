# Reddit Climate Comment Dataset

## Overview

This repository contains supporting data and scripts for the [Reddit Climate Comment dataset](https://huggingface.co/datasets/cathw/reddit_climate_comment), which includes over 80,000 Reddit comments and replies focused on climate change, energy, and environmental sustainability.

The dataset was collected using the Reddit API on February 21-22, 2024. The dataset includes multiple subreddits including `Climate`, `Energy`, `RenewableEnergy`, `ClimateChange`, `Environment`, `Sustainability`, `ZeroWaste`, and `ClimateActionPlan`.

## Repository Contents

- `climate_comments.csv.zip`: The main structured dataset in CSV format
- `climate_comments.json.zip`: An alternative version of the dataset in JSON format
For the latest version and dataset loading instructions, visit the [Hugging Face dataset page](https://huggingface.co/datasets/cathw/reddit_climate_comment).

## Dataset Features

Each record in the dataset represents a Reddit post with associated metadata, comments, and replies. The main fields include:

- `id`: Unique post identifier
- `post_title`, `post_author`, `post_body`
- `post_url`, `post_pic`, `subreddit`, `post_timestamp`, `post_upvotes`, `post_permalink`
- `comments`: List of comments, each with:
  - `CommentID`, `CommentAuthor`, `CommentBody`, `CommentTimestamp`, `CommentUpvotes`, `CommentPermalink`
  - `replies`: List of replies under each comment, including:
    - `ReplyID`, `ReplyAuthor`, `ReplyBody`, `ReplyTimestamp`, `ReplyUpvotes`, `ReplyPermalink`
      
## Citation

```bibtex
@InProceedings{huggingface:dataset,
title = {Reddit Climate Comment},
author = {Yicheng (Catherine) Wang},
year = {2024}
}
