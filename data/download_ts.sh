mkdir -p TinyStories/txt
cd TinyStories/txt

# wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
# wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# 如果你链接huggingface.co失败，可以尝试使用hf-mirror.com，具体操作如下：
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

mv TinyStoriesV2-GPT4-train.txt train.txt
mv TinyStoriesV2-GPT4-valid.txt valid.txt