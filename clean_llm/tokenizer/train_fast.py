import os
import re
import time
import heapq

from collections import defaultdict
from collections import Counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Dict, Tuple, Union, Pattern



PAT_COMPILED = re.compile(r"\S+")


def pre_tokenize_and_count(
    args: Tuple[bytes, Dict[str, int], Union[Pattern, None]]
) -> Counter:
    """
    Pre-tokenize a chunk of bytes into token ids, handling special tokens.
    Returns a Counter of tokens.
    """
    chunk_bytes, special_token_to_id, delimiter_pattern_compiled = args
    chunk = chunk_bytes.decode("utf-8", errors="ignore")
    special_tokens_set = set(special_token_to_id.keys())
    
    words_list = []
    
    if delimiter_pattern_compiled:
        sub_chunks = delimiter_pattern_compiled.split(chunk)
    else:
        sub_chunks = [chunk]

    for sub_chunk in tqdm(sub_chunks, desc="Pre-tokenizing subchunks"):
        if not sub_chunk:
            continue

        if sub_chunk in special_tokens_set:
            token_id = special_token_to_id[sub_chunk]
            words_list.append((token_id,))
        else:
            for word_str in PAT_COMPILED.findall(sub_chunk):
                if word_str:
                    byte_sequence = word_str.encode("utf-8")
                    id_sequence = tuple(byte_sequence)
                    words_list.append(id_sequence)

    return Counter(words_list)


# Placeholder, should be replaced with your chunk splitter logic
def find_chunk_boundaries(f, num_chunks, marker_bytes):
    f.seek(0, 2)
    file_size = f.tell()
    chunk_size = file_size // num_chunks
    boundaries = [0]
    for i in range(1, num_chunks):
        boundaries.append(min(i * chunk_size, file_size))
    boundaries.append(file_size)
    return boundaries



class Node:
    """表示词内一个 token 节点，便于链表原地更新。"""
    def __init__(self, value, word_freq):
        self.value = value
        self.word_freq = word_freq  # 共享引用，节省内存
        self.prev = None
        self.next = None

class PQItem:
    """定义优先队列元素，实现自定义比较：频率优先，其次按字典序逆序。"""
    def __init__(self, freq, id_pair, byte_pair):
        self.freq = freq
        self.id_pair = id_pair
        self.byte_pair = byte_pair

    def __lt__(self, other):
        if self.freq != other.freq:
            return self.freq > other.freq  # 频率高的先出
        return self.byte_pair > other.byte_pair  # 字典序大的先出



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = 4,
    num_processes: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    before_pretokenization_time = time.time()

    # 1. Set up initial byte vocab and special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes
    byte_to_token_id = {v: k for k, v in vocab.items()}
    special_token_to_id = {
        token: byte_to_token_id[token.encode("utf-8")] for token in special_tokens
    }

    # 2. Prepare special tokens regex delimiter
    delimiter_pattern_compiled = None
    if special_tokens:
        # Sort by length descending for proper greedy match
        special_tokens_sorted = sorted(
            [t.encode("utf-8") for t in special_tokens], key=len, reverse=True
        )
        escaped_tokens = [re.escape(t.decode("utf-8")) for t in special_tokens_sorted]
        delimiter_re = "|".join(escaped_tokens)
        if delimiter_re:
            delimiter_pattern_compiled = re.compile(f"({delimiter_re})")

    # 3. Read file, split into chunks for multiprocessing
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, "<|endoftext|>".encode("utf-8")
        )

        chunk_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_args.append(
                (
                    chunk_bytes,
                    special_token_to_id,
                    delimiter_pattern_compiled,
                )
            )

    # 4. Determine number of processes
    processes_to_use = num_processes
    if processes_to_use is None:
        processes_to_use = min(cpu_count(), 8)
    processes_to_use = min(processes_to_use, len(chunk_args))

    elapsed = time.time() - before_pretokenization_time
    print(f"Time taken before pretokenization: {elapsed:.2f} seconds")

    # 5. Multiprocess: Counter aggregation
    all_word_freqs = Counter()
    start_time = time.time()
    with Pool(processes=processes_to_use) as pool:
        print(
            f"Starting pre-tokenization with {processes_to_use} processes on {len(chunk_args)} chunks..."
        )
        results_iterator = pool.imap_unordered(pre_tokenize_and_count, chunk_args)
        for chunk_counter in tqdm(
            results_iterator, total=len(chunk_args), desc="Processing chunks", leave=True
        ):
            all_word_freqs.update(chunk_counter)

    print(f"Pre-tokenization and initial counting time: {time.time() - start_time:.2f} seconds")

    ### Pre-tokenization 结束


    pair_to_nodes = defaultdict(set)
    for word_tuple, count in all_word_freqs.items():
        if len(word_tuple) < 2:
            continue

        # 所有链表节点共享 word_freq 引用，节省内存
        word_freq = {'count': count}

        head = Node(word_tuple[0], word_freq)
        prev_node = head
        for i in range(1, len(word_tuple)):
            curr_node = Node(word_tuple[i], word_freq)
            prev_node.next = curr_node
            curr_node.prev = prev_node

            pair = (prev_node.value, curr_node.value)
            pair_to_nodes[pair].add(prev_node)
            prev_node = curr_node


    pair_freqs = Counter()
    for pair, nodes in pair_to_nodes.items():
        # 某个 pair 的出现次数 = 其所有节点所对应 word 的词频累加
        pair_freqs[pair] = sum(node.word_freq['count'] for node in nodes)

    pq = [
        PQItem(freq, p, (vocab[p[0]], vocab[p[1]]))
        for p, freq in pair_freqs.items()
    ]
    heapq.heapify(pq)

    ### BPE 开始
    merges = []
    num_merges = vocab_size - len(vocab)
    pbar = tqdm(total=num_merges, desc="Performing BPE merges")
    start_time = time.time()

    for _ in range(num_merges):
        if not pq:
            break

        # 取出频率最高的 pair，处理优先队列惰性删除的过期元素
        best_pair = None
        while pq:
            item = heapq.heappop(pq)
            if item.id_pair not in pair_freqs:
                continue  # 已经被合并删除
            if pair_freqs[item.id_pair] == item.freq:
                best_pair = item.id_pair
                break

        if best_pair is None:
            break

        p1, p2 = best_pair

        # 合成新 token，添加到 merges/vocab
        new_token_id = len(vocab)
        merged_token_bytes = vocab[p1] + vocab[p2]
        merges.append((vocab[p1], vocab[p2]))
        vocab[new_token_id] = merged_token_bytes

        # 逐个更新包含改 pair 的词
        nodes_to_process = list(pair_to_nodes[best_pair])
        for node1 in nodes_to_process:
            node2 = node1.next
            if node2 is None:
                continue
            word_freq = node1.word_freq['count']

            # 更新左侧相邻 pair 的频率及映射关系
            if node1.prev:
                left = node1.prev
                old_left_pair = (left.value, node1.value)
                pair_freqs[old_left_pair] -= word_freq
                heapq.heappush(pq, PQItem(pair_freqs[old_left_pair], old_left_pair, (vocab[old_left_pair[0]], vocab[old_left_pair[1]])))

                pair_to_nodes[old_left_pair].discard(left)
                new_left_pair = (left.value, new_token_id)
                pair_to_nodes[new_left_pair].add(left)
                pair_freqs[new_left_pair] += word_freq
                heapq.heappush(pq, PQItem(pair_freqs[new_left_pair], new_left_pair, (vocab[new_left_pair[0]], vocab[new_left_pair[1]])))

            # 更新右侧相邻 pair 的频率及映射关系
            if node2.next:
                right = node2.next
                old_right_pair = (node2.value, right.value)
                pair_freqs[old_right_pair] -= word_freq
                heapq.heappush(pq, PQItem(pair_freqs[old_right_pair], old_right_pair, (vocab[old_right_pair[0]], vocab[old_right_pair[1]])))

                new_right_pair = (new_token_id, right.value)
                pair_to_nodes[old_right_pair].discard(node2)
                pair_to_nodes[new_right_pair].add(node1)
                pair_freqs[new_right_pair] += word_freq
                heapq.heappush(pq, PQItem(pair_freqs[new_right_pair], new_right_pair, (vocab[new_right_pair[0]], vocab[new_right_pair[1]])))

            # 链表合并：node1、node2合成 new_token_id
            node1.value = new_token_id
            node1.next = node2.next
            if node2.next:
                node2.next.prev = node1

        # 删除被合并 pair 的所有统计
        del pair_freqs[best_pair]
        del pair_to_nodes[best_pair]

        pbar.update(1)

    end_time = time.time()
    print(f"Merge time: {end_time - start_time:.2f} seconds")
    pbar.close()

    
    return vocab, merges
