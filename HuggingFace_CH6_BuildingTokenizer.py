from datasets import load_dataset

dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

# with open("wikitext-2.txt", "w", encoding="utf-8") as f:
#     for i in range(len(dataset)):
#         f.write(dataset[i]["text"] + "\n")

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

#tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#print(tokenizer.pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

# pre_tokenizer = pre_tokenizers.WhitespaceSplit()
# print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
print(pre_tokenizer.pre_tokenize_str("Let's test my pre-tokenizer."))

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

