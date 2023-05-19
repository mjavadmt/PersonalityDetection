from data_collection.script import main_collector
from data_cleaning.script.script import main_cleaner

main_collector()
main_cleaner()

from tokenizing.tokenizer import word_tokenizer, sentence_tokenizer

word_tokenizer()
sentence_tokenizer()
