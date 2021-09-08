# DATASET_VOCAB
This folder container method to sample some clients' train data to generate vocab in nlp application.
run `sample_build_vocab.py` to get vocab, params including dataset name, clients select ratio, vocab limited size. 

example:
    `python sample_build_vocab.py --dataset "sent140" --data_select_ratio 0.25 --vocab_limit_size 30000
`
