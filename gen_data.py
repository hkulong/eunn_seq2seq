import os
import shutil
import random

max_len = 10

def generate_dataset(root, name, size):
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)

    # generate data file
    input_path = os.path.join(path, 'input.txt')
    output_path = os.path.join(path, 'output.txt')
    fin = open(input_path,'w')
    fout = open(output_path,'w')
    for _ in range(size):
        length = random.randint(1, max_len)
        seq = []
        for _ in range(length):
            seq.append(str(random.randint(0, 9)))
        fin.write(" ".join(seq))
        fin.write('\n')
        fout.write(" ".join(reversed(seq)))
        fout.write('\n')

    # generate vocabulary
    vocab = os.path.join(path, 'vocab.txt')
    with open(vocab, 'w') as fout:
        fout.write('<PAD>\n<UNK>\n<GO>\n<EOS>\n')
        fout.write("\n".join([str(i) for i in range(10)]))

if __name__ == '__main__':
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    toy_dir = os.path.join(data_dir, 'reverse_{}'.format(max_len))
    if not os.path.exists(toy_dir):
        os.mkdir(toy_dir)

    generate_dataset(toy_dir, 'train', 10000)
    generate_dataset(toy_dir, 'dev', 1000)
    generate_dataset(toy_dir, 'test', 1000)
