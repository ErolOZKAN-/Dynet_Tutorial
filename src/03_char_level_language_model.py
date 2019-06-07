import string

import dynet as dy

ITERATIONS = 1

m = dy.Model()
trainer = dy.SimpleSGDTrainer(m)


def read_file(filename):
    with open(filename, encoding="utf8") as file:
        text = file.read()
    return text


def get_vector(index, length):
    vector = [0] * length
    vector[index] = 1
    return vector


def preprocess(text):
    text.replace('SPEECH', ' ')
    text = text.lower()
    tokens = text.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    return tokens


text = read_file('../data/trumpspeeches.txt')
text = preprocess(text)
text = " ".join(text)
char_list = []

for id, char in enumerate(text):
    str = text[id:id + 33]
    char_list.append(str)

max = 0

all_inputs = []
all_outputs = []
for elem in char_list:
    input = []
    for id, char in enumerate(elem):
        if id == 32:
            order = ord(char)
            all_outputs.append(get_vector(order, 256))
            continue
        order = ord(char)
        binary_count = list(bin(order)[2:])
        for ii in range(len(binary_count), 8):
            binary_count.insert(0, "0")
        input += binary_count
    all_inputs.append(input)

HIDDEN_SIZE = 64
INPUT_VEC_SIZE = 256
OUTPUT_VECTOR_SIZE = 256

W = m.add_parameters((HIDDEN_SIZE, INPUT_VEC_SIZE))
b = m.add_parameters(HIDDEN_SIZE)
V = m.add_parameters((OUTPUT_VECTOR_SIZE, HIDDEN_SIZE))
a = m.add_parameters(OUTPUT_VECTOR_SIZE)

x = dy.vecInput(INPUT_VEC_SIZE)
y = dy.vecInput(OUTPUT_VECTOR_SIZE)
h = dy.tanh((W * x) + b)

y_pred = (V * h) + a
loss = dy.squared_distance(y_pred, y)

for iter in range(ITERATIONS):
    mloss = 0.0
    seen_instances = 0
    for id, input in enumerate(all_inputs):
        x.set(input)
        y.set(all_outputs[id])
        seen_instances += 1

        mloss += loss.value()
        loss.backward()
        trainer.update()

        if (seen_instances > 1 and seen_instances % 1000 == 0):
            print(seen_instances, "/", len(all_inputs), "***average loss is:", mloss / seen_instances)

    print("loss: %0.9f" % mloss)
