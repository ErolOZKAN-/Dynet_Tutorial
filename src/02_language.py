import dynet as dy

xsent = False
HIDDEN_SIZE = 8
ITERATIONS = 2000
FEATURE_VECTOR = 2
m = dy.Model()
trainer = dy.SimpleSGDTrainer(m)

W = m.add_parameters((HIDDEN_SIZE, FEATURE_VECTOR))
b = m.add_parameters(HIDDEN_SIZE)
V = m.add_parameters((FEATURE_VECTOR, HIDDEN_SIZE))
a = m.add_parameters(FEATURE_VECTOR)

x = dy.vecInput(FEATURE_VECTOR)
y = dy.vecInput(FEATURE_VECTOR)
h = dy.tanh((W * x) + b)

y_pred = (V * h) + a
loss = dy.squared_distance(y_pred, y)
T = 1
F = -1

for iter in range(ITERATIONS):
    mloss = 0.0
    for mi in range(4):
        x1 = mi % 2
        x2 = (mi // 2) % 2

        input_vector = [T if x1 else F, T if x2 else F]
        output_vector = [T if x1 else F, T if x2 else F]
        print(input_vector, output_vector)

        x.set(input_vector)
        y.set(output_vector)

        current_loss = loss.value()
        print(current_loss)
        mloss += current_loss

        loss.backward()
        trainer.update()

    mloss /= 4.
    print("loss: %0.9f" % mloss)

print("OUTPUTs\n\n\n")
x.set([T, F])
print("TF", y_pred.value())
x.set([F, F])
print("FF", y_pred.value())
x.set([T, T])
print("TT", y_pred.value())
x.set([F, T])
print("FT", y_pred.value())
