import numpy as np
import matplotlib.pyplot as plt


#modificati acest path local astfel incat sa puteti citi datele corect
data_path = "C:/Users/Student/Desktop/Examen/subiect_examen_2022023 (1)/subiect_examen_2022023/data/"
print(data_path)
#CIFRE
#incarca datele pentru cifre
exemple_cifre = np.loadtxt(data_path + "train_images.txt").astype(np.uint8)
etichete_cifre = np.loadtxt(data_path + "train_labels.txt").astype(np.int8)

print(exemple_cifre.shape)
print(etichete_cifre.shape)

#ploteaza primele 100 de cifre
nr_imagini = 10

etichete_100 = etichete_cifre[:nr_imagini**2]
print("Etichetele exemplelor sunt:", np.reshape(etichete_100,(nr_imagini,nr_imagini)))

# plt.figure(figsize=(5,5))
# for i in range(nr_imagini**2):
#     plt.subplot(nr_imagini,nr_imagini,i+1)
#     plt.axis('off')
#     plt.imshow(np.reshape(exemple_cifre[i,:],(28,28)),cmap = "gray")
# plt.show()

X_train = np.array(exemple_cifre[:700])
y_train = np.array(etichete_cifre[:700])

X_valid = np.array(exemple_cifre[700:900])
y_valid = np.array(etichete_cifre[700:900])

X_test = np.array(exemple_cifre[900:])
y_test = np.array(etichete_cifre[900:])

print(len(X_train), len(X_valid), len(X_test))


i = 0
index = np.ravel(np.where(y_train == i))
print("Imaginile de antrenare care contin clasa " + str(i) + " au indecsii:")
print(index)

medie = np.zeros(shape=(10, 784))
medie_label = [i for i in range(10)]
numar_cifre = np.zeros(10)
for i in y_train:
    numar_cifre[i] += 1


for i, nr in enumerate(X_train):
    for j in range(784):
        medie[y_train[i]][j] += X_train[i][j]

for i in range(10):
    for j in range(784):
        medie[i][j] = int(medie[i][j] / numar_cifre[i])

prototip = []
for i in range(10):
    prototip_i = medie[i].reshape(28, 28)
    prototip.append(prototip_i)
    plt.imshow(prototip_i, cmap="gray")
    plt.show()

from sklearn.neighbors import KNeighborsClassifier

model_euclid = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
model_manhattan = KNeighborsClassifier(n_neighbors=1, metric="manhattan")

model_euclid.fit(medie, medie_label)
model_manhattan.fit(medie, medie_label)

pred_euclid = model_euclid.predict(X_valid)
pred_manhattan = model_manhattan.predict(X_valid)

from sklearn.metrics import accuracy_score, confusion_matrix

print(f"Acuratete cu distanta euclideana: {accuracy_score(pred_euclid, y_valid)}")
print(f"Acuratete cu distanta manhattan: {accuracy_score(pred_manhattan, y_valid)}")

# Acuratete cu distanta euclideana: 0.77
# Acuratete cu distanta manhattan: 0.58

pred = model_euclid.predict(X_test)

print(f"Acuratete: {accuracy_score(pred, y_test)}")

print(f"Matricea de confuzie: {confusion_matrix(pred, y_test)}")


