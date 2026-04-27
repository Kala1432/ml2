import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split as split
from sklearn.naive_bayes import GaussianNB

f = fetch_olivetti_faces(shuffle=True, random_state=42)
xt, xv, yt, yv = split(f.data, f.target, test_size=0.2, random_state=42)
nb = GaussianNB().fit(xt, yt)

print(f"Accuracy: {nb.score(xv, yv):.2%}")

yp = nb.predict(xv)
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(xv[i].reshape(64, 64), cmap='gray')
    plt.title(f"P: {yp[i]}"), plt.axis('off')
plt.show()